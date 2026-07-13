#!/usr/bin/env python3
"""Pull a class-balanced Kinetics-400 subset from the CVDF mirror.

CVDF (https://github.com/cvdfoundation/kinetics-dataset) hosts the official
Kinetics-400 train split as 242 tar.gz shards on S3
(``https://s3.amazonaws.com/kinetics/400/train/part_<0..241>.tar.gz``). Each
shard is an arbitrary, class-mixed slice of the ~246k train clips (~1.6GB
each, ~400GB total) — there is no manifest mapping a class to a shard, so the
only way to know what a shard contains is to download and extract it.

This script keeps only the clips belonging to a fixed set of target classes
(the first ``--num-classes`` Kinetics-400 labels in alphabetical order,
matching ``src.data.kinetics_index``'s own ``--max-classes`` truncation),
and deletes everything else immediately so pod disk usage stays bounded. It
stops as soon as every target class has ``--min-per-class`` clips, so in
practice it only needs a fraction of the full ~400GB train split.

Downloads for the next ``--parallel-downloads`` shards are pipelined ahead
of extraction/filtering (a single wget stream tends to run well under a
pod's actual bandwidth, and extraction/filtering is comparatively fast), so
several shards' tars can be in flight or already on disk while the CPU-bound
extract-and-filter step works through them in order.

Run this on the compute node (RunPod pod, not the local machine) — this is
a large, network- and disk-heavy step, and raw video is intentionally never
committed to the repo (see .gitignore).

Typical use:

    uv run python -m scripts.fetch_kinetics_cvdf_subset --plan-only
    uv run python -m scripts.fetch_kinetics_cvdf_subset \\
        --output-dir data/kinetics/raw/c50 --work-dir /workspace/kinetics_cvdf

Then build the normalized subset index as usual:

    uv run python -m src.data.kinetics_index \\
        --video-root data/kinetics/raw/c50 \\
        --output-dir data/kinetics/subsets/c50_train100_heldout30 \\
        --subset-id c50_train100_heldout30 \\
        --max-classes 50 --train-per-class 100 --heldout-per-class 30 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path

TRAIN_CSV_URL = "https://s3.amazonaws.com/kinetics/400/annotations/train.csv"
TRAIN_PATHS_URL = "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt"


@dataclass
class ClipRef:
    filename: str
    label: str


@dataclass
class FetchState:
    processed_shards: set[int] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "FetchState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(processed_shards=set(data.get("processed_shards", [])))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"processed_shards": sorted(self.processed_shards)}), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/kinetics/raw/c50"),
        help="Where selected clips land, as <class>/<video>.mp4. Default: data/kinetics/raw/c50",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("data/kinetics/_cvdf_scratch"),
        help="Scratch space for downloaded tars, extraction, and resume state. "
        "Safe to delete once the run finishes.",
    )
    parser.add_argument("--num-classes", type=int, default=50,
                         help="Number of Kinetics-400 classes to target, taken "
                         "alphabetically. Default: 50")
    parser.add_argument("--min-per-class", type=int, default=140,
                         help="Stop once every target class has at least this many "
                         "clips (130 are needed for c50_train100_heldout30; the "
                         "extra 10 is slack for undecodable clips). Default: 140")
    parser.add_argument("--max-shards", type=int, default=242,
                         help="Safety cap on how many of the 242 train shards to "
                         "pull before giving up on any still-short classes.")
    parser.add_argument("--plan-only", action="store_true",
                         help="Only download the (tiny) annotations CSV, report the "
                         "selected classes and an estimated shard count, then exit. "
                         "No video download.")
    parser.add_argument("--parallel-downloads", type=int, default=6,
                         help="Number of shard tar.gz downloads to keep in flight "
                         "ahead of extraction. Downloads are the bottleneck (single "
                         "wget streams run well under typical pod bandwidth), so "
                         "this pipelines the next N shards while the current one is "
                         "being extracted/filtered. Default: 6")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def ensure_wget() -> None:
    if shutil.which("wget") is None:
        raise RuntimeError(
            "wget not found on PATH. Install it first (e.g. `apt-get install -y wget`) "
            "— this script shells out to it for resumable downloads."
        )


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["wget", "-q", "-c", url, "-O", str(dest)])


def load_annotations(csv_path: Path) -> list[tuple[str, str]]:
    """Return (label, filename) pairs for every train clip."""
    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            start = int(row["time_start"])
            end = int(row["time_end"])
            filename = f"{row['youtube_id']}_{start:06d}_{end:06d}.mp4"
            rows.append((row["label"], filename))
    return rows


def select_target_classes(rows: list[tuple[str, str]], num_classes: int) -> list[str]:
    return sorted({label for label, _ in rows})[:num_classes]


def existing_counts(output_dir: Path, target_classes: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for label in target_classes:
        class_dir = output_dir / label
        if class_dir.exists():
            counts[label] = sum(1 for _ in class_dir.glob("*.mp4"))
    return counts


def download_shard(shard_index: int, tar_url: str, *, work_dir: Path) -> Path:
    tar_path = work_dir / "tars" / f"part_{shard_index}.tar.gz"
    print(f"[shard {shard_index}] downloading {tar_url}", flush=True)
    download_file(tar_url, tar_path)
    print(f"[shard {shard_index}] download complete", flush=True)
    return tar_path


def extract_and_filter_shard(
    shard_index: int,
    tar_path: Path,
    *,
    work_dir: Path,
    output_dir: Path,
    filename_to_label: dict[str, str],
    counts: Counter[str],
    min_per_class: int,
) -> int:
    matched = 0
    print(f"[shard {shard_index}] extracting", flush=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as scratch:
        scratch_path = Path(scratch)
        with tarfile.open(tar_path) as tar:
            tar.extractall(scratch_path)

        for video_path in scratch_path.rglob("*.mp4"):
            label = filename_to_label.get(video_path.name)
            if label is None:
                continue
            if counts[label] >= min_per_class:
                continue
            class_dir = output_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(video_path), str(class_dir / video_path.name))
            counts[label] += 1
            matched += 1

    tar_path.unlink(missing_ok=True)
    print(f"[shard {shard_index}] kept {matched} clips", flush=True)
    return matched


def main() -> int:
    args = parse_args()
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = work_dir / "train.csv"
    ensure_wget()
    print(f"Fetching annotations: {TRAIN_CSV_URL}")
    download_file(TRAIN_CSV_URL, annotations_path)

    rows = load_annotations(annotations_path)
    target_classes = select_target_classes(rows, args.num_classes)
    print(f"Selected {len(target_classes)} classes: {', '.join(target_classes)}")

    by_class: dict[str, list[str]] = defaultdict(list)
    for label, filename in rows:
        if label in target_classes:
            by_class[label].append(filename)

    avg_per_class = sum(len(v) for v in by_class.values()) / len(target_classes)
    est_fraction = min(1.0, args.min_per_class / avg_per_class)
    est_shards = max(1, round(est_fraction * 242))
    print(
        f"Average train clips/class for these labels: {avg_per_class:.0f}. "
        f"Rough estimate to reach {args.min_per_class}/class: ~{est_shards} of 242 "
        f"shards (~{est_shards * 1.6:.0f} GB). Actual shard content is random, so "
        "this script downloads adaptively and stops as soon as targets are met "
        "rather than trusting this estimate."
    )

    if args.plan_only:
        return 0

    filename_to_label = {
        filename: label for label, filenames in by_class.items() for filename in filenames
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = existing_counts(args.output_dir, target_classes)

    state_path = work_dir / "state.json"
    state = FetchState.load(state_path)

    shard_urls_path = work_dir / "k400_train_path.txt"
    download_file(TRAIN_PATHS_URL, shard_urls_path)
    shard_urls = shard_urls_path.read_text(
        encoding="utf-8").splitlines()[: args.max_shards]

    pending_indices = [i for i in range(len(shard_urls))
                        if i not in state.processed_shards]

    pool = ThreadPoolExecutor(max_workers=args.parallel_downloads)
    inflight: dict[int, Future[Path]] = {}
    queue: deque[int] = deque(pending_indices)

    def top_up() -> None:
        while queue and len(inflight) < args.parallel_downloads:
            idx = queue.popleft()
            inflight[idx] = pool.submit(
                download_shard, idx, shard_urls[idx], work_dir=work_dir)

    try:
        for shard_index in pending_indices:
            short_classes = [c for c in target_classes if counts[c] < args.min_per_class]
            if not short_classes:
                print("All target classes reached --min-per-class. Stopping early.",
                      flush=True)
                break

            top_up()
            tar_path = inflight.pop(shard_index).result()
            top_up()

            extract_and_filter_shard(
                shard_index, tar_path,
                work_dir=work_dir, output_dir=args.output_dir,
                filename_to_label=filename_to_label, counts=counts,
                min_per_class=args.min_per_class,
            )
            state.processed_shards.add(shard_index)
            state.save(state_path)

            still_short = [c for c in target_classes if counts[c] < args.min_per_class]
            print(f"[progress] {len(target_classes) - len(still_short)}/{len(target_classes)} "
                  f"classes satisfied after {len(state.processed_shards)} shard(s)",
                  flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    print("\nFinal per-class counts:")
    shortfall = False
    for label in target_classes:
        status = "OK" if counts[label] >= args.min_per_class else "SHORT"
        if status == "SHORT":
            shortfall = True
        print(f"  {label}: {counts[label]} ({status})")

    if shortfall:
        print(
            "\nSome classes are still short of --min-per-class after "
            f"{len(state.processed_shards)} shard(s). Re-run the same command to "
            "resume — already-processed shards are skipped — or raise --max-shards.",
            file=sys.stderr,
        )
        return 1

    print(f"\nDone. Clips are under {args.output_dir}. Next: build the subset index "
          "with src.data.kinetics_index (see this script's module docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
