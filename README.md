# VRL

Early-stage video research pipeline for motion and appearance sensitivity experiments.

The current MVP starts with Something-Something-V2 as a motion-biased dataset and VideoMAE as the first pretrained video model. The immediate goal is a stable pipeline for reading local videos, sampling clips, forming model-ready batches, extracting embeddings, and then running a small KNN baseline.

## Setup

Create the project environment with uv:

```bash
uv sync --dev
```

Check that PyTorch can see CUDA on the target machine when GPU inference is needed:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The current lock file targets Python 3.12 via `.python-version`. If the remote server has an older CUDA driver, adjust the PyTorch version or wheel source before running embedding extraction.

## Current Progress

MVP status: the first acceptance point is complete. The project can read the local SSV2 debug subsets, preprocess clips, run VideoMAE, save embeddings, and reload the saved artifacts with alignment checks.

Phase progress:

| Phase | Status | Notes |
| --- | --- | --- |
| 0. Project setup | Done | `uv` environment, config dataclass, reproducibility helper, output directories. |
| 1. SSV2 data indexing | Done | Local train / validation / test indexes plus overlapping debug subsets. |
| 2. Video read and sampling | Done | PyAV decode, deterministic center/uniform sampling, fail-fast errors. |
| 3. Dataset / DataLoader | Done | `SSV2ClipDataset`, custom collate, single-worker and multi-worker smoke checks. |
| 4. VideoMAE model loading | Done | Bare `AutoModel` encoder, `MCG-NJU/videomae-base`, mean-pooled hidden-state embeddings. |
| 5. Embedding extraction | Done for debug splits | `.pt` artifacts include tensors, labels, video ids, metadata, config, model metadata, and summary. |
| 6. KNN baseline | Next | Debug train / validation artifacts are ready for a first KNN code path. |
| 7. Perturbation hooks | Later | Motion / appearance perturbation interfaces are still pending. |

Completed:

- Project environment, config skeleton, output directories, and reproducibility helper.
- Local SSV2 indexing under `data/ssv2/index/`.
- Debug subsets: `debug_train.jsonl` and `debug_validation.jsonl`, each with 33 labeled videos across 16 overlapping classes.
- PyAV video decoding with fail-fast errors that include video id and path context.
- Deterministic 16-frame clip sampling.
- VideoMAE preprocessing with Hugging Face `VideoMAEImageProcessor`.
- `SSV2ClipDataset`, `collate_video_batch`, and DataLoader smoke checks with `num_workers=0` and `num_workers=2`.
- VideoMAE model loading with the Hugging Face checkpoint [`MCG-NJU/videomae-base`](https://huggingface.co/MCG-NJU/videomae-base).
- Random-tensor and real-video VideoMAE forward checks, using `last_hidden_state` mean pooling as the MVP embedding.
- Split-level embedding extraction with `.pt` artifacts, config snapshots, model metadata, per-sample metadata, and save/reload validation.

Not completed yet:

- Full train / validation embedding extraction beyond the debug subsets.
- KNN baseline over train and validation embeddings.
- Perturbation hooks and motion / appearance sensitivity metrics.

Current verification:

- Test suite: `15 passed, 1 skipped` with `uv run python -m pytest`.
- Debug train extraction: 33 / 33 samples, embedding shape `[33, 768]`, failures `0`.
- Debug validation extraction: 33 / 33 samples, embedding shape `[33, 768]`, failures `0`.
- Saved artifacts were reloaded immediately after writing and validated for row alignment across embeddings, labels, video ids, metadata, and frame indices.

## Data

The local SSV2 sample is expected at:

```text
data/ssv2/labels/
data/ssv2/videos/train/
data/ssv2/videos/validation/
data/ssv2/videos/test/
```

Current local data contains 300 `.webm` videos, split as 100 train, 100 validation, and 100 test videos. The generated indexes contain:

- `data/ssv2/index/train.jsonl`: 100 local train samples, 66 classes.
- `data/ssv2/index/validation.jsonl`: 100 local validation samples, 67 classes.
- `data/ssv2/index/test.jsonl`: 100 local test samples, unlabeled by default.
- `data/ssv2/index/debug_train.jsonl`: 33 samples.
- `data/ssv2/index/debug_validation.jsonl`: 33 samples.
- `data/ssv2/index/summary.json`: index statistics and video probe metadata.

Regenerate indexes with:

```bash
uv run python -m src.ssv2_index
```

## Smoke Checks

Core Python modules live under `src/`.

The first VideoMAE smoke configuration is:

```bash
configs/ssv2_videomae_smoke.json
```

It uses the local SSV2 layout under `data/ssv2/`, 16 deterministic frames, batch size 1, PyAV decoding, and mean pooling over `last_hidden_state`.

Run the test suite:

```bash
uv run python -m pytest
```

Current verified result:

```text
15 passed, 1 skipped
```

Run a real-video DataLoader and preprocessing check:

```bash
uv run python -m src.video_sanity_check \
  --index-path data/ssv2/index/debug_train.jsonl \
  --limit 2 \
  --batch-size 2 \
  --num-workers 0
```

The current verified output shape for batch size 2 is:

```text
[2, 16, 3, 224, 224]
```

`src.video_sanity_check` also supports `--num-workers`, `--pin-memory`, `--shuffle`, and `--drop-last`, and reports `batch_load_seconds`. A `num_workers=2` check has been verified outside the sandbox because Python multiprocessing uses local socket operations that are blocked in the sandbox.

Run the VideoMAE model loading and forward check:

```bash
uv run python -m src.videomae_model_sanity_check \
  --config configs/ssv2_videomae_smoke.json \
  --index-path data/ssv2/index/debug_train.jsonl \
  --limit 1 \
  --batch-size 1 \
  --num-workers 0
```

The first online run downloads and caches `MCG-NJU/videomae-base`. After that, the same check can run offline with `--local-files-only`.

Current verified model-forward shapes are:

```text
pixel_values:       [1, 16, 3, 224, 224]
last_hidden_state:  [1, 1568, 768]
embedding:          [1, 768]
```

The checkpoint model card describes `MCG-NJU/videomae-base` as a pre-trained-only VideoMAE base model trained on Kinetics-400 with self-supervised pretraining. The MVP uses the bare encoder via `AutoModel` and saves no classification logits.

## Embedding Extraction

Extract a debug train split:

```bash
uv run python -m src.embedding_extraction \
  --config configs/ssv2_videomae_smoke.json \
  --split debug_train \
  --index-path data/ssv2/index/debug_train.jsonl \
  --output-path outputs/embeddings/ssv2_debug_train_videomae_base_16f_mean.pt \
  --batch-size 1 \
  --num-workers 0 \
  --local-files-only \
  --overwrite
```

Extract the matching debug validation split by changing `--split`, `--index-path`, and `--output-path` to `debug_validation`.

The exact debug validation command is:

```bash
uv run python -m src.embedding_extraction \
  --config configs/ssv2_videomae_smoke.json \
  --split debug_validation \
  --index-path data/ssv2/index/debug_validation.jsonl \
  --output-path outputs/embeddings/ssv2_debug_validation_videomae_base_16f_mean.pt \
  --batch-size 1 \
  --num-workers 0 \
  --local-files-only \
  --overwrite
```

The saved `.pt` artifact is a dictionary with:

- `embeddings`: `[N, D]` float tensor.
- `label_ids`: `[N]` tensor for labeled splits, otherwise `None`.
- `video_ids`, `sample_metadata`, and `frame_indices`.
- `config`, `model_metadata`, `extraction_options`, and `summary`.

The CLI reloads the artifact after saving and validates that tensor rows, labels, video ids, metadata, and frame indices stay aligned.

## Outputs

Generated embeddings should go under `outputs/embeddings/`. Logs should go under `outputs/logs/`.

Current generated debug artifacts:

- `outputs/embeddings/ssv2_debug_train_videomae_base_16f_mean.pt`: 33 samples, embedding shape `[33, 768]`.
- `outputs/embeddings/ssv2_debug_validation_videomae_base_16f_mean.pt`: 33 samples, embedding shape `[33, 768]`.

Embedding `.pt` files are ignored by git; keep `.gitkeep` files in the output directories.

## Next Steps

The next engineering step is the KNN baseline over the two debug artifacts. Use cosine similarity first, report `k = 1, 5, 10`, and keep the result labeled as a debug-subset baseline. Full train / validation extraction can follow once the KNN path is stable.

Known remaining environment checks:

- PyAV is currently the active decoder; TorchCodec is still only a future fallback option.
- The current lock file resolves a recent PyTorch build. Confirm the CUDA driver/runtime on any remote GPU server before relying on long extraction runs.
