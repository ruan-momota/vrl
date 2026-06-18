# VRL

Early-stage video research pipeline for motion and appearance sensitivity experiments.

The current pipeline starts with Something-Something-V2 as a motion-biased dataset and VideoMAE as the first pretrained video model. The MVP pipeline is in place: local videos can be indexed, decoded, sampled, batched, embedded with VideoMAE, evaluated with KNN, and routed through deterministic perturbation hooks.

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

MVP status: the debug-subset acceptance points are complete, the first local 100 train / 100 validation original baseline has been generated, and first-round validation perturbation embeddings are available. The current active work is paired sensitivity metrics and KNN accuracy drop for the VideoMAE + SSV2 perturbation experiment.

Phase progress:

| Phase | Status | Notes |
| --- | --- | --- |
| 0. Project setup | Done | `uv` environment, config dataclass, reproducibility helper, output directories. |
| 1. SSV2 data indexing | Done | Local train / validation / test indexes plus overlapping debug subsets. |
| 2. Video read and sampling | Done | PyAV decode, deterministic center/uniform sampling, fail-fast errors. |
| 3. Dataset / DataLoader | Done | `SSV2ClipDataset`, custom collate, single-worker and multi-worker smoke checks. |
| 4. VideoMAE model loading | Done | Bare `AutoModel` encoder, `MCG-NJU/videomae-base`, mean-pooled hidden-state embeddings. |
| 5. Embedding extraction | Done for debug and local original splits | `.pt` artifacts include tensors, labels, video ids, metadata, config, model metadata, and summary. |
| 6. KNN baseline | Done for debug and local original 100/100 artifacts | Cosine and L2 KNN code path, deterministic majority vote, JSON report output. |
| 7. Perturbation hooks and matrix | Design done | Deterministic frame perturbations are implemented; the first-round experiment matrix is recorded in `configs/ssv2_videomae_perturbation_matrix.json`. |
| 8. Paired perturbation embeddings | Done for first-round validation artifacts | Six validation perturbation artifacts generated and aligned with original validation rows. |

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
- KNN baseline over saved embedding artifacts, with cosine similarity as the default metric and L2 as an available option.
- Deterministic perturbation hooks between clip sampling and VideoMAE preprocessing: temporal reverse, temporal shuffle, freeze-tail, single-frame, grayscale, and center occlusion.
- Original local embeddings for `train.jsonl` and `validation.jsonl`, each with 100 samples and 768-dimensional VideoMAE embeddings.
- Original local 100/100 KNN baseline with cosine and L2 reports under `outputs/logs/`.
- Baseline interpretability report showing that only 46 / 100 validation samples have labels seen in the 100-sample train reference set.
- First-round perturbation matrix with motion / appearance grouping, default parameters, output names, and selected sweeps.
- First-round validation perturbation embeddings for temporal reverse, temporal shuffle, freeze-tail, single-frame, grayscale, and center occlusion.
- Paired artifact alignment helper and checks for video id order, labels, sample count, embedding dim, and sampled frame indices.

Not completed yet:

- Embedding sensitivity metrics and KNN accuracy drop reports.
- Strength sweeps, class-level sensitivity reports, plots, and experiment summary.

Current verification:

- Test suite: `38 passed` with `.venv/bin/python -m pytest`.
- Debug train extraction: 33 / 33 samples, embedding shape `[33, 768]`, failures `0`.
- Debug validation extraction: 33 / 33 samples, embedding shape `[33, 768]`, failures `0`.
- Local train original extraction: 100 / 100 samples, embedding shape `[100, 768]`, failures `0`.
- Local validation original extraction: 100 / 100 samples, embedding shape `[100, 768]`, failures `0`.
- First-round validation perturbation extractions: each 100 / 100 samples, embedding shape `[100, 768]`, failures `0`.
- Paired alignment checks: all six first-round perturbation artifacts match original validation video ids, labels, sample count, embedding dim, and sampled frame indices.
- Saved artifacts were reloaded immediately after writing and validated for row alignment across embeddings, labels, video ids, metadata, and frame indices.
- Debug cosine KNN baseline: `k=1` accuracy `0.1212` (4 / 33), `k=5` accuracy `0.0606` (2 / 33), `k=10` accuracy `0.0303` (1 / 33).
- Local 100/100 original KNN baseline, cosine and L2: `k=1` accuracy `0.0200` (2 / 100), `k=5` accuracy `0.0100` (1 / 100), `k=10` accuracy `0.0100` (1 / 100).
- Train-seen-only local baseline: 46 / 100 validation samples have labels present in the train reference; `k=1` accuracy is `0.0435` (2 / 46).
- Perturbation extraction smoke: `temporal_reverse` with `--limit 1`, embedding shape `[1, 768]`, saved artifact reloaded successfully with perturbation config and frame order metadata.
- Perturbation matrix JSON validation: `python -m json.tool configs/ssv2_videomae_perturbation_matrix.json`.

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
30 passed
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

The first local original artifacts have already been generated:

- `outputs/embeddings/ssv2_train100_videomae_base_16f_mean_original.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_original.pt`

Regenerate the local train original artifact:

```bash
uv run python -m src.embedding_extraction \
  --config configs/ssv2_videomae_smoke.json \
  --split train \
  --index-path data/ssv2/index/train.jsonl \
  --output-path outputs/embeddings/ssv2_train100_videomae_base_16f_mean_original.pt \
  --batch-size 1 \
  --num-workers 0 \
  --local-files-only \
  --overwrite
```

Regenerate the local validation original artifact by changing `--split`, `--index-path`, and `--output-path` to `validation`, `data/ssv2/index/validation.jsonl`, and `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_original.pt`.

## KNN Baseline

Run the debug KNN baseline over the saved train / validation embedding artifacts:

```bash
uv run python -m src.knn_baseline \
  --train-artifact outputs/embeddings/ssv2_debug_train_videomae_base_16f_mean.pt \
  --validation-artifact outputs/embeddings/ssv2_debug_validation_videomae_base_16f_mean.pt \
  --k 1 5 10 \
  --metric cosine \
  --output-path outputs/logs/ssv2_debug_knn_videomae_base_16f_mean.json \
  --overwrite
```

The KNN CLI validates that both artifacts are labeled, embedding dimensions match, and model checkpoint / embedding type metadata are compatible. Cosine KNN normalizes embeddings before similarity search; L2 can be selected with `--metric l2`.

Current debug-subset cosine result:

| k | Correct / Total | Accuracy |
| --- | --- | --- |
| 1 | 4 / 33 | 0.1212 |
| 5 | 2 / 33 | 0.0606 |
| 10 | 1 / 33 | 0.0303 |

Run the local 100/100 original cosine baseline:

```bash
uv run python -m src.knn_baseline \
  --train-artifact outputs/embeddings/ssv2_train100_videomae_base_16f_mean_original.pt \
  --validation-artifact outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_original.pt \
  --k 1 5 10 \
  --metric cosine \
  --output-path outputs/logs/ssv2_validation100_videomae_base_16f_mean_original_knn_cosine.json \
  --overwrite
```

Current local 100/100 original result:

| metric | k | Correct / Total | Accuracy |
| --- | --- | --- | --- |
| cosine | 1 | 2 / 100 | 0.0200 |
| cosine | 5 | 1 / 100 | 0.0100 |
| cosine | 10 | 1 / 100 | 0.0100 |
| L2 | 1 | 2 / 100 | 0.0200 |
| L2 | 5 | 1 / 100 | 0.0100 |
| L2 | 10 | 1 / 100 | 0.0100 |

The local train / validation label overlap is limited: 46 / 100 validation samples have a label present in the 100-sample train reference. Perturbation KNN reports should therefore include both all-validation accuracy and train-seen-only accuracy.

## Perturbation Hooks

Perturbations run after deterministic clip sampling and before VideoMAE preprocessing. The original video id, label, sampled frame indices, and split metadata are preserved; each sample metadata record also includes the perturbation config and operation details.

Supported perturbations:

- `temporal_reverse`
- `temporal_shuffle`
- `freeze_tail`
- `single_frame`
- `grayscale`
- `center_occlusion`

Run a perturbation extraction smoke:

```bash
uv run python -m src.embedding_extraction \
  --config configs/ssv2_videomae_smoke.json \
  --split debug_train \
  --index-path data/ssv2/index/debug_train.jsonl \
  --output-path outputs/embeddings/ssv2_debug_train_videomae_base_16f_mean_temporal_reverse_limit1.pt \
  --batch-size 1 \
  --num-workers 0 \
  --local-files-only \
  --limit 1 \
  --perturbation temporal_reverse \
  --overwrite \
  --no-progress
```

Additional perturbation parameters include `--perturbation-seed`, `--perturbation-frame-index`, `--freeze-start-fraction`, `--occlusion-size-fraction`, and `--occlusion-fill-value`.

## Perturbation Matrix

The first VideoMAE + SSV2 perturbation matrix is recorded in:

```text
configs/ssv2_videomae_perturbation_matrix.json
```

First-round validation perturbations:

| perturbation | group | default |
| --- | --- | --- |
| `temporal_reverse` | motion | reverse the sampled frame order |
| `temporal_shuffle` | motion | deterministic per-video shuffle with seed `0` |
| `freeze_tail` | motion | `freeze_start_fraction = 0.5` |
| `single_frame` | motion | repeat the center sampled frame |
| `grayscale` | appearance | convert RGB frames to grayscale RGB |
| `center_occlusion` | appearance | `occlusion_size_fraction = 0.25`, fill value `0` |

The first round applies perturbations only to the validation split. The train reference remains `outputs/embeddings/ssv2_train100_videomae_base_16f_mean_original.pt`, so KNN drop measures how perturbed validation embeddings move in the original train embedding space.

Selected second-round sweeps are also fixed in the matrix:

- `freeze_tail`: `freeze_start_fraction = 0.25, 0.5, 0.75`
- `center_occlusion`: `occlusion_size_fraction = 0.15, 0.25, 0.4`
- `single_frame`: first, center, and last sampled frame
- `temporal_shuffle`: seeds `0, 1, 2`

Example first-round extraction command:

```bash
uv run python -m src.embedding_extraction \
  --config configs/ssv2_videomae_smoke.json \
  --split validation \
  --index-path data/ssv2/index/validation.jsonl \
  --output-path outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_temporal_reverse.pt \
  --batch-size 1 \
  --num-workers 0 \
  --local-files-only \
  --perturbation temporal_reverse \
  --overwrite
```

## Outputs

Generated embeddings should go under `outputs/embeddings/`. Logs should go under `outputs/logs/`.

Current generated debug artifacts:

- `outputs/embeddings/ssv2_debug_train_videomae_base_16f_mean.pt`: 33 samples, embedding shape `[33, 768]`.
- `outputs/embeddings/ssv2_debug_validation_videomae_base_16f_mean.pt`: 33 samples, embedding shape `[33, 768]`.
- `outputs/logs/ssv2_debug_knn_videomae_base_16f_mean.json`: debug KNN report.

Embedding `.pt` files and log `.json` files are ignored by git; keep `.gitkeep` files in the output directories.

## Next Steps

The next engineering step is to run paired original / perturbed embedding extraction for one debug perturbation sweep, then add sensitivity metrics such as original-vs-perturbed cosine distance and KNN accuracy drop.

Known remaining environment checks:

- PyAV is currently the active decoder; TorchCodec is still only a future fallback option.
- The current lock file resolves a recent PyTorch build. Confirm the CUDA driver/runtime on any remote GPU server before relying on long extraction runs.
