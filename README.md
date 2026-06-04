# VRL

Early-stage video research pipeline for motion and appearance sensitivity experiments.

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

## Smoke Config

Core Python modules live under `src/`.

The first VideoMAE smoke configuration is:

```bash
configs/ssv2_videomae_smoke.json
```

It uses the local SSV2 layout under `data/ssv2/`, 16 deterministic frames, batch size 1, PyAV decoding, and mean pooling over `last_hidden_state`.

## Outputs

Generated embeddings should go under `outputs/embeddings/`. Logs should go under `outputs/logs/`.
