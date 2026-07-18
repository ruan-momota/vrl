# UCF101 × VideoMAE frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe`. They produce one original
train artifact, one original held-out artifact, and fourteen held-out
perturbation artifacts under the same `outputs/runs/{run_id}/` directory. The
six quantization/solarization artifacts have been extracted and evaluated.

The UCF101 subset index is expected at
`data/ucf101/subsets/c50_train100_heldout30/`. Build and audit it first:

```bash
uv run python -m src.data.ucf101_index --decode-audit
```

Run smoke extraction on a compute node before full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_smoke_train.json \
  --limit 1 --local-files-only

uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_smoke_heldout_original.json \
  --limit 1 --local-files-only

uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_smoke_temporal_shuffle_mid.json \
  --limit 1 --local-files-only

uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_smoke_color_mid.json \
  --limit 1 --local-files-only
```

The perturbation strengths intentionally match the SSV2 × VideoMAE anchor:

| Intervention | low | mid | high |
| --- | ---: | ---: | ---: |
| `freeze_tail` start fraction | 0.75 | 0.50 | 0.25 |
| `color_transform` strength | 0.15 | 0.35 | 0.60 |

`spatial_blur` uses a fixed 5x5 separable box kernel; `temporal_shuffle` uses
deterministic seed 42.

After full extraction, run:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_train_original.json

uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_original.json

for config in \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_temporal_shuffle_mid.json \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_freeze_tail_low.json \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_freeze_tail_mid.json \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_freeze_tail_high.json \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_color_low.json \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_color_mid.json \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_color_high.json \
  configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done

uv run python -m src.pipeline.evaluate \
  --config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_linear_probe_evaluation.json
```

## RGB quantization and solarization

Run these only after the train-only pixel audit freezes the strengths:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_rgb_quantization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_rgb_quantization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_rgb_quantization_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_solarization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_solarization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_videomae_linear_probe/ucf101_videomae_c50_heldout_solarization_high.json
```
