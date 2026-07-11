# Experiment configs

Run-scoped extraction configs live under `configs/runs/` and record every
choice needed to reproduce one embedding artifact:

- `dataset_name`: dataset identifier, for example `ssv2`.
- `subset_id` and `subset_summary_path`: fixed subset identity and its audited
  class/sample summary.
- `split` and `index_path`: the split and immutable normalized video index.
- `model_checkpoint`: Hugging Face model checkpoint.
- `num_frames`: sampled frame count per clip.
- `sampling_strategy`: deterministic or random frame sampling rule.
- `image_size`: model input image size.
- `batch_size`: inference batch size.
- `num_workers`: dataloader worker count.
- `device`: `cpu`, `cuda`, or `auto`.
- `model_revision`: optional checkpoint revision; a resolved revision is also
  recorded in model metadata when the loader exposes one.
- `input_profile`: human-readable preprocessing/sampling profile.
- `perturbation`: one deterministic intervention plus its serializable
  parameters. `artifact_label` distinguishes repeated variants of the same
  intervention in one run.
- `output_root` and `run_id`: run-scoped artifact location and shared
  experiment identity. Split/index/perturbation are artifact-specific and do
  not create a new run ID.
- `seed`: random seed.
- `deterministic`: whether deterministic behavior is requested.

The SSV2 × VideoMAE frozen-linear-probe sensitivity configs are in
`configs/runs/ssv2_videomae_linear_probe/`:

- two original-artifact configs and eight held-out perturbation configs for
  SSV2 C50 × frozen VideoMAE;
- `ssv2_videomae_c50_linear_probe_evaluation.json` configures the train-only linear
  probe, paired bootstrap, auxiliary KNN k=5, and run-scoped reporting.

Phase 4 SlowFast configs use the same run-scoped protocol:

- `configs/runs/ssv2_slowfast_linear_probe/`
- `configs/runs/ucf101_slowfast_linear_probe/`
