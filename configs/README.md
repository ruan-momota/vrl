# Experiment configs

Project configs should record every choice needed to reproduce an embedding extraction run:

- `dataset_name`: dataset identifier, for example `ssv2`.
- `split`: annotation split, for example `train` or `validation`.
- `video_root`: directory containing source videos.
- `annotation_path`: split annotation file.
- `label_mapping_path`: dataset label mapping, when available.
- `model_checkpoint`: Hugging Face model checkpoint.
- `num_frames`: sampled frame count per clip.
- `sampling_strategy`: deterministic or random frame sampling rule.
- `image_size`: model input image size.
- `batch_size`: inference batch size.
- `num_workers`: dataloader worker count.
- `device`: `cpu`, `cuda`, or `auto`.
- `embedding_type`: model output used as the saved embedding.
- `video_decoder`: selected video decoding backend.
- `output_path`: embedding artifact path.
- `seed`: random seed.
- `deterministic`: whether deterministic behavior is requested.
