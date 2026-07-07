# DINOv2 3 x 2 Interaction Notes

This note summarizes the completed `3 models x 2 datasets` C50 matrix. The DINOv2 runs are frame-wise image baselines, not temporal video encoders.

## Status

- All six cells have 2 original artifacts and 8 held-out perturbation artifacts.
- Quality audit overall status: `True`; both DINOv2 runs have 0 failed samples.
- DINOv2 original LP accuracy is 29.7% on SSV2 and 99.0% on UCF101.

## DINOv2 sanity check

- SSV2 `temporal-shuffle-mid`: mean cosine distance 3.79e-08, LP drop 0.0000.
- UCF101 `temporal-shuffle-mid`: mean cosine distance 3.20e-08, LP drop 0.0000.

This matches the frame-mean DINOv2 expectation: when the same frame set is preserved and only order changes, the simple-mean embedding is effectively unchanged.

## Main Interactions

- DINOv2 is nearly saturated on UCF101, with LP accuracy of 99.0%. This indicates that the C50 subset is highly readable from strong static image representations.
- DINOv2 reaches 29.7% LP accuracy on SSV2, higher than VideoMAE but lower than SlowFast. This shows that the SSV2 C50 subset still contains usable static cues, so VideoMAE/SlowFast versus DINOv2 differences should not be reduced to a simple motion-understanding claim.
- DINOv2 `freeze-tail-high` on SSV2 has LP drop 0.0207, much smaller than the high-strength freeze-tail drops for the video models. For DINOv2, this is mainly a frame-content distribution change, not evidence of temporal modeling.
- DINOv2 UCF101 `spatial-blur-mid` has a small LP drop of 0.0020. In contrast, VideoMAE x UCF101 still shows a much stronger appearance-related label effect under blur.

## Writing Boundaries

- UCF101 should be described as an `appearance-rich / context-correlated contrast`, not as a purely appearance-biased dataset.
- DINOv2 results cannot prove or disprove motion understanding; they mainly quantify how readable the current C50 subsets are from static frame-level representations.
- Representation shift and label-related drop should be reported separately.
