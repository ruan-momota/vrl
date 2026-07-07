# Diving48 3 x 3 Interaction Notes

This note summarizes the completed `3 models x 3 datasets` matrix. The added Diving48 C32 setting uses a balanced train50 / heldout15 subset. Its role is a fine-grained motion / pose contrast, not a full Diving48 benchmark.

## Status

- All nine cells have 2 original artifacts and 8 held-out perturbation artifacts.
- Quality audit overall status: `True`; all three Diving48 runs have 0 failed samples.
- Diving48 original LP accuracy: VideoMAE 7.5%, SlowFast 9.0%, DINOv2 8.7%.

## DINOv2 sanity check

- SSV2 `temporal-shuffle-mid`: mean cosine distance 3.79e-08, LP drop 0.0000.
- UCF101 `temporal-shuffle-mid`: mean cosine distance 3.20e-08, LP drop 0.0000.
- Diving48 `temporal-shuffle-mid`: mean cosine distance 3.71e-08, LP drop 0.0000.

This matches the frame-mean DINOv2 expectation: when the same frame set is preserved and only order changes, the simple-mean embedding is effectively unchanged.

## Main Interactions and Interpretation Boundaries

- DINOv2 is nearly saturated on UCF101, with LP accuracy of 99.0%. This indicates that the C50 subset is highly readable from strong static image representations.
- DINOv2 reaches 29.7% LP accuracy on SSV2, higher than VideoMAE but lower than SlowFast. This shows that the SSV2 C50 subset still contains usable static cues, so VideoMAE/SlowFast versus DINOv2 differences should not be reduced to a simple motion-understanding claim.
- All three Diving48 baselines are low but still above random 1/32. This should primarily be interpreted as fine-grained, small-sample difficulty in the C32 train50 setting, not as a failure of any single model implementation.
- On Diving48, VideoMAE temporal-shuffle LP drop is 0.0375, and SlowFast is 0.0188. Both are less pronounced than the fixed-mid label effects observed on SSV2/UCF101.
- SlowFast x Diving48 temporal-shuffle representation shift is 0.2053, much larger than DINOv2's 3.71e-08, but the label drop is only 0.0188. Representation shift and label-related drop must therefore be reported separately.
- Diving48 spatial-blur LP drops are: VideoMAE 0.0188, SlowFast -0.0083, DINOv2 0.0021. Current results do not support explaining Diving48 differences as simple static-appearance cue effects.

## Writing Boundaries

- UCF101 should be described as an `appearance-rich / context-correlated contrast`, not as a purely appearance-biased dataset.
- Diving48 should be described as a `fine-grained motion / pose contrast`, while explicitly stating that this experiment uses a C32 train50/heldout15 subset.
- DINOv2 results cannot prove or disprove motion understanding; they mainly quantify how readable the current subsets are from static frame-level representations.
- The low Diving48 baseline is an important result and should not be optimized away by post-hoc class or quota changes.
- Representation shift and label-related drop should be reported separately.
