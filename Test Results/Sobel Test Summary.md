# Sobel Sidewalk Texture Tests: Summary

**Date:** November 4, 2026
**Outcome:** Not proven yet. The Sobel stage works, is fast, and stays in the code as an optional check that is off by default. It lowers overall mask accuracy and flags many harmless edges. Its most promising result is outlining manhole cover rims.

## Goal
SegFormer labels many irregularities as sidewalk, and the GMM failed as a check because it mostly detected colour. We tested whether edge strength could act as a colour-independent second check: learn normal sidewalk texture from each frame, then remove pixels that differ from it. The original sidewalk and focus masks stay unchanged, and rejected pixels go into separate refined masks.

## Main findings
- **Accuracy dropped at every threshold.** Precision rose slightly, but recall fell more, so IoU went down.
- **It outlines manhole covers, which the GMM missed entirely.** At threshold 2.5, it rejected 38-45% of the rim and only 3-5% of the interior on the two closest frames tested.
- **It flags many harmless edges.** Painted lines, shadow edges, sunlit patches and the far end of the path were all flagged in the frames reviewed.
- **Speed is not a problem.** Sobel takes about 1.5 ms per frame, and the whole system about 40 ms of the estimated 500 ms between frames.
- **Hazard detection is still unmeasured.** We have no catch rate for the cone or the pothole.

## Final design tested
| Setting | Value |
|---|---|
| Input | Sobel gradient magnitude from the grayscale frame |
| Search area | Focus area bounding box plus a small margin |
| Local measure | 15 x 15 pixel average of gradient magnitude |
| Reference | Mean and SD of local averages, from the focus area eroded by the window size |
| Training exclusion | Pixels outside the threshold are dropped, then the reference is recalculated |
| Minimum sample | 500 pixels; below this, nothing is flagged in that frame |
| Decision | Not sidewalk if more than 2.5 reference SDs from the mean |
| Memory between frames | None; the reference is rebuilt every frame |

The starting threshold was 2.0. The default was changed to 2.5 after comparing overlays.

## Final results
Walk A, frames 1-600, each frame scored against its own labels.

| Run | IoU | Precision | Recall | Near-field coverage |
|---|---|---|---|---|
| Sobel off | 0.663 | 0.856 | 0.747 | 77.5% |
| Threshold 2.0 | 0.603 | 0.865 | 0.665 | 70.6% |
| Threshold 2.5 (default) | 0.635 | 0.861 | 0.708 | 74.6% |
| Threshold 3.0 | 0.649 | 0.858 | 0.727 | 76.2% |

These scores can understate Sobel, because the labels count irregularities as sidewalk. Correctly flagging a hazard on labelled sidewalk lowers the score.

| Live performance at 2 FPS | Sobel off | Sobel at 2.5 |
|---|---|---|
| Latency, mean / p95 | 37.9 / 41.1 ms | 39.0 / 42.9 ms |
| Sobel alone, mean / p95 | Not applicable | 1.31 / 2.28 ms |
| Dropped frames | 0 | 0 |

## What we learned

### Threshold
| Threshold | Share of focus rejected, frames 1-50 | Range across frames | Flagged pixels not labelled sidewalk* |
|---|---|---|---|
| 2.0 | 12.9% | 7.4-19.2% | 24.6% |
| 2.5 | 6.8% | 2.1-12.5% | 27.8% |
| 3.0 | 3.8% | 0.7-8.4% | 28.0% |

*From a 19-frame sample across the walk, where 11.7% of focus pixels were not labelled sidewalk.

- Raising the threshold mostly gives fewer flags, not better ones. The share of flags on non-sidewalk stayed at about 25-28%, roughly 2 to 2.4 times chance.
- Every pixel rejected at 3.0 in frames 1-50 was also rejected at 2.0.
- The threshold also shapes the reference, because it decides which pixels are dropped. In the 19-frame sample, this raised rejections at 2.0 from about 7% to 12%.

### Reference texture
- **The SD must come from local averages, not raw pixels.** Raw SD averaged 79 against a mean of 76, so the lower limit was negative on all 19 sampled frames and smooth areas could never be flagged. The SD of local averages was about 34.
- **The reference is rebuilt every frame.** This keeps rejected pixels out of it and avoids the lock-out seen with the GMM, where a new scene was flagged and so never learned.
- **Limiting the window to mask pixels did not help.** About half of flags fell within one window of the focus edge, a band covering 26% of the focus area. Masking moved that share only from 52% to 50%, so it was left out.

### Manhole cover
SegFormer is almost certain the cover is sidewalk, so its confidence cannot find it. The cover was outlined by hand in frames 20-22.

| Frame | `flat-sidewalk` confidence on cover | Same on surrounding pavement |
|---|---|---|
| 20 | 98.4% | 99.2% |
| 21 | 99.2% | 99.2% |
| 22 | 99.5% | 96.5% |

Sobel rejection on the same cover:

| Frame | Interior at 2.0 / 2.5 / 3.0 | Rim at 2.0 / 2.5 / 3.0 |
|---|---|---|
| 20 | 34% / 21% / 14% | 34% / 22% / 11% |
| 21 | 13% / 5% / 3% | 52% / 38% / 30% |
| 22 | 9% / 3% / 1% | 59% / 45% / 32% |

The rim stands out once the cover is close enough. In frame 20 the cover is only about 18 pixels tall, close to the 15-pixel window, so rim and interior do not separate. The break in the white line may contribute to the rim rejections.

### Other things seen in overlays
These come from reviewing a few frames, not from measurement.
- **Painted lines:** the white centre line and crossing markings are rejected (frames 8 and 43).
- **Shadows and light:** shadow edges and a sunlit patch are rejected (frames 30, 131 and 300).
- **Edges:** kerb, car and tree-base boundaries are rejected. Some of this usefully trims the mask.
- **Far end of the path:** compressed distant texture often looks stronger than the near-field reference.
- **Cone:** small patches at its base are rejected at 2.5 in frame 131, but the cone itself is not.

### Simulation problems found and fixed
- **Camera rate:** the frames were replayed at 9.46 FPS, but they are about one walking step apart. Using the cover's size as a ruler, and assuming a 0.7-0.8 m cover and 1.4 m/s walking pace, gave about 1.1 m per frame near frames 20-22 and about 2 FPS for the walk overall.
- **One-frame lag:** the old live rule always showed the previous frame's mask, which put Sobel labels in the wrong place. Agreement with the 2.5 overlay rose from 0.14 to 0.91 IoU once a frame could use its own mask. Hazard conclusions drawn from the lagged video were wrong and have been corrected here.
- **Video format:** MPEG-4 video at 2 FPS could not be decoded by Apple's media framework, so videos are now written as H.264.

## Limits of these results
- There is one walk, with one cover sequence, one cone and one pothole.
- The pothole was not tested, and no catch rates were measured for the cone.
- Cover outlines were drawn by hand, on three frames only.
- The labels count irregularities as sidewalk, so IoU cannot show whether hazards are caught.
- 2 FPS is an estimate and is applied as a constant, although motion between frames varies widely. The file timestamps do not hold capture times.
- At 2 FPS, live and no-lag scores are identical, because the dataset has no frames in between to measure a 40 ms delay. Treat them as a best case.
- The 19-frame sample results are approximate, and the 2.0 and 3.0 live runs used the older 9.46 FPS replay. Their no-lag scores are unaffected.

## Recommendations
- Outline the cone, pothole and covers by hand across their sequences, and measure catch rates at each threshold. This decides whether the stage is worth keeping.
- Keep Sobel off by default in anything user-facing until then.
- Test long curved edges as a manhole cover detector, since rims separate well from interiors at close range.
- Use a depth-dependent reference or window, because a fixed 15-pixel window and a single reference treat near and far texture the same.
- Try separating painted lines and shadows from cracks by edge shape or direction. Threshold tuning alone won't do this.
- Record real capture timestamps in future walks, so replay timing does not have to be estimated.
