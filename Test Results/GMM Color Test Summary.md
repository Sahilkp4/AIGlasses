# GMM Colour Check Tests: Summary

> **RETIRED (November 8, 2026).** This GMM colour check is retired from the full model. It was replaced by the median/MAD colour difference check in `color_difference.py`. On all 600 frames, with Sobel on, the replacement scored 0.650 IoU against the GMM's 0.648, with 47.7 ms mean latency against 69.9 ms and about 1/9 of the CPU. It catches somewhat less of the cone and racks. See `Color Difference Test Summary.md`. The code is kept for reference as `gmm_color_RETIRED.py`, with its sweep in `eval_gmm_RETIRED.py`.

**Date:** November 5, 2026
**Outcome:** Promising for coloured objects, not proven. The stage catches most of the cone and racks that SegFormer leaves in the focus area, and it barely lowers mask accuracy. The Sobel and GMM checks were turned on by default after these tests. It still misses the racks on some frames, flags dark ground under cars, and cannot see potholes or manhole covers.

## Goal
SegFormer labels many obstacles as sidewalk. Sobel checks texture, so this stage checks colour: learn the frame's normal sidewalk colours with a GMM on Lab `a`, `b`, then flag focus-area pixels that fit none of them.

An earlier GMM was removed as a failure (see `GMM Test Summary.md`). That one was judged on potholes and covers, and on noise. This one has a narrower job: coloured objects only. It also rebuilds its model every frame, as Sobel does, which rules out the earlier lock-out.

## Main findings
- **It finds coloured objects.** On held-out frames it caught 75.3% of yellow rack pixels left in the focus area.
- **It costs almost no mask accuracy.** IoU went from 0.663 to 0.662. Sobel at 2.5 gives 0.635.
- **Together, SegFormer and the GMM cover 95% of the cone.** SegFormer removes most of it, and the GMM catches 68% of what remains.
- **It misses potholes and covers,** as expected: 2.2% and 0%.
- **It sometimes misses the racks completely.** On 4 of 25 held-out frames it caught none of them.
- **It is still noisy:** about 125 flagged objects per minute on held-out clean frames, many of them dark ground under cars.

## Final design tested
| Setting | Value |
|---|---|
| Features | `a`, `b` (Lab colour) |
| Search area | Focus area bounding box plus a small margin |
| Local measure | 9 x 9 pixel average of `a` and `b` |
| Model | 4 components, full covariance, `cv2.ml.EM` |
| Training data | Up to 2000 random pixels from the focus area eroded by the window |
| Training exclusion | Samples beyond the threshold are dropped, then the GMM is refitted |
| Component filter | Components under 5% weight are ignored |
| SD floor | 1.0 Lab unit added to each component's SD |
| Minimum sample | 500 pixels; below this, nothing is flagged in that frame |
| Decision | Not sidewalk if more than 4 SD from every kept component |
| Minimum object | 100 pixels |
| Memory between frames | None; the GMM is rebuilt every frame |

## Final results
Walk A, frames 1-600, each frame scored against its own labels.

| Run | IoU | Precision | Recall | Near-field coverage |
|---|---|---|---|---|
| No checks | 0.663 | 0.856 | 0.747 | 77.5% |
| Sobel 2.5 | 0.635 | 0.861 | 0.708 | 74.6% |
| GMM | 0.662 | 0.859 | 0.743 | 77.1% |
| GMM + Sobel 2.5 | 0.634 | 0.863 | 0.705 | 74.3% |

These scores can understate both checks, because the labels count irregularities as sidewalk.

| Live performance at 2 FPS | Sobel 2.5 | GMM | GMM + Sobel |
|---|---|---|---|
| Latency, mean / p95 | 38.6 / 41.7 ms | 66.9 / 92.2 ms | 68.9 / 97.6 ms |
| Stage alone, mean / p95 | 1.44 / 2.55 ms | 33.7 / 58.8 ms | GMM 34.6 / 59.4 ms |
| Dropped frames | 0 | 0 | 0 |

### Catch rates
Share of target pixels left in the focus area that the GMM flagged. These 8 frames were used to choose the settings, so they run optimistic.

| Target | Frames | Share SegFormer left in focus | GMM catch rate |
|---|---|---|---|
| Cone | 129, 131, 133, 135 | 16% | 68.2% |
| Racks (yellow pixels) | 445, 451, 463 | 11% | 97.6% |
| Pothole | 129, 131, 133, 135 | 51% | 2.2% |
| Manhole cover | 21 | 100% | 0% |

### Held-out check
These frames played no part in choosing the settings. Racks were scored on every other frame from 441-480 with at least 30 yellow pixels in focus (25 frames). This target is colour alone, so it may include yellow paint in frame 457. Clean frames are 109 frames offset from the tuning set.

| Setting | Rack catch | Frames under 50% | False flags | Objects/min |
|---|---|---|---|---|
| 3 components, 10% weight limit | 56.6% | 4 | 1.69% | 196.0 |
| 4 components, 5% weight limit (final) | 75.3% | 5 | 0.48% | 125.5 |

The final setting's misses were frames 452 (37%), and 455, 456, 461 and 464 (0%).

## What we learned

### Cone by distance
| Frame | Cone pixels | Left in focus | GMM catch | Covered by SegFormer or GMM |
|---|---|---|---|---|
| 129 | 434 | 244 | 98% | 98.8% |
| 131 | 837 | 411 | 88% | 94.0% |
| 133 | 1904 | 285 | 35% | 90.3% |
| 135 | 4361 | 288 | 48% | 96.5% |

The catch rate falls as the cone gets closer, but SegFormer removes more of it there. The feared failure, a close cone being learned as sidewalk, was not what happened on these frames. What is left is a thin rim, which the window blends with pavement.

### Settings
Window 9, 4 components, 5% weight limit, tuning frames:

| Threshold | Cone | Racks | False flags | Objects/min |
|---|---|---|---|---|
| 3 | 73.9% | 97.9% | 0.88% | 267.5 |
| 4 (default) | 68.2% | 97.6% | 0.30% | 84.8 |
| 5 | 49.7% | 90.6% | 0.12% | 48.4 |
| 6 | 46.7% | 84.5% | 0.04% | 25.3 |

- **Window:** 5 lost cone and racks (51.9% and 86.5% at threshold 4). 15 scored slightly higher on the cone at threshold 4 (72.7%), but racks fell to 62.5% at threshold 5 and it flagged more objects.
- **The weight limit is a real trade-off.** At 10%, normal tactile strips lost their component and were flagged. Below 5%, objects start to earn components: at 3%, racks fell to 56.5%.
- **Rack results are unstable across settings.** With 3 components and a 5% limit, racks fell to 34.5%. Small changes decide whether a rack becomes its own component in a given frame, so treat the rack figures as approximate.
- **False flags are concentrated.** With the first settings tried, 50 of 109 clean frames had any flag, and 10 frames held 71% of flagged pixels.

### What gets flagged wrongly
These come from reviewing a few frames, not from measurement, and the review used the first settings tried.
- **Dark ground under cars:** neutral in colour against blue-tinted shaded pavement. Much of it is not labelled sidewalk, so some of it usefully trims the mask. On tuning frames, 56.7% of flags in clean frames were off labelled sidewalk.
- **Far end of the path:** sunlit, warmer pavement in the distance.
- **Tactile strips and red bands:** when they are a small part of the frame.

### Missed in the overlays
- **Frame 571:** the leaf-covered ground beside the path was predicted sidewalk and not labelled sidewalk, and the GMM did not flag it. A large wrong region can earn a component and count as normal.

### Regression check
Sobel-only output was unchanged by this work. Per-frame IoU matched the recorded run on all 600 frames, for both live and no-lag scores. Only timing differed.

## Limits of these results
- There is one walk, with one cone, one pothole, one cover and one rack sequence.
- Targets were outlined by hand on 8 frames, by reading coordinates off a grid. Rack outlines are loose, so only yellow pixels are scored and the black bands are not.
- Few target pixels are left in focus: 1228 for the cone and 1695 for the racks.
- False flags are measured on labelled sidewalk, and real objects inside labelled areas count against the stage.
- Each setting was run once, with a fixed seed.
- Walk B was not run, since it only mirrors walk A.
- The 2 FPS camera rate is an estimate, as in the Sobel tests.

## Recommendations
- Keep working on the rack misses and the object rate, now that the GMM is on by default.
- Test a short memory of the previous frame's sidewalk colours. It should help on frames where a rack earns a component, but it needs a guard against the earlier lock-out.
- Require an object to persist across two frames before reporting it. This should cut the object rate, and has not been tested.
- Try suppressing flags along the lower edge of cars, or use them to trim the mask on purpose.
- Test on a second walk with different pavement before trusting these settings.
