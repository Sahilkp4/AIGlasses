# Sobel Sidewalk Texture Tests: Summary

**Date:** November 4-5, 2026
**Outcome:** Promising, and now on by default with the GMM. A redesigned stage, Log 31x5, replaced the first design. On all 600 frames of walk A it cut false flags from 5.24% to 2.78% and far false flags from 14.07% to 4.72%, and it caught more of the pothole, manhole rim, racks and a sidewalk fault. It catches less of the cone, adds long thin false streaks, and is untested on cracks running along the path.

## Goal
SegFormer labels many irregularities as sidewalk, and the GMM failed as a check because it mostly detected colour. We tested whether edge strength could act as a colour-independent second check: learn normal sidewalk texture from each frame, then remove pixels that differ from it. The original sidewalk and focus masks stay unchanged, and rejected pixels go into separate refined masks.

The first design worked but flagged too much distant sidewalk. The second round of tests set out to fix that without losing small irregularities in the path.

## Main findings
- **Distant false flags came from texture that looked too strong, not too weak.** About 99% of first-design flags were "too strong". Pavers do blur together in the distance, but seams and shadows get packed into each window.
- **Log 31x5 roughly halved false flags and raised IoU.** With the GMM on, IoU rose from 0.634 to 0.648, and false objects fell from 580 to 413 per minute.
- **Log brightness is what made potholes detectable.** Catch rose from 8.6% to 37.8%. Earlier stages caught almost none of the pothole.
- **A per-row reference can learn a fault as normal.** A fault running across the path sits inside one row band, so the reference has to be a smooth curve, not separate bands.
- **The fault is caught better than before.** Log 31x5 caught 24 of 27 fault parts, compared with 15 for the first design.
- **Long thin false streaks are the main new cost.** They rose from 68 to 431 and now make up about a quarter of remaining false pixels.
- **Speed is not a problem.** Sobel takes about 2.4 ms per frame, and the whole system about 68 ms of the estimated 500 ms between frames.

## Final design
| Setting | Value |
|---|---|
| Input | Sobel gradient magnitude of log brightness, `log(1 + gray) x 46` |
| Search area | Focus area bounding box plus a small margin |
| Local measure | 31 x 5 pixel average, using only pixels inside the focus area eroded by 5 x 5 |
| Texture scale | Log of the local average |
| Reference sample | Focus area eroded by 15 x 15 |
| Reference | Median and scaled MAD in 20-row bands, each fitted with a quadratic curve over image rows |
| Band limits | Bands under 150 sample pixels are skipped; with 3 or fewer bands, values are interpolated instead of fitted |
| Minimum sample | 500 pixels and 2 bands; below this, nothing is flagged in that frame |
| Spread floor | 0.05 in log texture units |
| Decision | Not sidewalk if more than 2.45 spreads above normal for its row. Smooth areas are not flagged |
| Memory between frames | None; the reference is rebuilt every frame |

`--sobel-window` now takes a width and a height, for example `--sobel-window 31 5`. Both must be odd and at least 3.

## Final results
Walk A, frames 1-600, each frame scored against its own labels, with the GMM on in both runs. The earlier run with neither check scored 0.663 IoU and 77.5% near-field coverage.

| Run | IoU | Precision | Recall | Near-field coverage |
|---|---|---|---|---|
| First design 2.5 + GMM | 0.634 | 0.863 | 0.705 | 74.3% |
| Log 31x5 + GMM (default) | 0.648 | 0.864 | 0.722 | 75.3% |

These scores can understate both designs, because the labels count irregularities as sidewalk. Correctly flagging a hazard on labelled sidewalk lowers the score. Log 31x5 was not run with the GMM off on all 600 frames.

False flags are the share of labelled sidewalk in the focus area that Sobel flagged, on 543 frames away from the cone and rack sequences and outside the fault outlines. Far means above row 360.

| False flags | First design 2.5 | Log 31x5 |
|---|---|---|
| All | 5.24% | 2.78% |
| Far | 14.07% | 4.72% |
| Near | 4.03% | 2.52% |
| False objects per minute | 580 | 413 |
| Long thin streaks | 68 | 431 |

Log 31x5 had fewer false pixels on 377 of the 543 frames, more on 95, and the same on 71.

| Live performance at 2 FPS | First design 2.5 + GMM | Log 31x5 + GMM |
|---|---|---|
| Latency, mean / p95 | 68.9 / 97.6 ms | 67.9 / 93.3 ms |
| Sobel alone, mean / p95 | 0.71 / 1.57 ms | 2.36 / 4.63 ms |
| Dropped frames | 0 | 0 |

### Catch rates
Share of target pixels left in the focus area that Sobel flagged. These targets helped choose the settings, so they run optimistic.

| Target | Frames | First design 2.5 | Log 31x5 |
|---|---|---|---|
| Manhole rim | 21 | 41.0% | 50.3% |
| Pothole | 129, 131, 133, 135 | 8.6% | 37.8% |
| Racks (yellow pixels) | 445, 451, 463 | 38.9% | 56.0% |
| Cone | 129, 131, 133, 135 | 40.1% | 33.0% |
| Fault parts caught | 46-54 | 15 of 27 | 24 of 27 |

The GMM already catches most of the cone and racks, so Sobel's main job is the pothole, rim and fault, which the GMM almost entirely misses.

### Regression check
The code in `sobel_texture.py` flagged exactly the same pixels as the tested experiment on all 600 frames. Per-frame IoU from the final simulator run also matched the tested run on all 600 frames. Running only on the focus area box cut Sobel from 6.43 ms to 2.36 ms without changing any flags.

## What we learned

### Why distant sidewalk was flagged
First design at 2.5, on 104 clean frames.

| Rows | False flags | Flags that were too strong | Share of reference sample |
|---|---|---|---|
| 240-270 | 40.0% | 100% | 0.2% |
| 270-300 | 24.8% | 100% | 1.2% |
| 300-330 | 14.0% | 99.8% | 2.8% |
| 330-360 | 9.2% | 100% | 5.1% |
| 360-400 | 7.0% | 98.7% | 10.8% |
| 400-450 | 5.3% | 98.4% | 21.9% |
| 450-540 | 2.8% | 98.2% | 58.0% |

- **Windows spilled over the focus edge.** The path narrows to about 75 pixels in rows 240-300, where 62% of flags were within one window of the edge. Near the camera that share was 29%.
- **One reference fitted near ground only.** Typical texture barely changed with distance (median about 55-60), but the 90th percentile rose from 112 near the camera to 155 far away.
- **Dark ground was underscored.** In frame 133 the pothole was 1.5 times the texture of nearby pavement in plain Sobel, and 2.5 times on log brightness.
- **Only strong texture could be flagged.** The lower limit averaged about 74 - 2.5 x 30, which is below zero. Sobel on log brightness and a too-strong-only rule make this explicit.
- **Resolution is part of it.** Source frames are 3840 x 2160 and are reduced 4 times. Pavers visible at 4K blur together at 960 x 540. Measuring far rows at 4K brought far false flags level with near ones (2.7% and 2.8%), but pothole catch fell to 4.8% and racks to 17.7%, so this was dropped.

### First fix: the Combined version
Combined used per-row bands, windows limited to the focus area, and the stronger score from plain and log Sobel. On target frames and 104 clean frames, at about the same false flag rate as the first design:

| Run | False flags, all / far | Pothole | Rim | Cone | Racks |
|---|---|---|---|---|---|
| First design 2.5 | 5.15% / 13.7% | 8.6% | 41.0% | 40.1% | 38.9% |
| Combined 2.0 | 5.12% / 7.2% | 67.4% | 38.2% | 42.3% | 75.1% |
| Combined 2.0, flags above row 360 ignored | 4.26% / 0% | 45.7% | 38.2% | 17.9% | 61.2% |

On frames 1-100 with the GMM on, the gains were less clear:

| Run | False flags, all / far / near | Rim | IoU |
|---|---|---|---|
| First design 2.5 | 5.50% / 10.66% / 4.75% | 41.0% | 0.787 |
| Combined 2.5 | 3.52% / 6.17% / 3.13% | 18.9% | 0.804 |
| Combined 2.0 | 6.83% / 10.08% / 6.35% | 38.2% | 0.779 |

Combined 2.5 cut far false pixels by 64-82% in frames 1-50, but only 10-12% in frames 51-100. Combined 2.0 was worse than the first design in frames 51-100. Review of the overlays then showed Combined missing a fault in frames 49-52.

### Sidewalk fault, frames 46-54
A fault runs across the path on both sides of the centre line. The left part is sunken and broken with a dark void, the middle is a thin crack, and the right part is lifted pavers with dark cracks. It passes under the camera at frame 55. Each part was outlined by hand with a loose box on all nine frames.

- **Size:** 3-7 pixels tall at a distance and over 100 pixels wide across all parts.
- **Brightness:** the left part is about 25% darker than pavement in the same rows, with about twice the normal texture on log brightness.
- **Edge direction:** no help. Its edges are 70-80% horizontal, and so are ordinary pavers at about 70%.
- **Why Combined missed it:** the fault filled up to 39% of its 20-row band in frames 50-51, so the band learned it as normal. Scored against neighbouring bands, the left part rose from 2.4 to 4.4.

All variants below were set to the same false flag budget as Combined 2.5 (about 2.5%). Frames 49-52 were used for tuning, and frames 46-48 and 53-54 were set aside. A part counts as caught when at least 20 pixels and 15% of its outline are flagged.

| Variant | Threshold | False flags | Parts, frames 49-52 | Parts, other frames | Rim | Pothole | Cone | Racks |
|---|---|---|---|---|---|---|---|---|
| First design | 2.5 | 4.84% | 8 of 12 | 7 of 15 | 41.0% | 8.5% | 40.1% | 38.8% |
| Combined | 2.5 | 2.49% | 2 of 12 | 3 of 15 | 18.9% | 36.0% | 16.2% | 67.3% |
| Curve, plain and log, 15 x 15 | 2.65 | 2.48% | 6 of 12 | 6 of 15 | 56.5% | 35.7% | 27.1% | 62.9% |
| Curve, log only, 15 x 15 | 2.45 | 2.34% | 6 of 12 | 7 of 15 | 47.8% | 46.3% | 35.6% | 63.3% |
| Curve, log only, 21 x 9 | 2.45 | 2.33% | 9 of 12 | 10 of 15 | 52.6% | 43.4% | 38.2% | 61.3% |
| Curve, log only, 31 x 7 | 2.5 | 2.35% | 11 of 12 | 10 of 15 | 47.8% | 40.5% | 32.7% | 59.4% |
| Curve, log only, 31 x 5 (final) | 2.45 | 2.42% | 12 of 12 | 12 of 15 | 50.3% | 37.8% | 33.0% | 56.0% |

- **A wide, short window fits perspective.** Ground features are flattened, so a square window dilutes them.
- **Other smooth references were not better.** Wider 40-60 row bands, a cubic curve and median filtering across bands all recovered part of the fault, but the quadratic curve was the best overall at 15 x 15. Measuring spread from the lower half only did not help.
- **Growing weak flags from strong ones did not help.** Within the budget, it lost fault parts.
- **Log 31x5 is less sensitive to the threshold than 21 x 9.** Fault parts caught across all nine frames:

| Threshold | Log 31x5 false flags | Log 31x5 parts | 21 x 9 false flags | 21 x 9 parts |
|---|---|---|---|---|
| 2.3 | 2.98% | 25 of 27 | 2.88% | 23 of 27 |
| 2.4 | 2.59% | 25 of 27 | 2.50% | 22 of 27 |
| 2.45 | 2.42% | 24 of 27 | 2.33% | 19 of 27 |
| 2.5 | 2.25% | 22 of 27 | 2.17% | 17 of 27 |
| 2.6 | 1.96% | 21 of 27 | 1.88% | 17 of 27 |

On frames 1-100 in the full pipeline, 21 x 9 was slightly cleaner (2.95% false flags and 6.83 false objects per frame, against 3.02% and 7.59), but caught 19 parts against 24. It was not run on all 600 frames. Log 31x5 was chosen after review of its full overlay.

### What the full run exposed
- **Streaks:** a streak here is a flagged blob at least 60 pixels wide and 6 times wider than tall. They appear on 213 of the 543 frames, mostly in frames 1-100 and 501-600. A simple shape filter would not work, because the real fault looks the same.
- **Frames 401-500:** false pixels barely improved, from about 191,000 to 181,000, and false objects rose from 123 to 140.
- **Frame 484:** the tactile strip and barrier shadows are flagged more heavily than with the first design.
- **Frame 512:** long streaks cross the sidewalk.
- **Frame 529:** the largest IoU drop, at -0.115, but the focus area is only about 2% of the frame.
- **No reference:** this happened on 27 frames, against 24 for the first design. Many are in frames 349-369, where the focus area is tiny.
- **Painted lines:** flags on the white centre line were nearly gone in frame 51.
- **Still flagged:** dark ground under cars and the area around tree bases.

## First design, November 4
The first design below was replaced by Log 31x5. Its results are kept as the record of how the stage developed. Some of its conclusions changed in later tests: limiting the window to focus pixels helped once combined with the other changes, and the lower limit stayed near zero at 2.5.

### First design
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

### First design results
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

#### Threshold
| Threshold | Share of focus rejected, frames 1-50 | Range across frames | Flagged pixels not labelled sidewalk* |
|---|---|---|---|
| 2.0 | 12.9% | 7.4-19.2% | 24.6% |
| 2.5 | 6.8% | 2.1-12.5% | 27.8% |
| 3.0 | 3.8% | 0.7-8.4% | 28.0% |

*From a 19-frame sample across the walk, where 11.7% of focus pixels were not labelled sidewalk.

- Raising the threshold mostly gives fewer flags, not better ones. The share of flags on non-sidewalk stayed at about 25-28%, roughly 2 to 2.4 times chance.
- Every pixel rejected at 3.0 in frames 1-50 was also rejected at 2.0.
- The threshold also shapes the reference, because it decides which pixels are dropped. In the 19-frame sample, this raised rejections at 2.0 from about 7% to 12%.

#### Reference texture
- **The SD must come from local averages, not raw pixels.** Raw SD averaged 79 against a mean of 76, so the lower limit was negative on all 19 sampled frames and smooth areas could never be flagged. The SD of local averages was about 34.
- **The reference is rebuilt every frame.** This keeps rejected pixels out of it and avoids the lock-out seen with the GMM, where a new scene was flagged and so never learned.
- **Limiting the window to mask pixels did not help.** About half of flags fell within one window of the focus edge, a band covering 26% of the focus area. Masking moved that share only from 52% to 50%, so it was left out.

#### Manhole cover
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

#### Other things seen in overlays
These come from reviewing a few frames, not from measurement.
- **Painted lines:** the white centre line and crossing markings are rejected (frames 8 and 43).
- **Shadows and light:** shadow edges and a sunlit patch are rejected (frames 30, 131 and 300).
- **Edges:** kerb, car and tree-base boundaries are rejected. Some of this usefully trims the mask.
- **Far end of the path:** compressed distant texture often looks stronger than the near-field reference.
- **Cone:** small patches at its base are rejected at 2.5 in frame 131, but the cone itself is not.

#### Simulation problems found and fixed
- **Camera rate:** the frames were replayed at 9.46 FPS, but they are about one walking step apart. Using the cover's size as a ruler, and assuming a 0.7-0.8 m cover and 1.4 m/s walking pace, gave about 1.1 m per frame near frames 20-22 and about 2 FPS for the walk overall.
- **One-frame lag:** the old live rule always showed the previous frame's mask, which put Sobel labels in the wrong place. Agreement with the 2.5 overlay rose from 0.14 to 0.91 IoU once a frame could use its own mask. Hazard conclusions drawn from the lagged video were wrong and have been corrected here.
- **Video format:** MPEG-4 video at 2 FPS could not be decoded by Apple's media framework, so videos are now written as H.264.

## Limits of these results
- There is one walk, with one cover sequence, one cone, one pothole, one rack sequence and one fault.
- Settings were chosen on these same targets, so catch rates run optimistic.
- Frames 46-48 and 53-54 were set aside, but their results guided the choice of window shape, so they are not fully independent.
- Fault outlines are loose boxes, and the rule for a caught part was chosen during this work. Frame 54 scores 0 for Log 31x5, although the overlay shows strips on the left part.
- The manhole rim is scored on frame 21 only.
- Cracks running along the path were not tested. A wide, short window may dilute them.
- The labels count irregularities as sidewalk, so IoU cannot show whether hazards are caught.
- Streaks were counted by shape, and not all were reviewed. Frames 141-142 have the most and were not viewed during this work, so some may be real features.
- 2 FPS is an estimate and is applied as a constant, although motion between frames varies widely. The file timestamps do not hold capture times.
- At 2 FPS, live and no-lag scores are identical, because the dataset has no frames in between to measure the delay. Treat them as a best case.
- Walk B was not run, since it only mirrors walk A.

## Recommendations
- Test on a second walk with different pavement before trusting these settings.
- Find or record a crack running along the path, and measure how well Log 31x5 catches it.
- Reduce the streaks without using shape alone, since the real fault is also long and thin.
- Consider alerting only below about row 360, while keeping far flags as a warning. Very roughly, that is 5-7 m ahead, or 4-5 seconds of walking. It is untested with the final design.
- Find out why frames 401-500 barely improved.
- Test whether an object must persist across frames before it is reported.
- Test long curved edges as a manhole cover detector, since rims separate well from interiors at close range.
- Record real capture timestamps in future walks, so replay timing does not have to be estimated.
