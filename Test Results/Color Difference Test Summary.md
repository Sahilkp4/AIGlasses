# Colour Difference Check Tests: Summary

**Date:** November 8, 2026
**Outcome:** Adopted. The median/MAD colour difference check is now the colour stage of the full model, and the GMM colour check is retired (`gmm_color_RETIRED.py`).

On all 600 frames of walk A, with Sobel on in both runs, it scored slightly higher IoU than the GMM: 0.650 against 0.648. Mean latency fell from 69.9 to 47.7 ms and p95 from 97.7 to 56.0 ms, and it used about 1/9 of the CPU. It catches somewhat less of the coloured targets (86.0% against 91.1% of cone and rack pixels in the checked area), but flags 4.6 times less clean sidewalk.

In the same work, the focus area was extended so both checks also score the holes it encloses.

## Goal
Replace the 4-component GMM in the colour check with a standard robust statistic, judged on two things: correct classification and compute cost. The model is one median colour per frame. Each channel's spread is estimated from its MAD, and each pixel gets a normalised distance `D` from the median.

## Final design (`color_difference.py`)
| Setting | Value |
|---|---|
| Features | `a`, `b` (Lab colour), uint8 |
| Checked area | Focus area plus the non-sidewalk holes it fully encloses |
| Local measure | 3 x 3 box blur of `a`, `b` |
| Reference sample | Sidewalk-labelled focus pixels, eroded by the window, so no hole or outside pixel reaches the reference |
| Reference | Per-channel median, and SD = 1.4826 x MAD, both from 256-bin histograms |
| Score | `D = sqrt(((a - med_a)/(sd_a + EPS))^2 + ((b - med_b)/(sd_b + EPS))^2)`, read from lookup tables |
| Decision | Flag when `D > 3.5`, with `EPS = 4` |
| Minimum object | 50 pixels |
| Minimum sample | 500 reference pixels; below this, nothing is flagged in that frame |
| Memory between frames | None |

Command line: `--color/--no-color`, `--color-window`, `--color-threshold` in `sidewalk_system.py` and `simulate_sidewalk.py`.

## Main findings
- **Compute is the clear win.** The colour stage dropped from about 33 ms to about 2-3 ms per frame. After the change, SegFormer is about three quarters of total latency.
- **`EPS` was the setting that mattered most.** Pavement `a` often has a MAD of a single step (SD about 1.5). With a small `EPS`, any slight tint flags. With `EPS = 4`, the cut-off is about 3.5 x (1.5 + 4), roughly 19 Lab units, and it widens on varied pavement. In effect it is close to a fixed colour distance from the median.
- **The IoU rule only works once `EPS` is tuned.** With `EPS = 1`, IoU rose with every increase in threshold, because the labels call almost everything in focus sidewalk. Maximising it then just picked the setting that flagged least. With `EPS` swept, the chosen setting beats both "no colour check" and the GMM on IoU.
- **The GMM still catches somewhat more.** Its tight components make a yellow cone stand out. One wide median model needs the cone to be further away in colour before it flags.
- **Most rack "misses" are a focus-area effect, not a colour-model effect.** See the rack investigation below.

## First test: 86 frames (frames 1-30, 122-136, 220-260)
This was run before holes were scored, with `EPS = 1` and a minimum object of 100.

| Setting | IoU | Cone | False flags | Objects/min | ms |
|---|---|---|---|---|---|
| No colour check | 0.7585 | 0% | 0% | 0 | 0 |
| GMM, window 9, T 4 | 0.7573 | 68.2% | 0.15% | 71 | 31.5 |
| Diff, window 5, T 7 | 0.7576 | 44.7% | 0.19% | 34 | 1.94 |
| Diff, window 9, T 4 (cone matched to GMM) | 0.7436 | 68.6% | 3.02% | 271 | 2.04 |

- At textbook thresholds of 3-4, one median flagged 3-7% of clean sidewalk. The red bands and pink tiles in frames 220-260 were the worst.
- At equal noise, it caught about two thirds as much cone as the GMM.
- This test led to the full 600-frame tuning and to sweeping `EPS`.

## Focus area: holes are now scored
**Change.** `fill_holes` in `sobel_texture.py` adds every non-sidewalk region that the focus area fully encloses. A gap open to the region's edge or the frame border is not filled. Both checks score this filled area, but:
- **References use sidewalk pixels only.** They are built from `focus & sidewalk`. In the colour check, the erosion is the same size as the blur, so no reference window reaches a non-sidewalk pixel.
- **Sobel computes texture twice.** Its 31 px wide window is wider than its 15 x 15 reference erosion. So focus pixels and the reference use a texture averaged over sidewalk pixels only, and only hole pixels use a texture averaged over the filled area.
- **The focus mask itself is unchanged,** and the refined masks are still the focus or sidewalk mask minus rejections. A hole never becomes walkable, whether or not it is flagged.

**Checked on the 86 frames.** 79 frames had holes, about 111k pixels in total.
- `fill_holes` matched an independent connected-components implementation exactly.
- References were identical to before for every check in every frame.
- Nothing was flagged outside the filled area, and `refined_focus` never contained a hole pixel.
- **Sobel:** results on focus pixels were identical once texture was split. A first version averaged hole pixels into nearby focus pixels' texture and changed about 7,900 flags, so it was fixed.
- **Colour checks:** a few hundred extra sidewalk pixels were flagged. These are fringes under 100 px that join a flagged hole's blob.
- **Sidewalk islands:** about 1,000 sidewalk-labelled pixels inside holes, not in focus, can now be flagged.
- **Cost:** Sobel takes about 0.6 ms longer. Objects reported rose slightly: 31 to 36 for colour difference over 86 frames.

**Effect.** Colour now flags the body of the cone in frame 131 (481 of the hole's 591 px), which was previously invisible to it. The cone now counts as a hole.

## Tuning: all 600 frames of walk A
- **Command:** `python3 eval_color_difference.py --frames all`. SegFormer masks are cached and Sobel is off. Results are in `runs/color_difference_eval/sweep_all_stage{1,2,2_w1,3}.csv`.
- **Catch:** the share of target pixels inside the checked area that were flagged. Holes are included. **Catch** below is the mean of the cone (frames 129-135) and the racks (445, 451, 463, yellow pixels only).
- **False flags:** the share of labelled sidewalk in focus that was flagged, on the 543 frames outside 125-140 and 440-480.

| Stage | Swept | Finding |
|---|---|---|
| 1 | window 3-11 x T 4-9 (EPS 1, min object 100) | Window 3 was best at equal noise; larger windows dilute the cone |
| 2 | window 3: EPS 0.5-3 x min object 50/100/200 x T 5-8; window 1 | A larger EPS cut false flags sharply for a small loss of catch. Min object 50 beat 100 and 200. Raw pixels caught less |
| 3 | windows 3, 5: EPS 3-5 x min object 25/50 x T 3-5 | IoU levels off at 0.6705-0.6707. Min object 25 added objects without catching more |

| Setting | IoU | Cone | Racks | Catch | False flags | Objects/min | ms (1 thread) |
|---|---|---|---|---|---|---|---|
| No colour check | 0.6701 | 0% | 0% | 0% | 0% | 0 | 0 |
| GMM, window 9, T 4 (retired) | 0.6692 | 83.4% | 98.8% | 91.1% | 0.43% | 108 | 33.2 |
| Diff, window 3, T 6, EPS 1, min 50 (most catch within the GMM's noise) | 0.6686 | 78.1% | 98.3% | 88.2% | 0.40% | 101 | 1.66 |
| **Diff, window 3, T 3.5, EPS 4, min 50 (chosen)** | **0.6706** | **74.5%** | **97.5%** | **86.0%** | **0.093%** | **74.5** | **1.65** |
| Diff, window 3, T 3, EPS 5, min 25 (strict best IoU) | 0.6707 | 74.3% | 97.9% | 86.1% | 0.12% | 130 | 1.65 |

**Why the chosen setting.** It is within 0.0001 IoU of the strict best and better on catch, false flags and objects/min. Every value sits inside the swept ranges.

## Full live run (`simulate_sidewalk.py --walks A`, Sobel on, 2 FPS, MPS)
| | Sobel + colour difference | Sobel + GMM (retired) |
|---|---|---|
| IoU / precision / recall (live) | **0.650** / 0.863 / 0.724 | 0.648 / 0.864 / 0.722 |
| Focus covers near-field sidewalk | 75.5% | 75.3% |
| Latency mean / p95 | **47.7 / 56.0 ms** | 69.9 / 97.7 ms |
| Colour stage mean / p95 | **3.0 / 5.6 ms** | 33.7 / 61.9 ms |
| Colour objects over 600 frames | 459 | 634 |
| CPU per processed frame | **39 ms** (7.8% of one core) | 334 ms (66.9% of one core) |
| Dropped frames | 0 | 0 |

Outputs:
- **Colour difference:** `runs/sidewalk_diff/`, with `overlay_walkA.mp4`.
- **GMM:** `runs/sidewalk_gmm_holes/`.

### Where the time goes (per frame)
`SidewalkNavigator.process` now returns `timing_ms` per stage. On MPS, GPU work is synchronised at stage boundaries so it is not counted in the wrong stage. `simulate_sidewalk.py` prints the table below, plus CPU time and peak memory.

| Stage | Colour diff run: mean / p95 ms (share) | GMM run: mean / p95 ms (share) |
|---|---|---|
| SegFormer preprocess (resize, normalise, upload) | 11.3 / 16.0 (24%) | 6.9 / 7.8 (10%) |
| SegFormer inference | 24.5 / 25.2 (51%) | 24.4 / 24.9 (35%) |
| SegFormer postprocess (argmax, upsample, readback) | 1.2 / 1.7 (2%) | 0.8 / 1.5 (1%) |
| Sidewalk and focus masks | 1.4 / 1.7 (3%) | 0.8 / 1.3 (1%) |
| Sobel | 6.2 / 13.2 (13%) | 3.2 / 7.1 (5%) |
| Colour check | 3.0 / 5.6 (6%) | 33.7 / 61.9 (48%) |
| Refined masks | 0.1 / 0.2 (<1%) | 0.1 / 0.1 (<1%) |

- **Preprocessing is the next target.** It is plain NumPy float maths on a 512 x 512 image and costs more than Sobel and colour together.
- **CPU stages ran about 1.6-2 times slower in the lighter run.** This covers preprocessing, Sobel and masks, and colour difference ran at 3.0 ms against 1.65 ms single-threaded in the sweep. The likely cause is that macOS runs a mostly idle process on slower cores or clocks. This is not measured.
- **Peak memory** was about 2.0 GB, of which about 0.9 GB is the replay preloading 600 frames.

### Overlay
`overlay_walkA.mp4` colours each pixel by class, with a legend:
- **Regions:** focus area kept, SegFormer sidewalk outside focus, and holes in focus.
- **Flags on sidewalk:** Sobel, colour, or both.
- **Flags in holes:** Sobel, colour, or both.

The labelled sidewalk is outlined in white.

Spot checks:
- **Frame 131:** the cone's yellow top is flagged by colour, and its hole by both checks.
- **Frame 251:** pink-tiled pavement gives no colour flags.
- **Frame 451:** colour catches the yellow rack parts inside the focus area.
- **Frame 571:** leaf-covered ground beside the path stays in focus, and only Sobel catches part of it. This is a known miss.

## Why the racks flip between frames (frames 441-470)
The rack catch jumps from frame to frame: 445 is caught, 446 missed, 449 missed, 450 caught. This comes from SegFormer's focus area, not from the colour model.
- **Caught frames:** SegFormer joins the path and the plaza beyond the racks into one sidewalk region. The rack row is then inside the focus area or a hole in it, so it is scored.
- **Missed frames:** the plaza is unlabelled (446) or a separate sidewalk region (449). The focus edge stops at the rack row, and the racks are open to the outside, so they are not a hole. The colour check never scores them.
- **Test:** scoring a band 15 px beyond the focus edge recovered frame 446 from 0% to 95% and frame 449 from 0% to 74%. Most frames rose from 0-40% to 75-99%.

**In the missed frames, the racks were mostly not walkable anyway.** Of the yellow rack pixels near the path:

| Frame | Not labelled sidewalk | Sidewalk, not in focus | Hole in focus | In focus | Still walkable after colour + Sobel |
|---|---|---|---|---|---|
| 445 (caught) | 61% | 1% | 14% | 24% | 70 px |
| 446 (missed) | 94% | 1% | 0% | 5% | 29 px |
| 449 (missed) | 92% | 3% | 0% | 6% | 101 px |
| 450 (caught) | 34% | 0% | 26% | 40% | 17 px |

Over frames 441-470, 69% of rack pixels near the path were not labelled sidewalk, and 20% were in focus. After both checks, 1,685 rack pixels stayed walkable over the 30 frames. The colour check catches the racks when SegFormer leaves much of them in focus, which is when it matters. Its "misses" mostly cost a reported object, not a walkable hazard.

**The misses that matter are frames 455 and 457.** In 455, 74% of the rack is in focus and 324 px stay walkable; in 457, 153 px stay walkable. In both, the focus area is about 2% of the frame. The spread estimate grows (`b` SD 7.4 and 13.3), and the cut-off rises to 40-60 units. Two smaller effects add to this:
- **The blob filter drops slivers** under 50 px. Turning it off barely helps.
- **The spread moves in steps.** It is estimated on uint8 values, so `b`'s SD can only take values 1.5 apart, and the yellow cut-off jumps between about 19, 30 and 35 units from one frame to the next.

## Moving to colour difference
- **Retired GMM files:** `gmm_color.py` is now `gmm_color_RETIRED.py`, and `eval_gmm.py` is now `eval_gmm_RETIRED.py`. Both are kept only to reproduce results. Nothing in the pipeline imports them.
- **Pipeline:** `sidewalk_system.py` and `simulate_sidewalk.py` use `ColorDifference` as the only colour check. The colour debug panel moved into `color_difference.py`.
- **Eval:** shared helpers (frame loading, targets) moved from the GMM eval into `eval_targets.py`. `eval_color_difference.py` uses the GMM's measured 0.43% false-flag rate as a fixed budget instead of running the GMM.
- **Regression check:** colour and Sobel masks and colour object boxes on 40 frames (125-139, 441-465) were identical before and after the refactor.

### Final verification run
This run used the full model after the move, with default settings: `python3 simulate_sidewalk.py --walks A --out runs/sidewalk_final_color`. No flags were needed, and the GMM was not imported.

| | Before the move (`runs/sidewalk_diff`) | After the move (`runs/sidewalk_final_color`) |
|---|---|---|
| IoU / precision / recall (live) | 0.6496 / 0.8629 / 0.7243 | 0.6496 / 0.8629 / 0.7243 |
| Raw-model baseline IoU | 0.5338 | 0.5338 |
| Focus covers near-field sidewalk | 75.52% | 75.52% |
| Colour objects | 459 | 459 |
| Per-frame IoU, all 600 frames | | 0 differences |
| Latency mean / p95 | 47.7 / 56.0 ms | 46.2 / 54.7 ms |
| Colour stage mean / p95 | 3.0 / 5.6 ms | 2.7 / 5.1 ms |
| CPU per processed frame | 39.1 ms | 35.9 ms |

Accuracy is identical. Timing differs only by run-to-run variation.

Two more checks were run:
- **Offline eval.** `python3 eval_color_difference.py`, which now scores the current defaults, reproduced the stage 3 tuning row on every metric except timing.
- **Command line.** `sidewalk_system.py` was run on an 8-frame clip of frames 444-451, with colour on and with `--no-color`, and with `--debug-video`. The panel flagged the racks in frame 450. The old `--gmm` option is now rejected.

## Recommendations
- **Stabilise the spread estimate.** Compute MAD on un-rounded values, cap how far the spread can grow, and fall back to the previous frame's reference when the focus area is too small to trust. This targets the real walkable misses in frames 455 and 457.
- **Report racks more reliably.** Close small gaps in the focus area (about one rack width) before filling holes, or score a narrow band beyond the focus edge, after measuring the false-flag cost. This mainly improves object reporting, not safety.
- **Require objects to persist across two frames** before reporting them, to cut the object rate. Untested.
- **Speed up preprocessing:** move normalisation to the GPU, or use `cv2.dnn.blobFromImage`.
- **Test on a second walk with different pavement** before trusting these settings.

## Limits
- There is one walk, and few colour targets: the cone on 4 frames and the racks on 3. About 130 settings were compared on the same frames they were chosen on, so catch figures are optimistic and differences under a few points are noise.
- IoU differences between the top settings (0.0001-0.0006) are far below what this data can resolve.
- False flags are measured on labelled sidewalk, and the labels count some objects as sidewalk.
- The rack investigation uses yellow pixels near the path as a stand-in for the racks: the black bands are not counted, and some yellow paint may be.
- There is one live run per configuration on one machine. Stage times depend on CPU load, as noted above.
