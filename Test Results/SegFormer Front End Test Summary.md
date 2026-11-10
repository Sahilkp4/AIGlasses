# SegFormer Front End Tests: Summary

**Date:** November 9, 2026
**Outcome:** Margin method adopted. `sidewalk_system.py` now builds the sidewalk mask with the margin method instead of argmax followed by a nearest upsample. The 512 x 512 input, the resize chain and the RGB conversion are unchanged. A 672 x 384 aspect-preserving input was tested and rejected: it scored higher IoU but left much more of the pothole walkable, and its overlay was judged poor on review.

On all 600 frames of walk A, the full model with the margin method scored 0.6529 IoU against 0.6496 before. Segmentation post-processing plus mask building fell from 2.33 to 1.44 ms per frame. No other output changed format, and the raw-model baseline was identical on every frame.

## Goal
Check whether stage 0 (shrinking the frame) and stage 1 (SegFormer preprocessing, inference and post-processing) let SegFormer label sidewalk as well as it can, while keeping runtime down and the model light. Each design detail was tested on its own:
- Shrinking the 4K frame to 960 x 540 before resizing to the model input
- BGR to RGB conversion
- Running argmax before upsampling
- Model input size and shape
- Precision (fp16)

## What SegFormer recommends
| Source | Recommendation |
|---|---|
| Checkpoint `preprocessor_config.json` (`tobiasc/segformer-b0-finetuned-segments-sidewalk`) | Squash to 512 x 512 with PIL bilinear (`resample: 2`), RGB, ImageNet mean and SD |
| HF fine-tuning blog that produced the checkpoint | Default processor (512 x 512), colour jitter, no random rescaling. At inference, upsample logits bilinearly, then argmax |
| SegFormer paper (Xie et al. 2021) | Train on random rescale 0.5-2.0 with 512 or 1024 crops. At test time, rescale the short side to the crop size **and keep the aspect ratio** (ADE20K). B0 on Cityscapes: 71.9 mIoU at short side 512, 76.2 at 1024 |
| `transformers` SegFormer training loss | Logits are upsampled **bilinearly** (`align_corners=False`) to label size before the loss, so the model learned logits meant to be read after bilinear interpolation |

The checkpoint was fine-tuned on `segments/sidewalk-semantic` (Belgium, 2021), not on `dataset_flip`. That dataset is gated, so its native resolution and aspect ratio could not be checked. `dataset_flip` is an out-of-domain test set: 4K (3840 x 2160) frames with polygon labels.

## Final design (stage 0 and stage 1)
| Setting | Value | Changed? |
|---|---|---|
| Frame shrink | Wider than 960 px: `INTER_AREA` to 960 wide (`main()`) | No |
| Model input | 512 x 512, OpenCV `INTER_LINEAR`, aspect squashed | No |
| Colour and normalisation | BGR to RGB, ImageNet mean and SD, fp32 | No |
| Post-processing | **Margin:** at 1/4 resolution, best `SIDEWALK` logit minus best other logit; bilinear upsample of that one map; sidewalk where > 0 | **Yes** (was argmax at 1/4 resolution, nearest upsample of the class map, then `np.isin` on the CPU) |
| `segment()` returns | Boolean sidewalk mask | **Yes** (was a uint8 class map) |

## How the margin method works
SegFormer does not output a label per pixel. For a 512 x 512 input it outputs **logits**: a score for each of the 35 classes at every point of a 128 x 128 grid, 1/4 of the input size. The class with the highest score wins. Turning that coarse grid into a 960 x 540 sidewalk mask can be done three ways.

**Old method: argmax, then nearest upsample.**
```
logits (35 x 128 x 128) -> argmax -> class map (128 x 128) -> nearest upsample -> 540 x 960 -> np.isin(SIDEWALK)
```
Each grid cell's winning class is copied into a block of about 7.5 x 4.2 frame pixels. Edges are staircases snapped to the grid. The full-size class map is then copied to the CPU and checked against the sidewalk classes.

**Documented method: bilinear upsample of all logits, then argmax.**
```
logits (35 x 128 x 128) -> bilinear upsample all 35 -> 35 x 540 x 960 -> argmax -> np.isin(SIDEWALK)
```
Every class score is interpolated smoothly to full resolution before the winner is picked. The edge falls where the scores actually cross, between grid points. This is how SegFormer's training loss reads the logits, but it means interpolating and comparing about 18 million values per frame (30 ms on MPS).

**Margin method (now in `sidewalk_system.py`).**
```
logits (35 x 128 x 128)
  -> margin = (best of the 3 SIDEWALK scores) - (best of the other 32 scores)   (1 x 128 x 128)
  -> bilinear upsample that one map -> 540 x 960 -> sidewalk where margin > 0
```
In code (`SidewalkNavigator.sidewalk_margin`):
```python
margin = logits[:, sidewalk_ids].amax(1, keepdim=True) - logits[:, other_ids].amax(1, keepdim=True)
mask = F.interpolate(margin, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0
```

**Why it gives the same decision.** The pipeline never needs to know which non-sidewalk class won, only whether a sidewalk class won. The margin is positive exactly when the best sidewalk score beats the best other score, which is when argmax picks a sidewalk class. So at every one of the 128 x 128 grid points, margin and argmax agree. This was confirmed on every cell of the verification frames.

**Where it differs.** Between grid points, margin interpolates one score map, so edges are smooth and placed where the sidewalk-vs-other margin crosses zero, as in the documented method. Margin and the documented method can disagree slightly where the strongest competing class changes from one cell to the next (for example road on one side, kerb on the other). "Take the max, then interpolate" is not exactly "interpolate, then take the max". Away from edges, all three methods give identical masks.

**Why it is faster.**
- It interpolates 1 map instead of 35.
- It skips the full-size argmax.
- It skips the class-map copy and `np.isin` on the CPU: the mask comes back already boolean.

**Small example.** Two neighbouring grid cells, scored for sidewalk (best of the three sidewalk classes) and road (best other class):

| | Sidewalk score | Best other score | Margin | Old method | Margin method |
|---|---|---|---|---|---|
| Cell A | 4.0 | 1.0 | +3.0 | sidewalk | sidewalk |
| Cell B | 1.0 | 2.0 | -1.0 | not sidewalk | not sidewalk |
| Point 3/4 of the way from A to B | | | 3.0 + 0.75 x (-1.0 - 3.0) = 0.0 | depends on which block it falls in | edge falls here |

The old method switches at the block boundary, halfway between A and B. The margin method puts the edge where the scores actually cross, 3/4 of the way from A to B, because A is much more confident than B.

## Main findings
- **The frame shrink and the RGB conversion are correct and cost nothing.** Feeding BGR cost 8 points of focus IoU. The 960 step had no measurable effect.
- **Input shape mattered more than anything else, but IoU hid a safety cost.** Keeping the 16:9 shape at the same pixel count (672 x 384) raised IoU by about 2 points at no extra cost. It also labelled more of the pothole as sidewalk, so about half the pothole stayed walkable after both checks, against about a third before.
- **IoU cannot judge front-end changes on its own.** The labels count the pothole as sidewalk, so a front end that walks people into it scores better. Front-end changes must also report how much of each hazard is left walkable.
- **The documented post-processing is accurate but too slow.** Bilinear-then-argmax gained about 0.3 points but cost 30 ms on MPS and 55 ms on CPU, more than the model's own inference.
- **The margin method gets the same gain cheaply.** It makes exactly the argmax decision at each output cell and only changes edge pixels. In the real code path it is 2.2x faster than the old post-processing on MPS and 3.7-4.9x on CPU, though that only saves 0.7-1.8 ms per frame.
- **Margin matches how SegFormer was trained.** Under nine simulated camera and condition changes, it stayed 4-5x closer to the training-time recipe than the old method.
- **The focus area's largest-region choice is unstable on fragmented frames.** This existed before these changes; the comparisons exposed it.

## Test 1: each design detail on its own
`python3 eval_segformer_input.py`. This covered walk A, all 600 frames. Each variant changes one detail from the pipeline as it was. Scored on the focus area (largest sidewalk region) against the labels at 960 x 540, pooled over frames. Results are in `runs/segformer_input_eval/accuracy.json` and `per_frame_focus_iou.csv`.

### Resize chain (512 x 512, RGB)
| Variant | Focus IoU |
|---|---|
| **Current: 4K to 960 `INTER_AREA`, then 512 `INTER_LINEAR`** | **0.6635** |
| 960 from the JPEG 1/4 DCT decode (used by earlier evals), then 512 linear | 0.6631 |
| 4K straight to 512 `INTER_AREA` (no 960 step) | 0.6654 |
| 4K straight to 512 `INTER_LINEAR` | 0.6624 |
| 960 to 512 `INTER_AREA` | 0.6578 |
| 960 to 512 PIL bilinear (the checkpoint's documented resize) | 0.6482 |

The current chain is within noise of the best. The documented PIL resize was slightly worse and about 20x slower (2.68 ms against 0.12 ms).

### Colour order
| Variant | Focus IoU |
|---|---|
| RGB (current) | 0.6635 |
| BGR fed without the swap | 0.5832 |

### Post-processing (same logits, 512 x 512 input)
| Method | Sidewalk IoU | Focus IoU | P / R | Frames better / worse vs current |
|---|---|---|---|---|
| Argmax, then nearest (old) | 0.6703 | 0.6635 | 0.855 / 0.748 | |
| Bilinear upsample of all logits, then argmax (documented) | 0.6728 | 0.6666 | 0.853 / 0.753 | 58 / 7 |
| **Margin** | **0.6739** | **0.6664** | **0.855 / 0.752** | **35 / 9** |

Margin beat the old method by 0.2-0.5 focus IoU points on every one of the 18 input variants tested.

### Input size and shape (old post-processing)
| Input | Focus IoU | P / R | MPS inference | CPU inference |
|---|---|---|---|---|
| 256 x 256 | 0.5759 | 0.707 / 0.756 | | |
| 384 x 384 | 0.6302 | 0.788 / 0.758 | 7.2 ms | 85 ms |
| 448 x 448 | 0.6502 | 0.829 / 0.750 | 10.1 ms | 97 ms |
| **512 x 512 (current)** | **0.6635** | **0.855 / 0.748** | **13.2 ms** | **131 ms** |
| 640 x 640 | 0.6545 | 0.875 / 0.722 | 23.7 ms | 177 ms |
| 768 x 768 | 0.6485 | 0.898 / 0.700 | 46.9 ms | 298 ms |
| 1024 x 1024 | 0.6049 | 0.916 / 0.641 | | |
| 512 x 288 (16:9) | 0.6661 | 0.837 / 0.765 | | |
| 672 x 384 (16:9) | 0.6823 | 0.875 / 0.756 | 13.3 ms | 114 ms |
| 896 x 512 (16:9, the paper's short-side-512 protocol) | 0.6777 | 0.881 / 0.746 | 27.3 ms | 220 ms |
| 960 x 540 (native) | 0.7005 | 0.893 / 0.764 | about 27 ms | |

- Larger squares lose recall, probably because objects look bigger than in training. The checkpoint was fine-tuned on fixed 512 x 512 resizes without random rescaling.
- Keeping 16:9 helped at every size, which matches the paper's test protocol, not the checkpoint's squash.

### Precision
fp16 on MPS changed 0.01% of focus pixels and cut inference from 13.2 to 8.5 ms (-35%). Not adopted yet (see Recommendations).

### Timing benchmark (hot loop, median)
`python3 eval_segformer_input.py --timing`. Results are in `runs/segformer_input_eval/timing.json`.

| Step | MPS | CPU |
|---|---|---|
| Decode one 4K JPEG | 23.4 ms | |
| 4K to 960 `INTER_AREA` | 0.63 ms | |
| 4K to 512 `INTER_AREA` | 10.1 ms | |
| Preprocess, NumPy normalise (current) | 3.2 ms | 2.6 ms |
| Preprocess, upload uint8, normalise on device | 0.9 ms | 0.6 ms |
| Post-process: argmax + nearest | 0.69 ms | 2.4 ms |
| Post-process: bilinear all logits + argmax | 30.0 ms | 54.7 ms |
| Post-process: margin | 0.47 ms | 0.43 ms |

These are hot-loop numbers. The live 2 FPS run measured preprocessing at about 11 ms and inference at about 24 ms, likely because the CPU and GPU slow down between frames.

## Test 2: combined settings, walks A and B
Walk B is the mirrored copy of walk A. Results are in `runs/segformer_input_eval/combined.log`.

| Setting | Walk A focus IoU | Walk B focus IoU | Pre + inference + post (MPS) |
|---|---|---|---|
| 512 x 512, fp32, argmax + nearest (old) | 0.6631 | 0.6880 | 14.9 ms |
| 512 x 512, fp16, margin | 0.6663 | 0.6902 | 9.7 ms |
| 672 x 384, fp32, argmax + nearest | 0.6824 | 0.7112 | 15.0 ms |
| 672 x 384, fp16, margin | 0.6861 | 0.7142 | 9.8 ms |
| 960 x 540, fp32, argmax + nearest | 0.6998 | 0.7200 | 36.3 ms |
| 960 x 544, fp16, margin | 0.6852 | 0.7135 | 23.1 ms |

The last row was run by mistake at 960 x 544 rather than 960 x 540, so it is not a clean comparison of native resolution with fp16.

## Test 3: effect on the full model (focus area, Sobel, colour)
`python3 eval_segformer_pipeline.py`. This ran the full model on walk A with default Sobel and colour settings. It compared the old front end with 672 x 384 + margin. Results are in `runs/segformer_pipeline_eval/summary.json`.

| | Old (512 x 512, argmax) | 672 x 384 + margin |
|---|---|---|
| Focus IoU / P / R | 0.663 / 0.856 / 0.747 | 0.686 / 0.874 / 0.761 |
| Refined focus IoU | 0.650 | 0.676 |
| Near-field coverage | 77.5% | 79.3% |
| Holes per frame (px) | 4.96 (872) | 3.38 (397) |
| Frames with no Sobel / colour reference | 27 / 1 | 37 / 4 |
| Sobel false flags (clean frames) | 2.81% | 2.43% |
| Share of Sobel false flags within one window of the focus edge | 32.7% | 34.7% |
| Colour false flags | 0.093% | 0.085% |
| Colour objects per minute | 74.5 | 87.7 |
| Segmentation / total latency (mean) | 25.5 / 32.5 ms | 22.7 / 29.8 ms |
| Sobel / colour time | 2.9 / 1.7 ms | 3.0 / 1.7 ms |

### What was behind the odd results
- **The two focus areas often disagreed** (mean overlap 0.65, worst tenth 0.26). In 146 frames with overlap under 0.5, the sidewalk is small or in pieces: 49k labelled px against 80k on average. `_largest_region` picks a different piece from small mask changes. In about 70 frames the second-largest region is more than half the largest. Both front ends score about 0.27 IoU on those frames, so this is an existing weakness of the focus area, not of either front end.
- **Ten more frames had no Sobel reference.** Both front ends fail on frames 349-369 and 518-522, where the focus area is about 3,000 px. The new one adds 397, 399, 417, 418, 495, 498, 500 and 523-525. Sobel silently flags nothing on those frames.
- **Smoother edges did not reduce Sobel edge streaks.** The share of false flags at the focus edge stayed at about a third.

### Hazards left walkable
This is the share of each hand-outlined target inside the refined focus (walkable) after both checks. Frames: cone and pothole 129-135, racks 445/451/463, cover 21.

| Front end | Pothole | Cover | Cone | Racks |
|---|---|---|---|---|
| Old (512 x 512, argmax) | 31.6% | 77.4% | 5.2% | 0.5% |
| Margin only (512 x 512) | 31.7% | 77.2% | 6.2% | 0.2% |
| 672 x 384 only (argmax) | 49.3% | 87.8% | 3.6% | 0.6% |
| 672 x 384 + margin | 50.5% | 88.9% | 4.2% | 0.3% |

- **The 672 x 384 input causes the regression, not the margin method.** With it, SegFormer labels 70.6-71.5% of the pothole as focus sidewalk, against 50.9% before. Sobel removes about 20% of the pothole either way, so the extra pixels stay walkable.
- **Lowering Sobel's threshold does not fix it.** Results are in `runs/segformer_pipeline_eval/sobel_threshold_sweep.log`:

| Front end, Sobel threshold | Sobel false flags | Refined IoU | Pothole walkable |
|---|---|---|---|
| Old, 2.45 | 2.81% | 0.6496 | 31.6% |
| 672 x 384 + margin, 2.45 | 2.43% | 0.6764 | 50.5% |
| 672 x 384 + margin, 2.30 | 2.98% | 0.6734 | 46.9% |
| 672 x 384 + margin, 2.15 | 3.65% | 0.6696 | 43.2% |
| 672 x 384 + margin, 2.00 | 4.49% | 0.6648 | 39.0% |
| 672 x 384 + margin, 1.85 | 5.54% | 0.6586 | 33.1% |

Getting the pothole back to about a third needs threshold 1.85, which roughly doubles false flags and gives back most of the IoU gain.

### Overlay review
`python3 render_segformer_compare.py` wrote `runs/segformer_compare/overlay_walkA.mp4` and stills of the target frames. It shows three panels side by side: old, margin at 512 x 512, and margin at 672 x 384. Nothing else differs between panels. Panel 1 matched the last full run (`runs/sidewalk_final_color/per_frame.csv`) on all 600 frames.

On review, **margin + 672 x 384 performed poorly in accuracy, and margin alone looked identical to the old model.** That matches the measurements: margin changes about 0.4% of focus pixels, all at edges.

## Test 4: is margin faster, and will it hold on other data?
Small samples only. Results are in `runs/segformer_input_eval/margin_check.log`.

### Speed: real code paths, from logits to the mask `process()` uses
The old path includes copying the full-size class map to the CPU and `np.isin`, which the first benchmark left out. Runs were interleaved over 20 frames.

| Device | Condition | Old | Margin | Saved | Speed-up |
|---|---|---|---|---|---|
| MPS | hot loop | 1.24 ms | 0.56 ms | 0.69 ms | 2.2x |
| MPS | 2 FPS, idle between frames | 2.78 ms | 1.29 ms | 1.49 ms | 2.2x |
| CPU | hot loop | 2.21 ms | 0.45 ms | 1.76 ms | 4.9x |
| CPU | 2 FPS, idle between frames | 2.04 ms | 0.56 ms | 1.48 ms | 3.7x |

That is about 2-4% of a 46 ms frame. The gain is real but small.

### Robustness: 48 frames (40 spread over walk A + 8 target frames) under simulated changes
"Off-documented" is the share of pixels differing from bilinear-then-argmax, the recipe SegFormer was trained with.

| Condition | Focus IoU old to margin | Frames better / worse | Off-documented: old / margin | Holes per frame: old / margin | Hazard px in focus: old to margin |
|---|---|---|---|---|---|
| None | 0.670 to 0.673 | 10 / 1 | 0.68% / 0.16% | 4.74 / 3.87 | 29.3% to 32.2% |
| Dark (x0.45) | 0.651 to 0.654 | 4 / 1 | 0.64% / 0.14% | 4.72 / 3.68 | 34.7% to 35.3% |
| Bright (gamma 0.6) | 0.685 to 0.693 | 9 / 1 | 0.67% / 0.15% | 5.55 / 4.32 | 35.4% to 35.9% |
| Blur (Gaussian 7, SD 2) | 0.479 to 0.481 | 1 / 1 | 0.60% / 0.12% | 2.55 / 2.02 | 32.7% to 34.1% |
| JPEG quality 15 | 0.586 to 0.592 | 7 / 0 | 0.81% / 0.17% | 5.83 / 4.64 | 30.8% to 31.1% |
| Noise (SD 12) | 0.617 to 0.619 | 4 / 0 | 0.69% / 0.15% | 5.28 / 3.60 | 24.6% to 25.6% |
| Low-res camera (320 x 180) | 0.563 to 0.566 | 5 / 3 | 0.68% / 0.14% | 4.49 / 3.81 | 36.7% to 37.2% |
| Warm colour cast | 0.631 to 0.630 | 6 / 2 | 0.73% / 0.15% | 7.00 / 5.38 | 41.0% to 42.5% |
| Mirrored | 0.710 to 0.712 | 10 / 0 | 0.68% / 0.15% | 5.68 / 4.19 | 24.9% to 25.3% |

- **Margin was equal or better in 8 of 9 conditions.** Warm cast was a tie. Overall, 56 frames improved and 9 got worse.
- **It stayed 4-5x closer to the training recipe in every condition.** Nothing in it is tuned to this data: the threshold of 0 is the argmax decision itself.
- **One consistent cost:** about 20% fewer holes, and slightly more hazard pixels inside the focus area (+0.4 to +2.9 points). Smooth edges close some small gaps the blocky mask left open. In the full model the checks absorb most of this (Test 3: pothole 31.7% against 31.6%), but the cone was 6.2% against 5.2%.

## Adopting the margin method
- **`sidewalk_system.py`:**
  - New `_preprocess()` (the same resize and normalisation as before) and `sidewalk_margin()`.
  - `segment()` returns the boolean mask.
  - `process()` drops `np.isin`.
  - Timing keys and output keys are unchanged.
- **`eval_segformer_pipeline.py`:** its `FrontEnd` keeps the old argmax path as `"current"` for comparisons, and `render_segformer_compare.py` uses it.

### Checks
- **10 frames:** the new mask is pixel-identical to the margin method tested above. At every output cell it makes the argmax decision. It differs from the old method on 0.3-1.0% of pixels. The focus area stays inside the sidewalk mask.
- **Command line:** `sidewalk_system.py --source clip.mp4 --debug-video panel.mp4` on frames 129-138 printed JSON for every frame (about 38 ms per frame after warm-up) and wrote the panel.
- **Full model:** `python3 simulate_sidewalk.py --walks A --out runs/sidewalk_final_margin`

| | Before (`runs/sidewalk_final_color`) | After (`runs/sidewalk_final_margin`) |
|---|---|---|
| IoU / precision / recall (live and no-lag) | 0.6496 / 0.8629 / 0.7243 | **0.6529** / 0.8613 / 0.7296 |
| Raw-model baseline IoU | 0.5338 | 0.5338 (identical on every frame) |
| Focus covers near-field sidewalk | 75.5% | 76.0% |
| Postprocess + masks (mean) | 2.33 ms | **1.44 ms** |
| Latency mean / p95 / max | 46.2 / 54.7 / 61.6 ms | 45.7 / 55.5 / 76.0 ms |
| Sobel / colour mean | 5.6 / 2.7 ms | 6.1 / 3.2 ms |
| Colour objects | 459 | 488 |
| Dropped frames | 0% | 0% |

- **The output files are compatible.** `summary.json` has the same keys and `per_frame.csv` the same columns and 600 rows.
- **Per-frame changes:** 534 of 600 frames moved by under 0.01 IoU, 47 rose and 19 fell. The largest swings (-0.29 to +0.35) are frames where `_largest_region` picks a different piece of a fragmented sidewalk.
- **Timing noise:** Sobel and colour code did not change, so their small time increases are run-to-run variation. The 76 ms maximum is a single frame.

## Recommendations
- **Keep the 512 x 512 input** unless an aspect-preserving input is paired with a fix for texture hazards inside sidewalk-labelled regions and tested on more potholes and covers.
- **Report hazard walkability for every front-end change,** alongside IoU.
- **Try fp16 next.** It cut inference by 35% and changed 0.01% of focus pixels, but it has not been run through the full model and the hazard check.
- **Move normalisation to the GPU.** This cuts preprocessing from 3.2 to 0.9 ms in a hot loop, and likely more at 2 FPS. Untested in the live pipeline.
- **Stabilise the focus area on fragmented frames.** For example, prefer the region nearest the bottom centre of the frame, or keep the previous frame's choice. This affects about 146 of 600 frames.
- **Handle frames with no Sobel reference** (349-369, 518-525), for example by falling back to the previous frame's reference.
- **Test on a second, genuinely different walk.** The robustness test used synthetic changes to one walk.

## Limits
- **One walk.** Walk B only mirrors it, so it is not independent data.
- **Few hazards.** One pothole on 4 frames (with the cone standing in it), one manhole cover on 1 frame, and racks on 3 frames. The hazard results are consistent in direction but come from very few examples.
- **IoU differences under about 0.5 points are within noise for this data.** Frame-to-frame results are strongly correlated.
- **Synthetic robustness.** Test 4's conditions stand in for other cameras and datasets but are not a real second dataset, and they use 40-48 frames.
- **One machine, mostly hot-loop timings.** Live 2 FPS timings are higher. Relative differences held in both.
- **The training data is gated,** so its resolution and aspect ratio, and why aspect-preserving inputs score higher here, could not be confirmed.

## Files
- **Scripts:**
  - `eval_segformer_input.py` (Test 1 and timing)
  - `eval_segformer_pipeline.py` (Test 3)
  - `render_segformer_compare.py` (overlay)
- **Outputs:**
  - `runs/segformer_input_eval/`
  - `runs/segformer_pipeline_eval/`
  - `runs/segformer_compare/`
  - `runs/sidewalk_final_margin/`
- **One-off analyses:** the combined settings, per-frame diagnosis, threshold sweep, front-end split and Test 4 scripts were run from a scratch folder; their logs are saved in the output folders above.
