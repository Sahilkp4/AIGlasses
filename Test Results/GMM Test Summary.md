# GMM Sidewalk Irregularity Tests: Summary

**Date:** November 3-4, 2026
**Outcome:** The GMM is considered a failure for now and has been removed from the code. The focus area is a success and stays.

## Goal
SegFormer labels sidewalk, but it also calls many irregularities sidewalk. We tested whether a lightweight GMM could act as a check on that label. It learned what recent sidewalk looks like, assuming most of the label is correct, and flagged what didn't fit: bike racks, potholes, cones and manhole covers.

## Why it failed
- **It only detects things that are a different colour.** It caught the cone and the bike racks because they are bright red and yellow. A real sidewalk hazard has no reason to be brightly coloured.
- **It missed the pothole**, the kind of hazard it most needed to catch. SegFormer leaves 74% of the hole labelled as sidewalk, and the GMM caught only 2.4% of those pixels.
- **It missed manhole covers** (0%). A cover's shape spans many pixels, and each pixel on its own looks like normal pavement.
- **It barely changed what the user hears.** Obstacle-ahead cues went from 20 to 21 over the walk, while GMM disruptions rose to about 293 per minute, mostly small boxes on shadows and brick seams.

## Final design tested
| Setting | Value |
|---|---|
| Training data | 25 pixels per frame, last 30 locked frames, largest sidewalk region, SegFormer confidence of at least 0.7 |
| Features | `a`, `b` (Lab colour), gradient, local brightness variation (std L) |
| Model | 4 components, diagonal covariance, refit every frame |
| Decision | Not sidewalk if 5.5 SD or more from every component (about 1% false flags) |
| Training exclusion | Flagged pixels stay out of training |

## Final results
Walk A, frames 1-600. Catch rates are the share of pixels SegFormer left as sidewalk that the GMM flagged.

| Target | Share SegFormer left as sidewalk | GMM catch rate |
|---|---|---|
| Cone | 30% | 98% |
| Bike racks | 41% | 100% |
| Pothole | 74% | 2.4% |
| Manhole covers | 100% | 0% |

| Live performance | Baseline | With GMM |
|---|---|---|
| Latency, mean / p95 | 39 / 49 ms | 53 / 66 ms |
| Dropped frames | 0 | 0 |
| Lock, offset and heading | Not applicable | Identical on all 600 frames |

Cone detection came from frame 128, compared with frame 133 from segmentation alone, but no spoken cue fired before frame 135.

## What we learned

### Features
| Feature | Finding |
|---|---|
| `a` | Essential for the cone: removing it dropped the catch rate from 99% to 5% |
| `b` | Essential for racks: removing it dropped the catch rate from 100% to 2% |
| Gradient | Small value: lifted the cone from 92% to 98% |
| Std L | Small value: lifted the pothole from 3% to 5% |
| L, edge density, row | No contribution |

### Pothole and manhole limits
- Hole pixels are dark and warm-toned, while shadowed pavers are dark and cool. Full covariance and smoothed colour did not turn that into detection (3.7% at best).
- At looser thresholds the pothole reached only 8.7% at 2% false flags and 17.6% at 5%.
- For covers, tighter thresholds flagged normal pavement more often than covers at every level. At 2x MAD, 19% of cover patches were flagged against 25% of other sidewalk.
- Orientation and local binary pattern features scored about chance (AUC 0.22-0.54).

### Thresholds
| Rule | Target | Actual false flags | Cone | Racks | Pothole | Value |
|---|---|---|---|---|---|---|
| Combined distance | 1% | 0.86% | 96.7% | 100% | 2.1% | 5.5 SD |
| Per-feature limits | 1% | 0.97% | 95.8% | 100% | 1.6% | `a` 4.64, `b` 4.78, gradient 5.05, std L 4.14 |
| Combined distance | 0.5% | 0.50% | 93.2% | 100% | 1.3% | 6.05 SD |
| Per-feature limits | 0.5% | 0.46% | 89.1% | 100% | 1.0% | `a` 4.91, `b` 5.13, gradient 5.62, std L 4.48 |

Thresholds were tuned on one half of the walk and tested on the other. Combined distance was slightly better and needs only one number.

### Training
- **Excluding flagged pixels is essential.** Without it, racks fell to 21-31% at strict thresholds. An early bug let the cone into training, where it loosened the threshold until the cone stopped being flagged.
- **Exclusion and the threshold must be tuned together.** A cutoff chosen without exclusion gave 2.9% false flags instead of 1%. At a 5% target it gave 28%, with a 43-frame lock-out where a new scene was flagged and so never learned.
- **Data quantity was not the limit.** About 248 patches per frame were usable in the patch-level tests, 88% of those scored. More data or longer windows did not help: a 120-frame window dropped racks to 14%.
- **25 pixels per frame was enough.** Training on every pixel (2.3M per fit, 34.6 s) gave cone 98.6%, racks 100%, pothole 7.4%, compared with 98.1%, 99.6%, 5.6% at the same refit rate (about 9 ms per fit).
- **Strict confidence filtering hurts.** Keeping only pixels at 0.9 confidence or above dropped the cone to 82-95%, because it narrows the model.

### Other approaches ruled out
- **GrabCut:** too slow (about 100-300 ms per frame) with no memory between frames. Rejected on design grounds, not tested.
- **MOG2 background subtraction:** assumes a still camera. Rejected on design grounds, not tested.

## What stays: the focus area
Limiting detection to the focus area worked. In the patch-level tests, using it instead of the narrower corridor raised rack detection from 82% to 90% with no increase in false flags. It holds 88.4% of labelled sidewalk in the scored rows, compared with 82.2% for the corridor.

## Limits of these results
- There is one walk, one cone, one pothole and one rack sequence, so feature choices rest on very few objects.
- The labels call irregularities sidewalk, so targets were outlined by hand or found by colour.
- Most feature and selection tests used only 1-3 random seeds.
- Run outputs were deleted to clear space, so these figures are the record of the results.

## Recommendations
- Don't rely on colour or texture statistics to find potholes. A future attempt needs information beyond single pixels, such as region shape or depth.
- Long curved edges inside the focus area may be a cheap way to find manhole covers. This was not tested.
- Decide whether flush manhole covers need reporting at all, since they are level.
