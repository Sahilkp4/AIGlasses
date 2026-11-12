# Feature Tracking Tests: Summary

**Dates:** November 10-11, 2026

**Outcome:** The tracker is integrated into the full model, on by default, and no longer costs accuracy.

**Full walk A** (600 frames, live at 2 FPS):

| | Live IoU |
|---|---|
| New tracker | 0.653 |
| No tracking | 0.653 |
| First tracker | 0.647 |

- **Against no tracking:** 389 frames are better and 55 worse. The worst frame loses 0.014.
- **Against the first tracker:** the large losses from coasting boxes are gone. Frame 344, for example, goes from 0.532 to 0.740.
- **Cost:** 2.3 ms per frame.
- **Why IoU doesn't rise:** every current flag is still removed from the walkable area at once, whether the tracker confirms it or not.

**Frames 1-23**, against the motion measured from the ground:
- **Accepted matches:** 161; 3 rejected as outliers.
- **Coasting boxes** are off by 3.0 px.

**Hand-marked targets:**
- **Cover:** confirmed on frame 21.
- **Pothole:** confirmed on frames 129-133, but lost on frame 135, where it is barely flagged.

The tracker (`feature_tracking.py`) links the boxes flagged by the Sobel and colour checks across consecutive frames. It confirms boxes that persist, drops boxes that do not, and discards confirmed boxes that do not grow as an approaching obstacle should. Motion is predicted separately in a grid of 12 cells. Each cell has a fixed straight-walk direction, and its speed is voted.

The design changed many times. The biggest gains were these:
1. **Assuming the wearer walks straight ahead.**
2. **Measuring moves on box edges** instead of box centres.
3. **Replacing a single row-scaled motion** with one motion per grid cell, after the motion measured from the ground showed that movement grows with about the square of the distance below the horizon.
4. **Bounding each cell's vote** around its default.
5. **Keeping boxes that split out of the vote.**
6. **Ending coasting** once a box lands on clean sidewalk.

## Goal
Sobel and colour flags have no memory between frames. A false flag (a texture seam, chroma noise, a Sobel streak) appears and disappears, while a real obstacle persists and grows as the wearer approaches. The goal was to use that difference in a light tracker over a sliding window of two processed frames:
- **Confirm** obstacles seen in 3 consecutive frames.
- **Forget** boxes with no successor.
- **Discard** confirmed boxes that sit on sidewalk and do not grow.
- **Keep removing** confirmed obstacles from the walkable area while they are briefly lost.

## Final design (`feature_tracking.py`)
| Setting | Value |
|---|---|
| Input | Union of the Sobel `anomaly` and colour `color_anomaly` masks |
| Boxes | 8-connected blobs of at least 50 px (`MIN_OBJECT_PX`) |
| Grid | 4 regions from the bottom up: R1 = 35% of the height, R2 = 18%, R3 = 9%, R4 = 38% (`REGION_FRACS`). 3 columns: 25%, 50%, 25% of the width (`COLUMN_FRACS`). 12 cells in all. At 960x540, R1 is rows 351-540, R2 254-351, R3 205-254, R4 0-205; the columns are x 0-240, 240-720, 720-960 |
| Cell of a box | The cell holding most of its area (ties: the lower region). A track's motion comes from the cell of its box in the previous frame |
| Motion measure | Change of the bottom edge (down); gap between the two boxes' horizontal ranges (sideways, 0 when they overlap). Box centres are not used anywhere |
| Motion model | Per cell: a fixed straight-walk direction and a voted speed, in px per camera frame at the region's bottom edge `y_ref`. Inside a region the move is scaled by `s(y) = ((y − focus top) / (y_ref − focus top))²`, floored at 0.02, times the frame gap; `s = 1` for a region wholly above the focus area |
| Straight-walk defaults | Away from the point the wearer walks towards (frame centre, horizon at row 264): see the table below |
| Straight-walk cone | Every pairing must point within 60° of its cell's straight-walk direction (`DOWN_ANGLE_DEG`); moves under 3 px pass |
| Voters | Established tracks (streak at least 3, not coasting) whose old and new boxes are in the same cell. Left out: tracks the size check would discard, tracks that split, and tracks at or heading past the frame's bottom edge (box ending within 8 px of it, or predicted box reaching past it) |
| Vote | Per cell, the weighted median of the voters' down moves divided by `s(y)`; weight = streak / 3. From 1 vote up (`MIN_PAIRS = 1`) |
| Vote bound | The voted speed is held to 1/1.5 to 1.5 times the cell's default (`VOTE_BOUND`), and the bounded speed becomes the new default |
| Without a vote | The cell keeps its default: its last bounded vote, or its straight-walk default before its first vote |
| Matching | Overlap with the padded predicted box only, ranked by shared pixels. There is no distance fallback |
| Padding | 4 px + 0.5 x the expected move, on every side |
| Overlap candidate | Shared area of at least 20% of the smaller box (`MIN_OVERLAP`) |
| Outlier cut-off | 3 x max(cell spread, 4 px, 25% of the expected move), plus a 60° direction test against the expected move |
| Split | At least 2 boxes smaller than the track's box overlap its padded prediction (`MIN_SPLIT_PIECES`): the streak restarts at 1 (a guaranteed track stays guaranteed), the size history restarts, and the track does not vote |
| Confirmation | Streak of 3 (`TRACK_FRAMES`); guaranteed for the rest of the track's life |
| Size check, matched | A guaranteed track is discarded if neither its height nor its width grew over its last 3 matched frames, and at least 50% of its box is labelled sidewalk |
| Unmatched tentative | Forgotten at once |
| Unmatched guaranteed | Coasts with its cell's motion. Discarded if at least 50% of its predicted box is clean sidewalk (labelled sidewalk, no Sobel or colour flags), **or** if neither its height nor its width grew over its last 3 matched frames. Forgotten once outside the frame. Its box stays removed from the refined masks while it coasts |
| Memory between frames | The tracks, and each cell's default (its last bounded vote) |

**Straight-walk defaults at 960x540** (`--forward-px 120 11 2 2`). The sideways part is the region's speed × (column centre − frame centre) / (`y_ref` − horizon).

| Region | Left | Centre | Right | Angle from straight down |
|---|---|---|---|---|
| R1 | (−157, 120) | (0, 120) | (+157, 120) | 52° outward at the sides |
| R2 | (−46, 11) | (0, 11) | (+46, 11) | 76° outward at the sides |
| R3, R4 (above the horizon) | (0, 2) | (0, 2) | (0, 2) | straight down |

Lower regions get bigger moves, both sideways and down, at smaller angles from straight down. This matches the ground motion measured on frames 3-23: the R1 side columns moved at 51-54° from straight down, and R2-right at 76°.

**Command line** in `sidewalk_system.py` and `simulate_sidewalk.py`:
- `--tracking/--no-tracking`: on by default.
- `--track-frames`: default 3.
- `--forward-px R1 R2 R3 R4`: default 120 11 2 2. Each value seeds all three columns of its region.

The live loop's JSON line reports `tracks`, `guaranteed`, `coasting`, `split`, `tracked_frac` and `tracking_ms`.

**The defaults are tuned for this dataset**, whose frames are about one walking step apart (2 FPS). A live camera at 30 FPS needs them about 15 times smaller (`--forward-px 8 0.7 0.13 0.13`).

## How the tracker works
### Terms
- **Box:** the bounding box of one flagged blob in the current frame.
- **Track:** a box followed across frames. It holds its box, its **streak** (consecutive frames matched, 1 when new), a guaranteed flag, a coasting flag, and its last 3 box heights and widths.
- **Tentative:** a track with a streak below 3.
- **Guaranteed:** a track that reached a streak of 3. It stays guaranteed for the rest of its life, even if a split resets its streak.
- **Coasting:** a guaranteed track with no match this frame, moved by its cell's motion.
- **Cell:** one of the 12 grid cells. Each has its own speed, outlier spread and default.

### Three assumptions run through every step
**1. The wearer walks forward in a straight line**, towards the frame centre.
- **What that implies:** fixed ground objects move away from that point. They go down in the centre column, and down and outward in the side columns.
- **Rejected pairings:** any pairing whose move points more than 60° away from its cell's straight-walk direction is treated as a wrong match or a false feature, and is never formed.
- **Turning or walking backwards:** nothing matches. Tracks are dropped or coast, and new boxes start new tracks once forward walking resumes.

**2. Moves are measured on box edges, never on centres.** A blob's centre jumps whenever it merges with neighbouring flags, splits or grows.
- **Down:** the change of the bottom edge. This is where an obstacle meets the ground.
- **Sideways:** the gap between the two boxes' horizontal ranges, which is 0 when they overlap.

  Measuring the sideways move by the middle of the bottom edge was tried (November 11) and cut accepted matches nearly in half. The outward drift therefore comes from each cell's fixed direction, not from measurement.

**3. Ground motion depends on where it is in the frame.** The measured ground motion grows with about the square of the distance below the horizon, and drifts outward at the sides. One motion cannot fit every row, so each cell has its own. Inside a region, the move is scaled by the square of the distance below the focus top.

### Step by step, each frame
**1. Boxes.** The Sobel and colour flags are joined and split into 8-connected blobs. Blobs under 50 px are dropped, and their pixels return to the sidewalk.

**2. Vote on each cell's speed.**
1. **Who votes:** established tracks (streak at least 3, not coasting).
2. **Each voter:**
   - **Matching:** it is matched as in step 3, using its cell's default motion as its prediction.
   - **Vote:** its down move divided by `s(y)` and the frame gap is its vote on the cell's speed.
   - **Weight:** streak / 3, so a box tracked across 6 frames counts double one tracked across 3.
3. **Left out:**
   - a voter whose matched box is in a different cell from its previous box (it votes again once it has spent two frames in a row in one cell);
   - a voter that split (step 5);
   - a voter the size check would discard this frame;
   - a voter whose box ends within 8 px of the frame's bottom edge, or whose predicted box reaches past it. The box is clipped there, so its move is cut short.
4. **Result:**
   - **Speed:** the weighted median of the votes.
   - **No information:** a speed that isn't downward.
   - **Bound:** the speed is held to 1/1.5 to 1.5 times the cell's default.
   - **Motion:** the cell's direction at that speed, which also becomes the new default.
   - **Outlier spread:** 1.4826 x the weighted median distance of the voters' down moves from the speed.
5. **Without a vote:** the cell uses its default.

**3. Matching.**
1. **Predicted box:** the track's box, moved by its cell's motion (scaled for its row), then padded by 4 px + 0.5 x the expected move.
2. **Candidates:** current boxes that share at least 20% of the smaller box with the padded prediction, ranked by shared area in pixels.
   - **When a box splits,** the largest piece keeps the track.
   - **When several boxes merge,** the largest old box keeps the track.
3. **Straight-walk rule:** a candidate whose move points more than 60° from the cell's direction is discarded and counted as off-direction.
4. **Pairing:** one-to-one, best candidates first. A box that doesn't overlap the prediction is never matched.

**4. Outlier test.**
- **Residual:** the pair's distance from the predicted box (sideways gap and bottom-edge difference).
- **Rejected if:**
  - the residual is more than 3 x max(the cell's spread, 4 px, 25% of the expected move);
  - or the move points more than 60° from the expected move (when both are at least 3 px).
- **After a rejection:** the old track is treated as unmatched, and the new box starts a new track.

**5. Accepted pairs.**
- **Update:** the track takes the new box and records its height and width, and its streak goes up by 1. At 3 it becomes guaranteed.
- **Split:** if at least 2 boxes smaller than the track's old box overlap its padded prediction, the streak restarts at 1 and the size history restarts with the new box. A guaranteed track stays guaranteed.
- **Size check:** a guaranteed track is discarded if neither its height nor its width grew over its last 3 matched frames, and at least 50% of its box is labelled sidewalk. Its pixels go back to the refined masks in the same frame.
- **New tracks:** boxes from rejected pairs and unmatched boxes start new tentative tracks at streak 1.

**6. Unmatched tracks.**
- **Tentative:** forgotten at once. It is counted as *safe* if at least 50% of its predicted box is clean sidewalk, otherwise as *dropped*; the counts only feed the overlay.
- **Guaranteed:** moves by its cell's motion; its cell is re-read every frame. It is then:
  - **forgotten** if the box is entirely outside the frame;
  - **discarded** if at least 50% of the box is clean sidewalk, or if it had stopped growing;
  - **otherwise kept coasting**, with its box cut from the refined masks.

  A coasting track is matched every frame like any other, so it re-locks if its object is flagged again where predicted.

**7. Output.** The refined masks are the sidewalk and focus masks minus `tracked_anomaly`. That mask holds:
- every current kept blob, removed at once whether tentative or guaranteed, minus discarded ones;
- the boxes of coasting guaranteed tracks.

The tracker also returns:
- the tracks;
- each match: previous box, current box, predicted point, accepted or rejected, split or not;
- the vote details for each cell, with its exclusions;
- each cell's motion and its source (vote, last vote or default);
- the ids of tracks that split;
- the counts (lost, off-direction, safe, dropped, discarded);
- `tracking_ms`.

### What happens to a box
| Situation | Result |
|---|---|
| New box, no predecessor | New tentative track, streak 1; its pixels are removed from the walkable area at once |
| Matched and accepted | Streak + 1; guaranteed at 3 |
| Matched, but split into 2+ smaller boxes | Keeps the largest piece; streak back to 1 (guaranteed stays guaranteed); no vote this frame |
| Matched, rejected as an outlier | Old track treated as unmatched; the box starts a new track |
| Pairing points away from the cell's walk direction | Never formed (off-direction) |
| Moved into another cell | Matched normally; votes again after two frames in one cell |
| Tentative, no match | Forgotten |
| Guaranteed, no match, predicted box mostly clean sidewalk or not growing | Discarded |
| Guaranteed, no match, otherwise | Coasts; its box stays removed; can re-lock |
| Guaranteed, matched, not growing, on sidewalk | Discarded; its pixels return to the walkable area |
| Guaranteed, matched, not growing, in a hole | Kept |

## Main findings
- **Assuming a straight walk was the turning point** (November 10). Before it, the motion estimate from noisy boxes swung by hundreds of pixels sideways and flipped direction between frames.
- **Box centres, and bottom-edge middles, are unreliable for blobs.**
  - The cover's centre "moved" 44 px sideways only because its blob merged with flags to one side.
  - Measuring sideways moves by the middle of the bottom edge cut accepted matches on frames 1-23 from 160 to 92, and raised outlier rejections from 4 to 42.
  - Bottom edges and overlap follow growing and merging objects.
- **One row-scaled motion cannot fit the frame.**
  - **Shape:** ground motion grows with about the square of the distance below the horizon (best fit power 1.9, error 5.0 px, against 14.6 px for the linear row scale).
  - **Horizon:** the fitted horizon (rows 264-272) matches the focus top (median 264).
  - **Effect:** under the linear scale, a vote from a box high in the frame under-predicted low boxes 3-5 times, and coasting boxes moved 0.62 of their true distance.
- **A slow motion estimate feeds itself.**
  - **Smaller window:** a slow prediction also shrinks the padding, so the true box is out of reach.
  - **Wrong match:** the track pairs with a neighbouring thin paving-line streak that barely moved, which votes slow again.
  - **Fix:** bounding each cell's vote to 1/1.5 to 1.5 times its default broke the cycle. Coasting boxes went from 0.16 to 0.86 of their true move.
- **Boxes at the frame's bottom edge measure short moves.** The voter that first pulled the bottom region down (frame 4) measured 9 px against 90 px true, because its object was leaving the frame. Flags usually stop a few rows short of the edge, so a "touching the edge" test alone never fired; the predicted-box test catches it.
- **Splits fake motion.** A large Sobel area that breaks into pieces keeps its track on the largest piece, whose edges differ from the whole. Keeping split tracks out of the vote halved coasting (52 → 25 box-frames on frames 1-23) and stopped R2-centre drifting to 35 (true about 12).
- **Coasting was the whole IoU loss.**
  - **Before:** it cost the first tracker 0.006 IoU over the full walk, and up to 0.21 on single frames.
  - **Now:** ending coasting on clean sidewalk, or when the box had stopped growing, removes those losses.
  - **Price:** a real obstacle that is briefly unflagged is dropped instead of held, as happened to the pothole on frame 135.
- **Detection still limits the tracker.**
  - **Few true successors:** only 44% of flagged boxes have a true successor in the next frame, against 26% by chance.
  - **Missed objects:** the colour check cannot see grey objects.
  - **Racks:** they break into changing fragments and are only ever tentative.
- **Cost is small.** The tracker takes 2.3 ms per frame on average, 4.7% of the pipeline's latency.

## Evaluation method
### True motion from a ground-plane homography
The labels do not mark obstacles, so tracking was measured against an independent estimate of the true motion.

**Fitting the homography:**
- ORB features (4,000 per frame) inside the dilated sidewalk mask were matched between consecutive frames, with a ratio test.
- A homography was fitted with RANSAC (4 px). Inliers ranged from 175 to 641 per frame pair on frames 1-23.
- Frame pairs 1→2 and 2→3 barely move, so calibrations use frames 3-23.

**What it was used for:**
- **True move of any point:** the homography gives the true move of any ground point, and of a box's bottom-edge middle.
- **Coasting accuracy:** a coasting box's predicted move is compared with the true move of its bottom-edge middle. The report gives the median ratio of predicted to true down move, and the median error in pixels.
- **Row-scale test:** for sampled sidewalk points, the true down move divided by the model's row scale should be the same on every row if the model is right.

**When a match counts as true:**
- **Centre rule** (used until edge-based matching): the matched box's centre is within 25 px of where the old box's centre really went.
- **Box rule** (used from edge-based matching on): the matched box covers at least 20% of where the old box really went (the old box's corners moved by the homography).

The homography is exact only for flat ground, so flags on cars and kerbs are predicted roughly.

### Hand-marked targets (`gmm_targets.json`)
- **Pothole and cone:** frames 129, 131, 133 and 135.
- **Cover:** frame 21.
- **Racks:** frames 445, 451 and 463.

Targets were checked by running the full model in live order from frame 1 and listing the tracked boxes that cover at least 20% of each target.

### Review overlays (`render_tracking.py`)
| Mode | Shows | Output |
|---|---|---|
| default | Previous frame with its tracks; current frame with matches, predicted points and motion arrows per column | `runs/tracking_review/` |
| `--angles 60 105` | Current frame side by side under each `DOWN_ANGLE_DEG` | `runs/tracking_review_angles/` |
| `--arrows` | Previous boxes (white), current boxes, an arrow per pairing (green accepted, red rejected), predicted boxes (cyan), discarded boxes (orange) | `runs/tracking_arrows*/` |
| `--current` | Only the current tentative (yellow) and guaranteed (red) boxes, labelled `id:streak` | `runs/tracking_current/` |
| `--status` | Every box by status: tentative (yellow), guaranteed (red), coasting (dashed red, `coast`), discarded (orange), split (`split`). Grid lines with each cell's motion and source (v vote, l last vote, d default). Header with counts, the vote and the matches | `runs/tracking_status*/` |

Each mode writes stills, an H.264 `.mp4` and a VP8 `.webm`. VS Code cannot play H.264, so use the `.webm` there.

### Replay scripts (session scratchpad, not in the repository)
The masks of frames 1-23 were cached once, and the tracker was replayed on them, which takes seconds. The scripts:

| Script | What it does |
|---|---|
| `cache.py` | Caches the masks of frames 1-23 |
| `homog.py` | Fits the homographies and runs the row-scale test |
| `replay*.py` | Measures coasting accuracy and votes against the truth |
| `sweep_bound.py` | Sweeps the vote bound |
| `ablate.py` | Switches off one rule at a time |
| `targets_check.py` | Checks the hand-marked targets |

The synthetic tests were run inline and are not saved either.

## Development tests
All results in this section are on frames 1-23 of walk A unless stated otherwise.

### November 10: from the first version to edge-based matching

#### First version: median of matched pairs
**Design.** Tracks were matched to the nearest box centre within 80 px, then the motion was taken as the median of the matched pairs. Flags were still removed at once.

**Scored simulation** (walk A, frames 1-200):

| Run | IoU | Precision | Recall | Tracking ms |
|---|---|---|---|---|
| No tracking | 0.778 | 0.940 | 0.819 | - |
| First version | 0.776 | 0.940 | 0.816 | 1.9 |
| After the fixes below | 0.769 | 0.939 | 0.810 | 2.5 |

**Fixes.**
- **The failure:** the motion estimate blew up to about (−15,000, 20,000) and stuck. Tracks near the top of the focus area divided by a row scale of about 1/540, and once predictions left the frame, nothing matched to correct it.
- **The fix:** a floor of 0.1 on the row scale, and dropping an estimate after 5 frames without an update.

IoU fell because tracking could only remove more pixels (coasting boxes), never give any back.

#### Direction vote from all box pairs
**The problem:** matches picked a closer box even when the true box had moved towards the camera.

**The change:** a consensus vote ran before matching. Every track-box pair within 200 px proposed a motion, and the motion that most other pairs agreed with won.

**Results:**
- **The vote flipped sideways every frame** (+358, +213, −132, +405, −462 ...).
- **With a fixed tolerance,** the best support was 3-7 of 100-330 candidate pairs, close to chance.
- **In a synthetic test with decoy boxes,** the vote found the true motion.

#### Who votes, and the direction limit
- **Voting by established tracks only:** the vote ran on only 3 of 23 frames.
- **Direction limit:** `MAX_ANGLE_DEG` at 60 and 90 gave identical results. The distance test already limits accepted moves to about 49°.

| Vote pool | Median pool | Votes | Correct | Accepted / rejected |
|---|---|---|---|---|
| All tracks | All tracks | 22 | 4 (18%) | 163 / 84 |
| Established | All tracks | 3 | 1 | 158 / 84 |
| All tracks | Established | 8 | 4 (50%) | 113 / 140 |
| Established | Established | 3 | 1 | 160 / 83 |

#### Ground truth: is the problem matching or detection?
| Measure | Result |
|---|---|
| Boxes with a flagged box within 25 px of their true next position | 146 of 335 (44%) |
| Same, with the true distance in a random direction (chance) | 26% |
| Tracker matches to the true box (centre rule) | 80 of 243 (33%) |
| Accepted matches, true / wrong | 54 / 106 |
| True matches rejected as outliers | 26 |

Both were problems. Matching picked the wrong box two times in three, and fewer than half the boxes had anything to match.

#### Straight-walk rule (cone around straight down)
| Cone | True moves allowed | Matches to true box | Accepted, true / wrong | True rejected |
|---|---|---|---|---|
| None | 146 / 146 | 80 of 243 (33%) | 54 / 106 | 26 |
| 60° (kept) | 92 / 146 | 67 of 176 (38%) | 56 / 85 | 11 |
| 105° | 138 / 146 | 82 of 242 (34%) | 47 / 101 | 35 |

37% of true moves failed the 60° cone around straight down. Most were obstacles spreading outward near the image edges. This was fixed on November 11 by centring the cone on each cell's direction.

#### Forgetting tentatives, size check, default motion
- **Forgetting tentatives:** unmatched tentatives were forgotten at once (166 forgotten).
- **Size check:** guaranteed tracks that didn't grow on sidewalk were discarded (12 tracks).
- **The side effect:** with fewer surviving tracks no frame could vote, and coasting boxes piled up to 15.
- **Default motion:** a default of (0, 120) row-scaled, calibrated from the flagged boxes, brought matches to the true box up to 48%.

#### Pothole and cover with centre-based matching
- **Pothole:** confirmed, but true matches were rejected on frame 129 (moved (−12, 10) against a predicted (0, 20)), and on frame 135, where its flags broke into fragments.
- **Cover:**
  - **Detection:** 23% Sobel-flagged on frame 21, along its lower rim only, and 0% colour-flagged.
  - **Merging:** its rim merged with a paving-line band.
  - **Rejection:** its centre "moved" 44 px sideways when the merged box widened, so it was rejected one frame short of guaranteed.

| Frame | Pothole px labelled sidewalk | Sobel-flagged | Colour-flagged |
|---|---|---|---|
| 129 | 96% | 43% | 7% |
| 131 | 94% | 85% | 4% |
| 133 | 54% | 54% | 0% |
| 135 | 42% | 8% | 0% |

#### Edge-based geometry, overlap matching, final vote of November 10
**What changed:**
- **No centres:** they were removed from every step.
- **Matching:** switched to overlap with the padded predicted box.
- **Overlap ranking:** took three tries; shared pixels was kept.
- **Size check:** switched from area to height.
- **Voters:** only the established tracks with the highest streak voted.
- **No carry-over:** nothing was reused between frames.

| Version | Matches | To true box (box rule) | Accepted, true / wrong | True rejected | Discarded |
|---|---|---|---|---|---|
| Default motion, centres | 141 | - | - | 27 | - |
| Edge-based, consensus vote | 195 | 107 (55%) | 83 / 58 | 24 | 12 |
| Edge-based, final vote | 204 | 112 (55%) | 101 / 52 | 11 | 9 |

**Targets:**
- **Pothole:** one guaranteed track from frame 126 to 135.
- **Cover:** guaranteed on frame 21.

**Full walk A with that version:**
- **IoU:** 0.647, against 0.653 without tracking.
- **The losses:** coasting boxes cut 27-41% of small focus areas on frames 344 and 451-452 (see "Full live run").

### November 11: from one motion to a 12-cell grid

#### Change set 1: size check, carried-over vote, no fallback, weighted vote
**What changed:**
- **Size check:** a guaranteed track is kept if its height **or** its width grows, and discarded only when neither grows. A first version read this backwards and was corrected the same day.
- **Default motion:** each successful vote became the default, instead of reverting to (0, 120).
- **No fallback:** the 80 px distance fallback was removed, so only overlap matches.
- **Weighted vote:** all established tracks vote, weighted by streak / 3, with a weighted median.

**Results** (status overlay, `runs/tracking_status/`):
- **Box-frames:** 282 tentative, 62 guaranteed, 83 coasting, 6 discarded.
- **Cover:** guaranteed on frame 21.
- **Slow votes stuck:** the first votes were very slow (15-21 against 120), and frames 7-9 reused 15. Coasting boxes piled up to 8 at once, and one (track 13) coasted on open sidewalk from frame 8 to frame 21.

#### Why coasting boxes lagged: the row scale
The ground motion was measured on all sidewalk points (`homog.py`). If the linear row scale `r(y) = (y − top) / H` were right, the last column would be constant.

| Rows | True down move | `r(y)` | True move / `r(y)` |
|---|---|---|---|
| 300-330 | 4.2 | 0.100 | 41 |
| 360-390 | 17.3 | 0.189 | 92 |
| 420-450 | 41.9 | 0.300 | 143 |
| 480-510 | 81.9 | 0.411 | 200 |
| 510-540 | 107.5 | 0.467 | 230 |

**Fitted shape:**
- **Best fit:** `dy ∝ (y − 272)^1.9`, error 5.0 px.
- **Squared, power fixed at 2:** `dy ∝ (y − 264)²`, error 5.0 px.
- **The linear row scale:** best constant 176, error 14.6 px.

After dividing by the squared scale, the speed across bands varies 1.1x; after the linear scale, 3.4x.

**Replay of the tracker against the truth:**
- **Votes high in the frame were accurate,** within 1-3 px on rows 314-419. Under the linear scale they still under-predicted low boxes: a box at row 350 correctly reports about 70, which predicts about 30 px at row 500, where the true move is about 90.
- **Coasting boxes moved 0.62 of their true distance** (median of 83 steps). Examples:

| Track and frame | Predicted | True |
|---|---|---|
| Track 7, frame 8 | 7.6 px | 89.5 px |
| Track 101, frame 19 | 21.3 px | 104.5 px |

- **A feedback loop.** The slow prediction gives a small padding, so the true box is out of reach, and the track matches a neighbouring thin streak that barely moved (frame 18, track 137: predicted 5.9, padding 6.9, true 34).
- **Streak weighting favoured slow boxes near the horizon,** which are easy to keep matching and moved only 1-5 px.

#### Three bands of 180 rows, with a squared scale inside each
| Band | Coasting predicted ÷ true move |
|---|---|
| Middle | 0.97 |
| Bottom | 0.07 |

- **The in-band scale was right:** the true speed at the band's bottom was flat across the bottom band (113 vs 125 between its halves).
- **The bottom band's votes were wrong:** median 26 against 119 true.
- **Two causes:**
  - **Frame-edge clipping:** on frame 4 the only voter was moving past the bottom of the frame and measured 9 px against 90 px, voting 13.
  - **Slow matches:** a slow band then matched thin streaks. In 6 of 11 votes the matched box did not cover the true position.
- **Box-frames:** 303 tentative, 41 guaranteed, 62 coasting, 6 discarded.
- **Cover:** tentative only on frame 21.

#### Four regions: 35% / 18% / 9% / 38%
Defaults from the homography: R1 120, R2 11, R3 and R4 2 (above the horizon). R1 then had more voters:
- **Frames 6-12:** R1 held 69-127.
- **From frame 13:** R1 collapsed to 4-30.

| Bottom rows of coasting boxes | Predicted ÷ true move |
|---|---|
| 264-360 | 0.92 |
| 360-450 | 0.08 |
| 450-540 | 0.12 |

Box-frames: 293 tentative, 49 guaranteed, 70 coasting, 8 discarded.

#### Bottom-edge exclusion and the vote bound
**Bottom-edge exclusion.** Flags usually stop short of the bottom edge (the lowest flagged row ranged from 489 to 539), so "touching the bottom" never fired. Two tests were added:
- **Margin:** a box ending within 8 px of the edge doesn't vote.
- **Predicted box:** a box whose predicted box would reach past the edge doesn't vote. This caught the frame-4 voter.

**Vote bound sweep:**

| Bound | Predicted-edge test | Coasting predicted ÷ true | Error | Rows 450-540 |
|---|---|---|---|---|
| None | off | 0.16 | 20.6 px | 0.12 |
| 2 | off | 0.64 | 11.8 px | 0.48 |
| 1.5 | off | 0.76 | 6.9 px | 0.71 |
| **1.5 (kept)** | **on** | **0.86** | **6.8 px** | **0.91** |
| 1.25 | on | 0.81 | 5.6 px | 0.86 |

- **Why 1.5:** it follows a true change faster than 1.25. In a synthetic test a drop from 120 to 40 was followed in 3 frames (120 → 80 → 53 → 42).
- **Status overlay** (`runs/tracking_status_bounded/`): 46 coasting box-frames, and the cover guaranteed again (`153:3`).
- **Left unresolved:** R2 drifted from 11 to 35.

#### Twelve cells: three columns of 25% / 50% / 25%
| | Result |
|---|---|
| Coasting predicted ÷ true move | 0.91 |
| Down-move error | 6.6 px |
| Coasting box-frames | 52 |
| Cells that ever voted | 4 of 12 (R1 left, centre and right; R2 centre) |
| Predicted sideways moves | always 0 |

The predicted sideways moves were 0 while the true ones were large: −69 px in the left column and +100 px in the right. The gap measure registers no sideways move while boxes overlap.

#### Splits
Added the split rule (see the design table).

| | Before | After |
|---|---|---|
| Coasting box-frames | 52 | 25 |
| Coasting predicted ÷ true move | 0.91 | 0.97 |
| R2-centre motion, highest (true ~12) | 35 | 16 |
| Size-check discards | 7 | 0 |

- **Splits found:** 46 split resets in 23 frames (6 on frame 20).
- **Votes excluded:** 9.
- **Split resets also come from nearby unrelated small flags:** near the bottom of the frame, the padded prediction reaches about 36 px on each side.
- **Size-check discards stopped** because size histories restart after splits.

#### Majority cells, straight-walk defaults, and the sideways measure
**What changed:**
- **Cells:** a box's cell is now the one holding most of its area.
- **Vote rule:** a box votes only if it is in the same cell in two consecutive frames (tested: a box crossing from R2-centre into R1-centre was excluded on the crossing frame and voted on the next).
- **Defaults:** they follow the straight-walk geometry (table in the design section).
- **Cone:** centred on each cell's direction.

**Measured straight-walk motion** (frames 3-23, lower third of each cell):

| Cell | Down | Sideways | Angle from straight down |
|---|---|---|---|
| R1 left | 102 | −126 | −51° |
| R1 centre | 93 | +3 | +2° |
| R1 right | 96 | +132 | +54° |
| R2 right | 8 | +34 | +76° |

Lower in the frame the sideways move in pixels grows (right column: +44 px at rows 340-390, +135 px at rows 490-540), while the angle from straight down shrinks (69° to 54°).

**Measuring the sideways move.** Variants compared, with one rule switched at a time (`ablate.py`):

| Variant | Accepted / rejected | Off-direction | Votes | Guaranteed | Coasting error, down | Sideways error |
|---|---|---|---|---|---|---|
| Bottom-edge middle for everything | 92 / 42 | 127 | 6 | 20 | 9.2 px | - |
| Gap to match, bottom-edge middle to vote | 148 / 20 | 47 | 13 | 63 | 5.7 px | 9.3 px |
| Gap for everything (direction voted) | 160 / 4 | 22 | 13 | 70 | 4.0 px | 10.7 px |
| **Gap, votes set speed only (kept)** | **167 / 2** | **22** | **13** | **71** | **3.9 px** | 9.6 px |
| Previous version (bottom-edge cells, straight-down defaults and cone) | 156 / 4 | 20 | 15 | 66 | 4.7 px | - |

- **Bottom-edge middle:** flagged blobs change shape every frame, so the middle of the bottom edge jumps.
- **Gap with voted direction:** the first vote turns a side cell straight down.
- **Speed-only votes (kept):** they keep each cell's outward direction and give the best matching.
- **Synthetic check:** a side cell's speed followed a drop from 120 to about 41 while its angle stayed at 52.5°.
- **Status overlay** (`runs/tracking_status_walkgrid/`): 279 tentative, 71 guaranteed and 25 coasting box-frames, and the cover guaranteed (`143:3`).
- **Not exercised:** all 25 coasting steps were in the centre column, so the side cells' directions were never used by a coasting box.

#### Ending coasting on clean sidewalk (OR rule)
The coasting check was changed from "not growing **and** at least 50% labelled sidewalk" to "at least 50% **clean** sidewalk (no Sobel or colour flags) **or** not growing".

| | AND (before) | OR (now) |
|---|---|---|
| Coasting box-frames | 25 | 2 |
| Coasting tracks discarded | 0 | 9 |
| Accepted / rejected matches | 167 / 2 | 161 / 3 |
| Guaranteed box-frames | 71 | 62 |
| Coasting down-move error | 3.9 px | 3.0 px |
| Coasting tracks that re-locked onto their object | 8 (after 1-2 frames) | 1 |
| Focus area cut by coasting boxes | - | 0.03% per frame (max 0.6%) |

- **Status overlay** (`runs/tracking_status_or/`): 288 tentative, 62 guaranteed, 2 coasting and 9 discarded box-frames.
- **Frame 16:** a coasting box beside a parked van (`91:4`) is kept, because the van isn't sidewalk.

## Full live run (`simulate_sidewalk.py --walks A --video`, 2 FPS, MPS)
All three runs use SegFormer, Sobel and colour.

| | New tracker | First tracker | No tracking |
|---|---|---|---|
| IoU / precision / recall (live) | **0.653** / 0.861 / 0.731 | 0.647 / 0.860 / 0.723 | 0.653 / 0.861 / 0.730 |
| Raw-model baseline IoU | 0.534 | 0.534 | 0.534 |
| Focus covers near-field sidewalk | 76.1% | 75.4% | 76.0% |
| Latency mean / p95 | 48.6 / 58.3 ms | 47.4 / 57.1 ms | 46.4 / 54.7 ms |
| Tracking stage mean / p95 | 2.30 / 3.34 ms (4.7%) | 2.27 / 3.64 ms (4.8%) | - |
| CPU per processed frame | 40.9 ms | 38.3 ms | 36.1 ms |
| Peak memory | 1.85 GB | 1.93 GB | 1.63 GB |
| Dropped frames | 0 | 0 | 0 |

**Outputs:**

| Run | Folder | Notes |
|---|---|---|
| New tracker | `runs/sidewalk_tracking_grid/` | `overlay_walkA.mp4` (H.264: open in QuickTime), `summary.json`, `per_frame.csv` |
| First tracker | `runs/sidewalk_tracking_final/` | |
| No tracking | `runs/sidewalk_no_tracking_final/` | |

### Per-frame comparison
| | Against no tracking | Against the first tracker |
|---|---|---|
| Better | 389 | 197 |
| Worse | 55 | 41 |
| Same | 156 | 362 |
| Mean change | +0.0007 | +0.0054 |
| Largest loss | −0.014 (frame 535) | −0.027 (frame 325) |
| Largest gain | +0.008 (frame 524) | +0.208 (frame 344) |

The frames the first tracker lost to coasting boxes are recovered:

| Frame | New tracker | First tracker | No tracking |
|---|---|---|---|
| 344 | 0.740 | 0.532 | 0.737 |
| 451 | 0.296 | 0.128 | 0.295 |
| 452 | 0.454 | 0.275 | 0.453 |
| 484 | 0.851 | 0.737 | 0.849 |
| 535 | 0.506 | 0.426 | 0.521 |

**Why IoU doesn't rise above no tracking:**
- **Current flags are removed regardless:** every current flag is removed at once, whether its track is confirmed or not.
- **Tracking only changes the edges:** it adds coasting boxes, which are now rare, and gives back size-check discards, which are also rare.
- **The labels can't reward it:** they count irregularities as sidewalk, so cutting out a real obstacle lowers the score.

### Hand-marked targets with the final tracker
Full model in live order from frame 1. The last column lists tracked boxes covering at least 20% of the target, as `id:streak` (g guaranteed, t tentative) with the share covered.

| Frame | Target | Share removed | Tracked boxes covering it |
|---|---|---|---|
| 21 | Cover | 22% | 147:3 g (78%) |
| 129 | Pothole / cone | 57% / 88% | 941:6 g (86% / 98%) |
| 131 | Pothole / cone | 89% / 97% | 941:8 g (100% / 100%) |
| 133 | Pothole / cone | 64% / 88% | 941:10 g (90% / 100%) |
| 135 | Pothole / cone | 5% / 1% | none |
| 445 | Racks | 8% | 2080:1 t (49%) |
| 451 | Racks | 9% | none |
| 463 | Racks | 20% | 2174:1 t (62%) |

- **Frame 135:** the pothole is lost. Only 8% of it is Sobel-flagged there. Its track went unmatched, and its predicted box landed mostly on clean sidewalk, so the OR rule discarded it instead of coasting. The first tracker still held it on frame 135.
- **Racks:** they stay tentative (streak 1), because they break into changing fragments.

## Recommendations
- **Delay the clean-sidewalk test on coasting boxes by 1-2 frames.**
  - **Today:** it ends coasting almost at once, which removes the IoU losses but drops briefly unflagged obstacles (the pothole on frame 135).
  - **On frames 1-23:** all 8 re-locks under the old rule happened within 2 frames.
- **Tighten the split test.** It fired 46 times in 23 frames, partly from unrelated small flags inside the wide padding near the bottom of the frame. Each false split costs a track its streak and its vote. Counting only pieces that lie mostly inside the predicted box is the stricter option.
- **Bring back a size check that can act.** Split resets restart size histories, so size-check discards fell to 0.
- **Most cells never vote.** On frames 1-23 only 4 of 12 cells voted, each from 1-2 boxes. R3 and R4 never vote, which is expected above the horizon. If side cells stay starved on longer walks, consider sharing votes between the cells of a region.
- **Test the side-column directions on a walk with obstacles at the sides.** No coasting box in frames 1-23 was in a side column.
- **Retune for the live camera.**
  - **Defaults:** `--forward-px` about 15 times smaller at 30 FPS.
  - **Settings sized for 2 FPS:** re-check the padding, the 8 px edge margin and the vote bound.
  - **Horizon:** `HORIZON_FRAC` (264 / 540) is calibrated on this dataset; a different camera mounting needs a new one.
- **Decide whether tentative flags should be removed at once.** Tracking can only raise IoU if it gives pixels back. Today every current flag is removed whether or not its track is confirmed.
- **Improve detection.** The colour check cannot see grey objects such as the cover. Racks should be joined into one box, for example by closing small gaps before taking blobs, so they can be tracked.
- **Handle turns.** The straight-walk defaults and cone assume the wearer walks towards the frame centre.

## Limits
- **One walk, few targets:** one pothole, one cover and one rack sequence. Most tuning used frames 1-23, which were also the frames measured, so the frame 1-23 figures are optimistic.
- **The truth is approximate:**
  - The homography is exact only for flat ground.
  - The straight-walk defaults, the region speeds and `HORIZON_FRAC` come from the same 23 frames at 2 FPS.
- **One motion per cell:** the outward drift really grows with the distance from the frame centre and with the row, but every box in a cell gets the cell's direction.
- **Coasting:** a coasting box keeps its last size, so its prediction lags if the object grows while unseen.
- **Frame rate untested:** at 2 FPS objects move 30-120 px between frames, and at 30 FPS motion per frame is far smaller. That should help, but it is untested.
- **The labels:** they count irregularities as sidewalk, so IoU cannot show whether hazards are caught.
- **Scripts not saved:** the synthetic tests and replay scripts are in the session scratchpad, not in the repository.
