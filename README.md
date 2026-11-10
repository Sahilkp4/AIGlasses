# AIGlasses

# AIGlasses: Sidewalk Focus Area

Finds the walkable sidewalk ahead in each camera frame for AI glasses, then rejects parts of it that look like hazards.

## How it works
1. **Segmentation:** SegFormer-B0 (`tobiasc/segformer-b0-finetuned-segments-sidewalk`) labels sidewalk pixels. The largest connected sidewalk region becomes the **focus area**.
2. **Texture check:** flags pixels with stronger-than-normal edge texture, such as potholes, cracks and manhole rims.
3. **Colour check:** flags pixels whose colour differs from the frame's typical pavement colour, such as cones and bike racks.
4. **Refined focus area:** the focus area minus everything either check rejected. This is what the system treats as walkable.

## Key files
| File | Goal |
|---|---|
| `sidewalk_system.py` | Main pipeline: runs SegFormer, builds the focus area, applies both checks and outputs JSON per frame. Works on a video file or webcam (`--source`) |
| `sobel_texture.py` | Texture check: Sobel edge strength on log brightness, compared with a per-row reference learned from the focus area |
| `color_difference.py` | Colour check: Lab colour distance from the frame's median sidewalk colour |
| `simulate_sidewalk.py` | Replays the labelled dataset as a live 2 FPS camera and scores the full model (IoU, latency, per-stage timing) |
| `eval_sidewalk.py` | Scores the raw SegFormer model against the hand-drawn sidewalk labels |
| `eval_color_difference.py`, `eval_targets.py` | Tuning and evaluation of the colour check against hand-outlined hazards |
| `eval_segformer_input.py`, `eval_segformer_pipeline.py`, `render_segformer_compare.py` | Tests of SegFormer preprocessing, input size and post-processing, and their effect on the full model |
| `Test Results/` | Write-ups of each round of testing: what was tried, the results and what was adopted |
| `*_RETIRED.py` | Earlier GMM colour check, kept only to reproduce past results |

## Quick start
python3 simulate_sidewalk.py --walks A                             # score the full model on the dataset (Dataset is avaliable here https://zenodo.org/records/19937098)

