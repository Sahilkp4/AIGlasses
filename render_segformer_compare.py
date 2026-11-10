"""Side-by-side overlays of the full model (SegFormer, focus area, Sobel, colour difference) with three front ends.

  1  current          sidewalk_system.py before the margin change: 512x512 input, argmax at 1/4 resolution, nearest upsample
  2  margin, 512x512  the same, with the margin post-processing (sidewalk_system.py as it is now)
  3  margin, 672x384  margin post-processing and a 672x384 input

Nothing else differs: the same model weights, device and fp32; the same OpenCV linear resize, RGB swap and
normalisation; the same Sobel and colour difference defaults; the same frames (walk A at 960x540, decoded as
simulate_sidewalk.py does). Each frame is scored against its own labels, as simulate_sidewalk.py's processed_no_lag; panel
1 is checked frame by frame against the most recent full run (runs/sidewalk_final_color/per_frame.csv).

The overlay classes, colours and legend are simulate_sidewalk.py's. Writes a video and stills of the target frames.

  python3 render_segformer_compare.py
  python3 render_segformer_compare.py --limit 60
"""
import argparse
import csv
import os

import cv2
import numpy as np

import eval_segformer_pipeline as pipeline
from color_difference import ColorDifference
from eval_sidewalk import EVAL_H, EVAL_W, gt_mask
from simulate_sidewalk import (BOTH_COLOR, COLOR_COLOR, FOCUS_COLOR, HOLE_BOTH_COLOR, HOLE_COLOR, HOLE_COLOR_COLOR,
                               HOLE_SOBEL_COLOR, LABEL_COLOR, OTHER_SIDEWALK_COLOR, SOBEL_COLOR, SRC, legend,
                               overlay_classes, text)
from sobel_texture import SobelTexture

OUT = "runs/segformer_compare"
LAST_RUN = "runs/sidewalk_final_color/per_frame.csv"
PANELS = [("1  current: 512x512, argmax + nearest", "current", None),
          ("2  margin, 512x512", "new", (512, 512)),
          ("3  margin, 672x384", "new", (672, 384))]
STILLS = (21, 129, 131, 133, 135, 445, 451, 463)  # the frames with hand-outlined targets
HEADER_H = 50
ITEMS = [("focus area, kept", FOCUS_COLOR), ("SegFormer sidewalk outside focus", OTHER_SIDEWALK_COLOR),
         ("hole in focus (not sidewalk)", HOLE_COLOR), ("Sobel", SOBEL_COLOR), ("colour", COLOR_COLOR),
         ("Sobel + colour", BOTH_COLOR), ("hole + Sobel", HOLE_SOBEL_COLOR), ("hole + colour", HOLE_COLOR_COLOR),
         ("hole + Sobel + colour", HOLE_BOTH_COLOR), ("labelled sidewalk (outline)", LABEL_COLOR)]


def panel(frame, r, gt, title):
    """One frame's overlay, drawn as simulate_sidewalk.render draws it, with a header of its own."""
    classes = overlay_classes(r["sidewalk"], r["focus"], r["anomaly"], r["color_anomaly"])
    blend = frame.copy()
    for mask, c, blended in classes:
        if blended:
            blend[mask] = c
    out = cv2.addWeighted(frame, 0.55, blend, 0.45, 0)
    for mask, c, blended in classes:
        if not blended:
            out[mask] = c
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, LABEL_COLOR, 1, cv2.LINE_AA)
    ref = r["refined_focus"]
    tp, fp, fn = (ref & gt).sum(), (ref & ~gt).sum(), (~ref & gt).sum()
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    t = r["timing_ms"]
    head = np.zeros((HEADER_H, EVAL_W, 3), np.uint8)
    text(head, title, (10, 20), 0.6)
    text(head, f"refined IoU {iou:.3f}   focus {100 * r['focus'].mean():.1f}%   Sobel {int(r['anomaly'].sum())} px   "
               f"colour {int(r['color_anomaly'].sum())} px, {len(r['color_objects'])} obj   "
               f"segment {t['preprocess_ms'] + t['inference_ms'] + t['postprocess_ms']:.0f} ms", (10, 42), 0.5,
         (0, 255, 255))
    return np.vstack([head, out]), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    nav = pipeline.FrontEnd(args.device, SobelTexture(), ColorDifference())
    last = {int(r["frame"]): float(r["proc_iou"]) for r in csv.DictReader(open(LAST_RUN)) if r["walk"] == "A"
            and r["proc_iou"] != ""}
    width = 3 * EVAL_W
    foot_h = legend(np.zeros((1, width, 3), np.uint8), ITEMS, 22, draw=False) + 10
    writer = None
    mismatch = []
    for i in range(1, args.limit + 1):
        path = f"{SRC}/{i:06d}.jpg"
        frame = cv2.resize(cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))
        gt = gt_mask(path[:-4] + ".json")
        tiles = []
        for k, (title, mode, size) in enumerate(PANELS):
            nav.mode = mode
            if size:
                pipeline.NEW_IN = size
            tile, iou = panel(frame, nav.process(frame), gt, title)
            tiles.append(tile)
            if k == 0 and i in last and not np.isnan(iou) and abs(iou - last[i]) > 1e-4:
                mismatch.append((i, round(iou, 4), last[i]))
        foot = np.zeros((foot_h, width, 3), np.uint8)
        text(foot, f"walk A  frame {i}", (width - 160, 22), 0.5)
        legend(foot, ITEMS, 22)
        img = np.vstack([np.hstack(tiles), foot])
        if writer is None:
            writer = cv2.VideoWriter(f"{OUT}/overlay_walkA.mp4", cv2.VideoWriter_fourcc(*"avc1"), 2,
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
        if i in STILLS:
            cv2.imwrite(f"{OUT}/frame_{i:06d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i % 100 == 0:
            print(f"{i}/{args.limit}", flush=True)
    writer.release()
    print(f"panel 1 vs last full run: {len(mismatch)} frames differ in refined IoU by > 1e-4 {mismatch[:10]}")
    print(f"wrote {OUT}/overlay_walkA.mp4 and stills of frames {STILLS}")


if __name__ == "__main__":
    main()
