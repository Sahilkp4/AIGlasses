"""Sweep the GMM colour check over window and threshold on walk A, scored against hand-outlined targets.

SegFormer runs once per frame; only the colour check is repeated. Scores:
  catch        share of a target's pixels left in the focus area that were flagged (targets in gmm_targets.json)
  false flags  share of labelled sidewalk in the focus area that was flagged, on every CLEAN_STEP-th frame away from
               the cone and rack sequences, whose labels call those objects sidewalk
  objects/min  flagged objects on those same frames, at the camera rate

  python3 eval_gmm.py
  python3 eval_gmm.py --windows 9 --thresholds 4 --per-frame
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

from eval_sidewalk import EVAL_H, EVAL_W, gt_mask
from gmm_color import GmmColor, ab_features
from sidewalk_system import SidewalkNavigator
from simulate_sidewalk import FPS, N_ORIGINAL, SRC

TARGETS = "gmm_targets.json"
OUT = "runs/gmm_eval"
CLEAN_STEP = 5
NOT_CLEAN = (range(125, 141), range(440, 481))  # the labels call the cone and racks sidewalk here
RACK_MIN_B = 150  # OpenCV 8-bit Lab: rack outlines are loose, so only clearly yellow pixels count as rack


def load(i):
    return cv2.resize(cv2.imread(f"{SRC}/{i:06d}.jpg", cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))


def fill(polygons):
    mask = np.zeros((EVAL_H, EVAL_W), np.uint8)
    cv2.fillPoly(mask, [np.array(p, np.int32) for p in polygons], 1)
    return mask.astype(bool)


def target_masks(targets, i, frame):
    """Masks of the targets outlined on frame i. The cone stands in the pothole, so it is cut out of it."""
    out = {name: fill(frames[str(i)]) for name, frames in targets.items() if str(i) in frames}
    if "rack" in out:
        out["rack"] &= ab_features(frame)[:, :, 1] >= RACK_MIN_B
    if "pothole" in out and "cone" in out:
        out["pothole"] &= ~out["cone"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 9, 15])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--per-frame", action="store_true", help="also print the catch rate on each target frame")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    targets = {k: v for k, v in json.load(open(TARGETS)).items() if not k.startswith("_")}

    nav = SidewalkNavigator(args.device)
    target_ids = sorted({int(i) for frames in targets.values() for i in frames})
    clean_ids = [i for i in range(1, N_ORIGINAL + 1, CLEAN_STEP) if not any(i in r for r in NOT_CLEAN)]
    data = {}
    for i in sorted(set(target_ids + clean_ids)):
        frame = load(i)
        r = nav.process(frame)
        data[i] = (frame, r["sidewalk"], r["focus"])
    gts = {i: gt_mask(f"{SRC}/{i:06d}.json") for i in clean_ids}
    print(f"{len(target_ids)} target frames, {len(clean_ids)} clean frames")

    rows = []
    for window in args.windows:
        for threshold in args.thresholds:
            gmm = GmmColor(window, threshold)
            hit, left, ms, detail = {k: 0 for k in targets}, {k: 0 for k in targets}, [], []
            for i in target_ids:
                frame, sidewalk, focus = data[i]
                flagged = gmm.verify(frame, sidewalk, focus)["color_anomaly"]
                for name, mask in target_masks(targets, i, frame).items():
                    in_focus = mask & focus
                    hit[name] += int((in_focus & flagged).sum())
                    left[name] += int(in_focus.sum())
                    detail.append(f"{name} {i}: {(in_focus & flagged).sum()}/{in_focus.sum()} of {mask.sum()}")
            false_px = clean_px = objects = other_px = 0
            for i in clean_ids:
                frame, sidewalk, focus = data[i]
                r = gmm.verify(frame, sidewalk, focus)
                ms.append(r["gmm_ms"])
                labelled = focus & gts[i]
                false_px += int((r["color_anomaly"] & labelled).sum())
                other_px += int((r["color_anomaly"] & ~gts[i]).sum())
                clean_px += int(labelled.sum())
                objects += len(r["color_objects"])
            row = {"window": window, "threshold": threshold}
            row.update({k: round(100 * hit[k] / max(left[k], 1), 1) for k in targets})
            row.update({"false_flags": round(100 * false_px / max(clean_px, 1), 2),
                        "flags_off_label": round(100 * other_px / max(other_px + false_px, 1), 1),
                        "objects_per_min": round(60 * FPS * objects / len(clean_ids), 1),
                        "ms_mean": round(float(np.mean(ms)), 1)})
            rows.append(row)
            print(row, flush=True)
            if args.per_frame:
                print("\n".join(f"  {d}" for d in detail))

    with open(f"{OUT}/sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT}/sweep.csv")


if __name__ == "__main__":
    main()
