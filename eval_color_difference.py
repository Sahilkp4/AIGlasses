"""Tune and check the colour difference check (color_difference.py) on walk A.

SegFormer runs once per frame and the masks are cached; only the colour check is repeated, Sobel off. Scores:
  iou, precision, recall  refined sidewalk (sidewalk minus colour flags) against the labels, pooled over all frames
  cone, rack, pothole,    share of a target's pixels inside the checked area (focus plus its holes) that were flagged
  cover                   (gmm_targets.json); *_seen is the share of the target inside the checked area at all
  catch                   mean of cone and rack, the two colour targets
  false_flags             share of labelled sidewalk in the focus area that was flagged, on clean frames (outside
                          NOT_CLEAN, whose labels call the cone and racks sidewalk)
  flags_off_label         share of flags on clean frames that fall off labelled sidewalk
  objects_per_min         flagged objects on clean frames, at the camera rate
  ms_mean, ms_p95         stage time per frame, single thread, median of --repeats runs per frame

Two settings are reported: the highest pooled IoU (the selection rule; the defaults in color_difference.py won it to
within 0.0001), and the highest catch within --max-false-flags (default 0.43%, the retired GMM's rate on all 600 frames;
ties: fewer objects/min). With a small EPS, IoU falls with every flag, as the labels call almost everything in focus
sidewalk, so the IoU rule is only meaningful once EPS is swept too.

  python3 eval_color_difference.py                  # score the current defaults on all 600 frames
  python3 eval_color_difference.py --windows 3 5 --eps 3 4 5 --min-object 25 50 --thresholds 3 3.5 4 4.5 5   # a sweep
  python3 eval_color_difference.py --frames sample --windows 5 --eps 1 --min-object 100 --thresholds 7
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import color_difference
from color_difference import ColorDifference
from eval_sidewalk import gt_mask
from eval_targets import NOT_CLEAN, TARGETS, load, target_masks
from sidewalk_system import SidewalkNavigator
from simulate_sidewalk import FPS, N_ORIGINAL, SRC
from sobel_texture import fill_holes

OUT = "runs/color_difference_eval"
FRAME_SETS = {"sample": (range(1, 31), range(122, 137), range(220, 261)), "all": (range(1, N_ORIGINAL + 1),)}
COLOR_TARGETS = ("cone", "rack")


def load_data(name, device):
    """frame, SegFormer sidewalk mask, focus mask, checked area and label mask per frame; SegFormer masks cached."""
    ids = sorted({i for r in FRAME_SETS[name] for i in r})
    cache = f"{OUT}/masks_{name}.npz"
    frames = {i: load(i) for i in ids}
    shape = next(iter(frames.values())).shape[:2]
    if os.path.exists(cache) and list(np.load(cache)["ids"]) == ids:
        z = np.load(cache)
        unpack = lambda p: np.unpackbits(p, count=shape[0] * shape[1]).reshape(shape).astype(bool)
        sidewalk = {i: unpack(p) for i, p in zip(ids, z["sidewalk"])}
        focus = {i: unpack(p) for i, p in zip(ids, z["focus"])}
    else:
        nav = SidewalkNavigator(device)
        sidewalk, focus = {}, {}
        for i in ids:
            r = nav.process(frames[i])
            sidewalk[i], focus[i] = r["sidewalk"], r["focus"]
        np.savez_compressed(cache, ids=ids, sidewalk=[np.packbits(sidewalk[i]) for i in ids],
                            focus=[np.packbits(focus[i]) for i in ids])
    return {i: (frames[i], sidewalk[i], focus[i], fill_holes(focus[i]), gt_mask(f"{SRC}/{i:06d}.json")) for i in ids}


def evaluate(name, window, threshold, check, data, targets, clean, repeats):
    """Runs check on every frame and scores it as it goes."""
    tp = fp = fn = false_px = clean_px = off_px = objects = 0
    hit, inside, total = ({k: 0 for k in targets} for _ in range(3))
    ms = []
    for i, (frame, sidewalk, focus, area, gt) in data.items():
        if check is None:
            flagged, n_obj = np.zeros_like(focus), 0
        else:
            runs = [check.verify(frame, sidewalk, focus) for _ in range(repeats)]
            ms.append(float(np.median([r["color_ms"] for r in runs])))
            flagged, n_obj = runs[0]["color_anomaly"], len(runs[0]["color_objects"])
        refined = sidewalk & ~flagged
        tp += int((refined & gt).sum())
        fp += int((refined & ~gt).sum())
        fn += int((~refined & gt).sum())
        for k, mask in target_masks(targets, i, frame).items():
            hit[k] += int((mask & area & flagged).sum())
            inside[k] += int((mask & area).sum())
            total[k] += int(mask.sum())
        if i in clean:
            false_px += int((flagged & focus & gt).sum())
            off_px += int((flagged & ~gt).sum())
            clean_px += int((focus & gt).sum())
            objects += n_obj
    row = {"method": name, "window": window, "threshold": threshold,
           "eps": getattr(check, "eps", "-"), "min_object_px": getattr(check, "min_object_px", "-"),
           "iou": round(tp / max(tp + fp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
           "recall": round(tp / max(tp + fn, 1), 4)}
    row.update({k: round(100 * hit[k] / max(inside[k], 1), 1) for k in targets})
    row["catch"] = round(float(np.mean([row[k] for k in COLOR_TARGETS if k in row])), 1)
    row.update({f"{k}_seen": round(100 * inside[k] / max(total[k], 1), 1) for k in targets})
    row.update({"false_flags": round(100 * false_px / max(clean_px, 1), 3),
                "flags_off_label": round(100 * off_px / max(off_px + false_px, 1), 1),
                "objects_per_min": round(60 * FPS * objects / max(len(clean), 1), 1),
                "ms_mean": round(float(np.mean(ms)), 2) if ms else 0.0,
                "ms_p95": round(float(np.percentile(ms, 95)), 2) if ms else 0.0})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--frames", choices=sorted(FRAME_SETS), default="all")
    ap.add_argument("--windows", type=int, nargs="+", default=[color_difference.WINDOW])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[color_difference.THRESHOLD])
    ap.add_argument("--eps", type=float, nargs="+", default=[color_difference.EPS])
    ap.add_argument("--min-object", type=int, nargs="+", default=[color_difference.MIN_OBJECT_PX])
    ap.add_argument("--max-false-flags", type=float, default=0.43,
                    help="false-flag budget in %% for the catch rule (0.43: the retired GMM on all 600 frames)")
    ap.add_argument("--repeats", type=int, default=1, help="timing runs per frame; the median is kept")
    ap.add_argument("--tag", default="", help="suffix for the output csv")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cv2.setNumThreads(1)
    targets = {k: v for k, v in json.load(open(TARGETS)).items() if not k.startswith("_")}
    data = load_data(args.frames, args.device)
    clean = {i for i in data if not any(i in r for r in NOT_CLEAN)}
    print(f"{len(data)} frames, {len(clean)} clean", flush=True)

    rows = [evaluate("none", "-", "-", None, data, targets, clean, 1)]
    print(rows[-1], flush=True)
    budget = args.max_false_flags
    for w in args.windows:
        for e in args.eps:
            for m in args.min_object:
                for t in args.thresholds:
                    rows.append(evaluate("color_diff", w, t, ColorDifference(w, t, e, m), data, targets, clean,
                                         args.repeats))
                    print(rows[-1], flush=True)

    path = f"{OUT}/sweep_{args.frames}{args.tag}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    diff = [r for r in rows if r["method"] == "color_diff"]
    print(f"\nBest colour difference by IoU: {max(diff, key=lambda r: r['iou'])}")
    ok = [r for r in diff if r["false_flags"] <= budget]
    if ok:
        best = max(ok, key=lambda r: (r["catch"], -r["objects_per_min"]))
        print(f"Best colour difference within {budget}% false flags: {best}")
    else:
        print(f"No colour difference setting stays within {budget}% false flags")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
