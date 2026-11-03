"""Evaluate tobiasc/segformer-b0-finetuned-segments-sidewalk on dataset_flip sidewalk labels.

Ground truth: union of 'sidewalk' polygons, minus any object polygons (cars, poles, people...)
drawn over them. Scored per pixel at EVAL_W x EVAL_H. Also writes an overlay video:
green = correct sidewalk, red = predicted but not labelled, blue = labelled but missed.
"""
import csv
import glob
import json
import os
import re
from collections import Counter

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

SRC = "dataset_flip"
OUT = "runs/sidewalk_eval"
MODEL = "tobiasc/segformer-b0-finetuned-segments-sidewalk"
EVAL_W, EVAL_H = 960, 540
BATCH = 8
FPS = 30

# Objects that occlude the ground; carve them out of the sidewalk GT
OCCLUDERS = {"car", "car_group", "truck", "bus", "person", "rider", "bicycle", "bicycle_group",
             "motorcycle", "pole", "pol", "traffic sign", "traffic light"}


def gt_mask(json_path):
    d = json.load(open(json_path))
    sx, sy = EVAL_W / d["imageWidth"], EVAL_H / d["imageHeight"]
    side = np.zeros((EVAL_H, EVAL_W), np.uint8)
    occ = np.zeros_like(side)
    for s in d["shapes"]:
        base = re.sub(r"(_\d+|\d+)$", "", s["label"])
        pts = np.round(np.array(s["points"]) * [sx, sy]).astype(np.int32)
        if base == "sidewalk":
            cv2.fillPoly(side, [pts], 1)
        elif base in OCCLUDERS:
            cv2.fillPoly(occ, [pts], 1)
    return (side & (1 - occ)).astype(bool)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device).eval()
    labels = model.config.id2label
    SIDEWALK = next(k for k, v in labels.items() if v == "flat-sidewalk")

    frames = sorted(glob.glob(f"{SRC}/*.jpg"))
    writer = cv2.VideoWriter(f"{OUT}/sidewalk_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (EVAL_W, EVAL_H))
    TP = FP = FN = 0
    missed_as = Counter()  # what the model called the GT sidewalk pixels it missed
    rows = []

    for b in range(0, len(frames), BATCH):
        paths = frames[b:b + BATCH]
        imgs = [cv2.resize(cv2.imread(p), (EVAL_W, EVAL_H), interpolation=cv2.INTER_AREA) for p in paths]
        inputs = processor(images=[i[:, :, ::-1] for i in imgs], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.nn.functional.interpolate(logits, size=(EVAL_H, EVAL_W), mode="bilinear",
                                                align_corners=False).argmax(1).cpu().numpy()

        for path, img, cls_map in zip(paths, imgs, preds):
            pred = cls_map == SIDEWALK
            gt = gt_mask(path[:-4] + ".json")
            tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            missed_as.update(dict(zip(*np.unique(cls_map[gt & ~pred], return_counts=True))))
            rows.append({"frame": os.path.basename(path), "gt_px": tp + fn, "pred_px": tp + fp,
                         "tp": tp, "fp": fp, "fn": fn,
                         "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else ""})

            ov = img.copy()
            ov[pred & gt] = (0, 200, 0)
            ov[pred & ~gt] = (0, 0, 230)
            ov[~pred & gt] = (230, 120, 0)
            vis = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
            iou_txt = f"IoU {rows[-1]['iou']:.2f}" if rows[-1]["iou"] != "" else "no sidewalk"
            cv2.putText(vis, f"{rows[-1]['frame']}  {iou_txt}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            writer.write(vis)
        print(f"{min(b + BATCH, len(frames))}/{len(frames)}", flush=True)

    writer.release()
    with open(f"{OUT}/per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    prec, rec = TP / (TP + FP), TP / (TP + FN)
    with_gt = [r for r in rows if r["gt_px"] > 0]
    ious = np.array([r["iou"] for r in with_gt])
    no_gt = [r for r in rows if r["gt_px"] == 0]
    print(f"\nFrames: {len(rows)}  (with labelled sidewalk: {len(with_gt)}, without: {len(no_gt)})")
    print(f"Sidewalk pixel IoU (dataset): {TP / (TP + FP + FN):.3f}")
    print(f"Precision: {prec:.3f}   Recall: {rec:.3f}   F1/Dice: {2 * prec * rec / (prec + rec):.3f}")
    print(f"Per-frame IoU over frames with sidewalk: mean {ious.mean():.3f}, median {np.median(ious):.3f}, "
          f"p10 {np.percentile(ious, 10):.3f}, p90 {np.percentile(ious, 90):.3f}")
    if no_gt:
        print(f"Frames without sidewalk where model still predicts >1% sidewalk: "
              f"{sum(r['pred_px'] > 0.01 * EVAL_W * EVAL_H for r in no_gt)}/{len(no_gt)}")
    tot = sum(missed_as.values())
    print("\nMissed sidewalk pixels were predicted as:")
    for k, v in missed_as.most_common(6):
        print(f"  {labels[int(k)]:22s} {v / tot:6.1%}")
    print(f"\nVideo: {OUT}/sidewalk_overlay.mp4\nCSV:   {OUT}/per_frame.csv")


if __name__ == "__main__":
    main()
