"""Run YOLO11n over dataset_flip frames as a continuous video (with tracking),
write an annotated video, and score detections against the LabelMe ground truth."""
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

SRC = "dataset_flip"
OUT = "runs/dataset_flip"
FPS = 30
OUT_WIDTH = 1920  # source frames are 4K; downscale the output video
IOU_THR = 0.5

# GT label (instance suffix stripped) -> COCO class name used by YOLO
GT_TO_COCO = {
    "car": "car", "truck": "truck", "bus": "bus", "person": "person",
    "rider": "person", "bicycle": "bicycle", "motorcycle": "motorcycle",
    "traffic light": "traffic light",
}


def gt_boxes(json_path):
    boxes = []
    for s in json.load(open(json_path))["shapes"]:
        base = re.sub(r"[_\d]+$", "", s["label"])
        if base in GT_TO_COCO:
            p = np.array(s["points"])
            boxes.append((GT_TO_COCO[base], *p.min(0), *p.max(0)))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = sorted(glob.glob(f"{SRC}/*.jpg"))
    model = YOLO("yolo11n.pt")
    eval_classes = set(GT_TO_COCO.values())

    writer = None
    tp, fp, fn = Counter(), Counter(), Counter()
    track_ids = defaultdict(set)
    rows = []

    for i, path in enumerate(frames):
        r = model.track(path, persist=True, verbose=False)[0]
        plot = r.plot()
        h, w = plot.shape[:2]
        plot = cv2.resize(plot, (OUT_WIDTH, int(h * OUT_WIDTH / w)))
        cv2.putText(plot, f"{os.path.basename(path)}  ({i + 1}/{len(frames)})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        if writer is None:
            writer = cv2.VideoWriter(f"{OUT}/yolo11n_tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                     FPS, (plot.shape[1], plot.shape[0]))
        writer.write(plot)

        names = [model.names[int(c)] for c in r.boxes.cls]
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(names)
        for n, t in zip(names, ids):
            if t is not None:
                track_ids[n].add(t)

        # Greedy IoU matching per class against GT
        preds = [(n, *b) for n, b in zip(names, r.boxes.xyxy.tolist()) if n in eval_classes]
        gts = gt_boxes(path[:-4] + ".json")
        for cls in eval_classes:
            P = [p[1:] for p in preds if p[0] == cls]
            G = [g[1:] for g in gts if g[0] == cls]
            used = set()
            for p in P:
                best, bj = 0, -1
                for j, g in enumerate(G):
                    if j not in used and (v := iou(p, g)) > best:
                        best, bj = v, j
                if best >= IOU_THR:
                    used.add(bj); tp[cls] += 1
                else:
                    fp[cls] += 1
            fn[cls] += len(G) - len(used)

        c = Counter(names)
        rows.append({"frame": os.path.basename(path), "n_detections": len(names),
                     "speed_ms": round(sum(r.speed.values()), 1), **c})
        if i % 100 == 0:
            print(f"{i + 1}/{len(frames)}  {dict(c)}")

    writer.release()

    keys = ["frame", "n_detections", "speed_ms"] + sorted({k for r in rows for k in r} - {"frame", "n_detections", "speed_ms"})
    with open(f"{OUT}/per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=0)
        w.writeheader(); w.writerows(rows)

    print(f"\nFrames: {len(rows)}   avg detections/frame: {np.mean([r['n_detections'] for r in rows]):.1f}"
          f"   avg inference: {np.mean([r['speed_ms'] for r in rows]):.1f} ms/frame")
    print("\nUnique tracked objects:", {k: len(v) for k, v in sorted(track_ids.items(), key=lambda x: -len(x[1]))})
    print(f"\nVs ground truth (box IoU >= {IOU_THR}):")
    print(f"{'class':15s}{'GT':>7s}{'TP':>7s}{'FP':>7s}{'FN':>7s}{'prec':>7s}{'recall':>7s}")
    for cls in sorted(eval_classes, key=lambda c: -(tp[c] + fn[c])):
        g = tp[cls] + fn[cls]
        prec = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] else 0
        rec = tp[cls] / g if g else 0
        print(f"{cls:15s}{g:7d}{tp[cls]:7d}{fp[cls]:7d}{fn[cls]:7d}{prec:7.2f}{rec:7.2f}")
    print(f"\nVideo: {OUT}/yolo11n_tracked.mp4\nCSV:   {OUT}/per_frame.csv")


if __name__ == "__main__":
    main()
