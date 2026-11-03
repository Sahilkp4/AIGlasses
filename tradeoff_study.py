"""Speed/accuracy trade-off study for running SegFormer (sidewalk) + YOLO11n together in real time.

Settings compared (all on the CPU, a stand-in for a phone / edge device):
  SegFormer input size 512 / 384 / 256, post-processing full (as in eval_sidewalk.py) or fast
  YOLO input size 640 / 480 / 320
  SegFormer on every update, or only every 2nd / 3rd update (YOLO runs on every update)

Stage `infer`: run every setting on every dataset_flip frame, saving predictions and their timings.
  -> accuracy against the labels (no delay) and per-model latency.
Stage `freshness` (optional, needs a 30 FPS street walking video): run every setting on it and measure how well a prediction made
  k frames ago still agrees with the current frame's prediction. dataset_flip frames are seconds apart, so the
  cost of a delay can only be measured on real 30 FPS footage. The video is unlabelled, so this is agreement
  with the model itself, not accuracy.
Stage `analyze`: simulate a single worker on a 30 FPS camera at 30 / 10 / 5 updates per second (frames are
  dropped while it is busy). If freshness.json exists, live accuracy is estimated as no-delay accuracy x
  agreement at the result's age; otherwise the no-delay accuracy is reported and delay is not accounted for.

  python3 tradeoff_study.py infer       # ~5 min
  python3 tradeoff_study.py freshness --video walk.mov   # optional, ~3 min
  python3 tradeoff_study.py analyze     # seconds, re-run freely
"""
import argparse
import csv
import glob
import itertools
import json
import os
import pickle
import re
import time

import cv2
import numpy as np
import torch

from eval_sidewalk import OCCLUDERS
from run_dataset import GT_TO_COCO

SRC = "dataset_flip"
OUT = "runs/tradeoff"
SEG_MODEL = "tobiasc/segformer-b0-finetuned-segments-sidewalk"
YOLO_MODEL = "yolo11n.pt"
W, H = 960, 540          # frame size both models receive
EW, EH = 480, 270        # resolution sidewalk/road masks are scored at
SEG_SIZES = [512, 384, 256]
YOLO_SIZES = [640, 480, 320]
POSTS = ["full", "fast"]
SEG_EVERY = [1, 2, 3]    # run SegFormer on every m-th update (YOLO runs on every update)
RATES = [30, 10, 5]      # target updates per second
VIDEO_FPS = 30
LAGS = [1, 2, 3, 4, 6, 9, 12, 18]  # delays (in 30 FPS frames) measured by the freshness stage
CONF = 0.25
IOU_THR = 0.5
NEAR_FRAC = 0.01         # boxes covering >= 1% of the frame count as "nearby" objects
IGNORE = {"car_group", "bicycle_group"}  # crowd regions: detections inside them are not false positives
WARMUP = 10


def base_label(label):
    return re.sub(r"(_\d+|\d+)$", "", label)


def load_models(args):
    """Returns run(img) -> ({setting: sidewalk/road masks}, {setting: boxes}, {setting: ms})."""
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
    seg = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).eval()
    labels = seg.config.id2label
    if args.compile:  # eager PyTorch runs SegFormer's depthwise convs as thousands of tiny convs on Mac CPUs;
        seg = torch.compile(seg, dynamic=True)  # dynamic=True also gets a much faster conv path (same outputs)
    SIDEWALK = next(k for k, v in labels.items() if v == "flat-sidewalk")
    ROAD = [k for k, v in labels.items() if v in ("flat-road", "flat-crosswalk")]
    yolo = YOLO(YOLO_MODEL)

    def seg_once(rgb, s):
        t0 = time.perf_counter()
        inputs = processor(images=rgb, size={"height": s, "width": s}, return_tensors="pt")
        with torch.no_grad():
            logits = seg(**inputs).logits
        t1 = time.perf_counter()
        full = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear",
                                               align_corners=False).argmax(1)[0].numpy().astype(np.uint8)
        t2 = time.perf_counter()
        fast = cv2.resize(logits.argmax(1)[0].numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        t3 = time.perf_counter()
        return {"full": (full, (t2 - t0) * 1000), "fast": (fast, (t1 - t0 + t3 - t2) * 1000)}

    def yolo_once(img, s):
        t0 = time.perf_counter()
        r = yolo.predict(img, imgsz=s, conf=CONF, device="cpu", verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        return [(yolo.names[int(c)], float(cf), *b) for c, cf, b in
                zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist())], ms

    def run(img):
        masks, boxes, ms = {}, {}, {}
        for s in SEG_SIZES:
            for post, (cm, t) in seg_once(img[:, :, ::-1], s).items():
                k = f"seg{s}_{post}"
                small = cm[::H // EH, ::W // EW]
                masks[k] = (np.packbits(small == SIDEWALK), np.packbits(np.isin(small, ROAD)))
                ms[k] = t
        for s in YOLO_SIZES:
            boxes[f"yolo{s}"], ms[f"yolo{s}"] = yolo_once(img, s)
        return masks, boxes, ms

    return run


def collect(run, images, n):
    """Runs every setting on n images (an iterator of 960x540 BGR frames) after a warm-up."""
    masks, boxes, T = {}, {}, {}
    for i, img in enumerate(images):
        if i < WARMUP:
            run(img)
            continue
        m, b, ms = run(img)
        for k, v in m.items():
            masks.setdefault(k, []).append(v)
        for k, v in b.items():
            boxes.setdefault(k, []).append(v)
        for k, v in ms.items():
            T.setdefault(k, []).append(v)
        if (i - WARMUP) % 100 == 0:
            print(f"{i - WARMUP + 1}/{n}  " + "  ".join(f"{k} {v:.0f}ms" for k, v in ms.items()), flush=True)
    return masks, boxes, {k: np.array(v) for k, v in T.items()}


def resized(img):
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- stage: infer (dataset accuracy + timing)
def infer(args):
    run = load_models(args)
    frames = sorted(glob.glob(f"{SRC}/*.jpg"))[:args.limit or None]
    images = (resized(cv2.imread(p)) for p in frames[:WARMUP] + frames)
    masks, boxes, T = collect(run, images, len(frames))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/predictions.pkl", "wb") as f:
        pickle.dump({"frames": frames, "times": T, "masks": masks, "boxes": boxes,
                     "threads": args.threads, "compiled": args.compile}, f)
    print(f"Saved {OUT}/predictions.pkl")


# ---------------------------------------------------------------- scoring helpers
POP = np.array([bin(i).count("1") for i in range(256)], np.int64)


def popcount(a):
    return int(POP[a].sum())


def ground_truth(json_path):
    d = json.load(open(json_path))
    sx, sy = EW / d["imageWidth"], EH / d["imageHeight"]
    bx, by = W / d["imageWidth"], H / d["imageHeight"]
    side, road, occ = (np.zeros((EH, EW), np.uint8) for _ in range(3))
    objs, ignore = [], []
    for s in d["shapes"]:
        base = base_label(s["label"])
        pts = np.array(s["points"])
        poly = np.round(pts * [sx, sy]).astype(np.int32)
        if base == "sidewalk":
            cv2.fillPoly(side, [poly], 1)
        elif base == "road":
            cv2.fillPoly(road, [poly], 1)
        elif base in OCCLUDERS or base in IGNORE:
            cv2.fillPoly(occ, [poly], 1)
        box = (*(pts.min(0) * [bx, by]), *(pts.max(0) * [bx, by]))
        if base in GT_TO_COCO:
            objs.append((GT_TO_COCO[base], *box))
        elif base in IGNORE:
            ignore.append(box)
    keep = 1 - occ
    return (np.packbits((side & keep).astype(bool)), np.packbits((road & keep).astype(bool)),
            objs, ignore)


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(preds, gt_objs, ignore=()):
    """Greedy per-class matching. Returns tp, fp, fn, near_gt, near_tp."""
    eval_classes = set(GT_TO_COCO.values())
    preds = [p for p in preds if p[0] in eval_classes]
    if ignore and preds:  # drop detections whose centre lies inside a crowd region
        ig = np.array(ignore)
        c = np.array([((p[2] + p[4]) / 2, (p[3] + p[5]) / 2) for p in preds])
        inside = ((c[:, None, 0] >= ig[None, :, 0]) & (c[:, None, 0] <= ig[None, :, 2]) &
                  (c[:, None, 1] >= ig[None, :, 1]) & (c[:, None, 1] <= ig[None, :, 3])).any(1)
        preds = [p for p, x in zip(preds, inside) if not x]
    tp = fp = fn = near_gt = near_tp = 0
    for cls in eval_classes:
        P = sorted([p for p in preds if p[0] == cls], key=lambda p: -p[1])
        G = [g[1:] for g in gt_objs if g[0] == cls]
        near = [(g[2] - g[0]) * (g[3] - g[1]) >= NEAR_FRAC * W * H for g in G]
        used = set()
        if P and G:
            M = iou_matrix([p[2:] for p in P], G)
            for r in range(len(P)):
                M[r, list(used)] = 0
                j = int(M[r].argmax())
                if M[r, j] >= IOU_THR:
                    used.add(j)
        tp += len(used); fp += len(P) - len(used); fn += len(G) - len(used)
        near_gt += sum(near); near_tp += sum(near[j] for j in used)
    return tp, fp, fn, near_gt, near_tp


def prf(tp, fp, fn, near_gt, near_tp):
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9), near_tp / max(near_gt, 1)


# ---------------------------------------------------------------- stage: freshness (cost of delay, 30 FPS video)
def freshness(args):
    run = load_models(args)
    cap = cv2.VideoCapture(args.video)
    video = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        video.append(resized(f))
    video = video[:args.limit or None]
    masks, boxes, _ = collect(run, video[:WARMUP] + video, len(video))
    N = len(video)
    out = {}
    for k, m in masks.items():  # sidewalk IoU between the prediction `lag` frames ago and the current one
        out[k] = {}
        for lag in LAGS:
            S = np.zeros(3)
            for t in range(lag, N):
                a, b = m[t - lag][0], m[t][0]
                S += (popcount(a & b), popcount(a & ~b), popcount(~a & b))
            out[k][lag] = {"sidewalk": S[0] / max(S.sum(), 1)}
    for k, b in boxes.items():  # object F1 / nearby recall of old detections vs current detections
        out[k] = {}
        for lag in LAGS:
            D = np.zeros(5)
            for t in range(lag, N):
                D += match(b[t - lag], [(p[0], *p[2:]) for p in b[t]])
            _, _, f1, near = prf(*D)
            out[k][lag] = {"obj_f1": f1, "near_recall": near}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/freshness.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nAgreement of a prediction made N frames earlier with the current one ({args.video}, {N} frames):")
    print(f"  {'setting':14s}" + "".join(f"{f'{l}f/{l * 1000 // VIDEO_FPS}ms':>12s}" for l in LAGS))
    for k, v in out.items():
        key = "sidewalk" if k.startswith("seg") else "obj_f1"
        print(f"  {k:14s}" + "".join(f"{v[l][key]:12.3f}" for l in LAGS))
    print(f"Saved {OUT}/freshness.json")


# ---------------------------------------------------------------- stage: analyze
def simulate(seg_ms, yolo_ms, rate, seg_every):
    """Single worker on a 30 FPS camera. Returns achieved update rate, sidewalk update rate and the mean age
    (ms since the frame was captured) of the object and sidewalk results available at each camera frame."""
    N, dt, period = len(seg_ms), 1000 / VIDEO_FPS, 1000 / rate
    jobs, free, last_start, n = [], 0.0, -1e9, 0
    while True:
        start = max(free, last_start + period)
        i = int(start // dt)
        if i >= N:
            break
        start = max(start, i * dt)
        run_seg = n % seg_every == 0
        end = start + yolo_ms[i] + (seg_ms[i] if run_seg else 0)
        jobs.append((i, end, run_seg))
        free, last_start, n = end, start, n + 1
    yolo_age, seg_age = [], []
    k, cur_seg, cur_yolo = 0, -1, -1
    for j in range(N):
        while k < len(jobs) and jobs[k][1] <= j * dt:
            cur_yolo = jobs[k][0]
            if jobs[k][2]:
                cur_seg = jobs[k][0]
            k += 1
        if cur_seg >= 0:
            yolo_age.append((j - cur_yolo) * dt)
            seg_age.append((j - cur_seg) * dt)
    dur = N * dt / 1000
    return (len(jobs) / dur, sum(r for _, _, r in jobs) / dur, float(np.mean(yolo_age)), float(np.mean(seg_age)))


def analyze(args):
    with open(f"{OUT}/predictions.pkl", "rb") as f:
        P = pickle.load(f)
    fresh = None
    if os.path.exists(f"{OUT}/freshness.json"):
        with open(f"{OUT}/freshness.json") as f:
            fresh = json.load(f)
    else:
        print("No freshness.json: accuracy below is with no delay; the cost of stale results is NOT included.")
    frames, T, masks, boxes = P["frames"], P["times"], P["masks"], P["boxes"]
    N = len(frames)
    print(f"Loaded {N} frames (inference ran with {P['threads']} CPU threads, SegFormer compiled: {P['compiled']})")
    GT = [ground_truth(p[:-4] + ".json") for p in frames]

    # no-delay accuracy of each setting against the labels
    acc = {}
    for k in T:
        if k.startswith("seg"):
            S = np.zeros(6)
            for j in range(N):
                (ps, pr), (gs, gr) = masks[k][j], GT[j][:2]
                S += (popcount(ps & gs), popcount(ps & ~gs), popcount(~ps & gs),
                      popcount(pr & gr), popcount(pr & ~gr), popcount(~pr & gr))
            acc[k] = {"sidewalk": S[0] / S[:3].sum(), "road": S[3] / S[3:].sum()}
        else:
            D = np.sum([match(boxes[k][j], GT[j][2], GT[j][3]) for j in range(N)], 0)
            p, r, f1, near = prf(*D)
            acc[k] = {"precision": p, "recall": r, "obj_f1": f1, "near_recall": near}

    def agree(k, metric, age_ms):  # agreement after a delay of age_ms, interpolated; 1.0 at zero delay
        if fresh is None:
            return 1.0
        lags = [0] + LAGS
        vals = [1.0] + [fresh[k][str(l)][metric] for l in LAGS]
        return float(np.interp(age_ms / (1000 / VIDEO_FPS), lags, vals))

    print("\nEach setting alone (dataset_flip labels, no delay):")
    print(f"  {'setting':14s} {'median ms':>9s} {'p95 ms':>7s}   accuracy")
    for k, v in T.items():
        if k.startswith("seg"):
            a = f"sidewalk IoU {acc[k]['sidewalk']:.3f}   road IoU {acc[k]['road']:.3f}"
            if fresh:
                a += f"   | sidewalk agreement @100ms {agree(k, 'sidewalk', 100):.3f}"
        else:
            a = (f"obj P {acc[k]['precision']:.3f}  R {acc[k]['recall']:.3f}  F1 {acc[k]['obj_f1']:.3f}"
                 f"   nearby recall {acc[k]['near_recall']:.3f}")
            if fresh:
                a += f"   | obj agreement @100ms {agree(k, 'obj_f1', 100):.3f}"
        print(f"  {k:14s} {np.median(v):9.1f} {np.percentile(v, 95):7.1f}   {a}")

    rows = []
    combos = list(itertools.product(SEG_SIZES, POSTS, YOLO_SIZES, SEG_EVERY))
    for rate, (ss, post, ys, m) in itertools.product(RATES, combos):
        sk, yk = f"seg{ss}_{post}", f"yolo{ys}"
        hz, sw_hz, y_age, s_age = simulate(T[sk], T[yk], rate, m)
        sw = acc[sk]["sidewalk"] * agree(sk, "sidewalk", s_age)
        f1 = acc[yk]["obj_f1"] * agree(yk, "obj_f1", y_age)
        near = acc[yk]["near_recall"] * agree(yk, "near_recall", y_age)
        rows.append({
            "target_hz": rate, "seg_size": ss, "seg_post": post, "yolo_size": ys, "seg_every": m,
            "achieved_hz": round(hz, 2), "sidewalk_hz": round(sw_hz, 2), "meets_target": hz >= 0.95 * rate,
            "obj_age_ms": round(y_age, 1), "sidewalk_age_ms": round(s_age, 1),
            "sidewalk_iou_nodelay": round(acc[sk]["sidewalk"], 4), "obj_f1_nodelay": round(acc[yk]["obj_f1"], 4),
            "near_recall_nodelay": round(acc[yk]["near_recall"], 4),
            "est_live_sidewalk_iou": round(sw, 4), "est_live_obj_f1": round(f1, 4),
            "est_live_near_recall": round(near, 4), "score": round((sw + f1) / 2, 4),
        })

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    hdr = (f"  {'seg':>4s} {'post':>4s} {'yolo':>4s} {'seg/':>4s} {'upd Hz':>7s} {'sw Hz':>6s} {'obj age':>8s}"
           f" {'sw age':>7s} {'sw IoU':>7s} {'obj F1':>7s} {'near R':>7s} {'score':>6s}")
    fmt = lambda r: (f"  {r['seg_size']:4d} {r['seg_post']:>4s} {r['yolo_size']:4d} {r['seg_every']:4d}"
                     f" {r['achieved_hz']:7.1f} {r['sidewalk_hz']:6.1f} {r['obj_age_ms']:6.0f}ms"
                     f" {r['sidewalk_age_ms']:5.0f}ms {r['est_live_sidewalk_iou']:7.3f} {r['est_live_obj_f1']:7.3f}"
                     f" {r['est_live_near_recall']:7.3f} {r['score']:6.3f}")
    for rate in RATES:
        ok = sorted([r for r in rows if r["target_hz"] == rate and r["meets_target"]], key=lambda r: -r["score"])
        print(f"\n=== {rate} updates/s: {len(ok)} of {len(combos)} settings keep up. Top 5 by score ="
              f" mean(sidewalk IoU, object F1) ===")
        if ok:
            print(hdr)
            for r in ok[:5]:
                print(fmt(r))
        else:
            best = max((r for r in rows if r["target_hz"] == rate), key=lambda r: r["achieved_hz"])
            print(f"  none; fastest setting reaches {best['achieved_hz']:.1f} updates/s")
    print(f"\nAll results: {OUT}/results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["infer", "freshness", "analyze"])
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (phone-class CPUs have ~4 fast cores)")
    ap.add_argument("--no-compile", dest="compile", action="store_false", help="run SegFormer in eager mode")
    ap.add_argument("--video", help="30 FPS street walking video for the freshness stage")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N frames")
    args = ap.parse_args()
    {"infer": infer, "freshness": freshness, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
