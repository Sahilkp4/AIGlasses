"""Test how stage 0 and stage 1 (frame shrink, colour order, input size, post-processing) affect SegFormer's sidewalk mask.

Each variant changes one design detail of the current pipeline (sidewalk_system.py) and keeps the rest:
  current       4K -> 960x540 INTER_AREA (main()) -> 512x512 INTER_LINEAR, RGB, fp32; argmax at 1/4 res, nearest upsample
  resize chain  other ways of getting from the 4K frame to 512x512 (direct, area, PIL bilinear as in the HF processor,
                and libjpeg's 1/4 DCT decode that the earlier evals and simulate_sidewalk.py fed the model)
  colour order  BGR fed without the channel swap
  input size    other model input sizes, square and 16:9
  precision     fp16 weights and inputs
Every variant is scored with three post-processing methods on the same logits:
  nearest   argmax at 1/4 resolution, nearest upsample of the class map (current)
  bilinear  bilinear upsample of all logits, then argmax (Hugging Face post_process_semantic_segmentation)
  margin    max sidewalk logit minus max other logit at 1/4 resolution, bilinear upsample of that one channel, > 0

Scores on walk A (frames 1-600) against the labels (eval_sidewalk.gt_mask, 960x540), pooled over frames:
  sidewalk IoU/P/R  the sidewalk-class mask (SIDEWALK classes); focus IoU/P/R  its largest region, which feeds the checks
  focus_vs_current  share of pixels where the focus area differs from the current pipeline's
  frames +/-        frames whose focus IoU rose / fell by more than 0.01 against current

  python3 eval_segformer_input.py                 # accuracy, all variants (about 10 min on MPS)
  python3 eval_segformer_input.py --timing        # latency per stage, MPS and CPU
"""
import argparse
import copy
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from eval_sidewalk import EVAL_H, EVAL_W, gt_mask
from sidewalk_system import MEAN, MODEL, SIDEWALK, STD, SidewalkNavigator

SRC = "data/dataset_flip"
OUT = "runs/segformer_input_eval"
N = 600
SIZE = (EVAL_H, EVAL_W)
largest = SidewalkNavigator._largest_region


def lin(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def area(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def rgb(bgr):
    return np.ascontiguousarray(bgr[:, :, ::-1])


def pil(bgr, w, h):
    return np.asarray(Image.fromarray(rgb(bgr)).resize((w, h), Image.BILINEAR))


# name -> (group, input builder from (4K frame, 960 INTER_AREA frame, 960 DCT frame), half precision)
VARIANTS = {
    "current": ("baseline", lambda F4, M, D: rgb(lin(M, 512, 512)), False),
    "960 DCT decode > 512 linear": ("resize chain", lambda F4, M, D: rgb(lin(D, 512, 512)), False),
    "960 > 512 area": ("resize chain", lambda F4, M, D: rgb(area(M, 512, 512)), False),
    "960 > 512 PIL bilinear (HF)": ("resize chain", lambda F4, M, D: pil(M, 512, 512), False),
    "4K > 512 area (no 960 step)": ("resize chain", lambda F4, M, D: rgb(area(F4, 512, 512)), False),
    "4K > 512 linear (no 960 step)": ("resize chain", lambda F4, M, D: rgb(lin(F4, 512, 512)), False),
    "BGR fed (no swap)": ("colour order", lambda F4, M, D: lin(M, 512, 512), False),
    "256x256": ("input size", lambda F4, M, D: rgb(area(M, 256, 256)), False),
    "384x384": ("input size", lambda F4, M, D: rgb(lin(M, 384, 384)), False),
    "448x448": ("input size", lambda F4, M, D: rgb(lin(M, 448, 448)), False),
    "640x640": ("input size", lambda F4, M, D: rgb(lin(M, 640, 640)), False),
    "768x768": ("input size", lambda F4, M, D: rgb(lin(M, 768, 768)), False),
    "1024x1024": ("input size", lambda F4, M, D: rgb(lin(M, 1024, 1024)), False),
    "512w x 288h (16:9)": ("input size", lambda F4, M, D: rgb(area(M, 512, 288)), False),
    "672w x 384h (16:9)": ("input size", lambda F4, M, D: rgb(lin(M, 672, 384)), False),
    "896w x 512h (16:9)": ("input size", lambda F4, M, D: rgb(lin(M, 896, 512)), False),
    "960w x 540h (native)": ("input size", lambda F4, M, D: rgb(M), False),
    "fp16": ("precision", lambda F4, M, D: rgb(lin(M, 512, 512)), True),
}


class Post:
    def __init__(self, n_classes, ids, device):
        self.ids = torch.tensor(ids, device=device)
        self.other = torch.tensor([c for c in range(n_classes) if c not in ids], device=device)

    def nearest(self, lg):
        cls = F.interpolate(lg.argmax(1, keepdim=True).float(), size=SIZE, mode="nearest")[0, 0]
        return torch.isin(cls.long(), self.ids).cpu().numpy()

    def bilinear(self, lg):
        cls = F.interpolate(lg.float(), size=SIZE, mode="bilinear", align_corners=False).argmax(1)[0]
        return torch.isin(cls, self.ids).cpu().numpy()

    def margin(self, lg):
        lg = lg.float()
        m = lg[:, self.ids].amax(1, keepdim=True) - lg[:, self.other].amax(1, keepdim=True)
        return (F.interpolate(m, size=SIZE, mode="bilinear", align_corners=False)[0, 0] > 0).cpu().numpy()


POSTS = ("nearest", "bilinear", "margin")


def to_input(img, device, half):
    x = (img.astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(device)
    return x.half() if half else x


def load_model(device):
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device).eval()
    names = {v: int(k) for k, v in model.config.id2label.items()}
    return model, [names[n] for n in SIDEWALK]


def frames(i):
    path = f"{SRC}/{i:06d}.jpg"
    full = cv2.imread(path)
    return full, area(full, EVAL_W, EVAL_H), cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)


def counts(pred, gt):
    return np.array([(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()], np.int64)


def scores(c):
    tp, fp, fn = c
    return {"iou": tp / (tp + fp + fn), "p": tp / (tp + fp), "r": tp / (tp + fn)}


def accuracy(args):
    device = args.device
    model, ids = load_model(device)
    model16 = copy.deepcopy(model).half()
    post = Post(model.config.num_labels, ids, device)
    keys = [(v, p) for v in VARIANTS for p in POSTS]
    side, focus = {k: np.zeros(3, np.int64) for k in keys}, {k: np.zeros(3, np.int64) for k in keys}
    diff = {k: 0 for k in keys}
    per_frame = []
    t0 = time.time()
    for i in range(1, args.limit + 1):
        full, mid, dct = frames(i)
        gt = gt_mask(f"{SRC}/{i:06d}.json")
        row, ref = {"frame": i}, None
        for v, (_, build, half) in VARIANTS.items():
            with torch.no_grad():
                lg = (model16 if half else model)(pixel_values=to_input(build(full, mid, dct), device, half)).logits
            for p in POSTS:
                s = getattr(post, p)(lg)
                f = largest(s)
                side[v, p] += counts(s, gt)
                c = counts(f, gt)
                focus[v, p] += c
                if ref is None:
                    ref = f  # "current", nearest comes first
                diff[v, p] += int((f != ref).sum())
                row[f"{v}|{p}"] = c[0] / max(c.sum(), 1)
        per_frame.append(row)
        if i % 50 == 0:
            print(f"{i}/{args.limit}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/per_frame_focus_iou.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_frame[0]))
        w.writeheader()
        w.writerows(per_frame)
    base = np.array([r["current|nearest"] for r in per_frame])
    results = []
    for v, p in keys:
        ious = np.array([r[f"{v}|{p}"] for r in per_frame])
        results.append({"variant": v, "group": VARIANTS[v][0], "post": p,
                        **{f"sidewalk_{k}": round(x, 4) for k, x in scores(side[v, p]).items()},
                        **{f"focus_{k}": round(x, 4) for k, x in scores(focus[v, p]).items()},
                        "focus_vs_current_pct": round(100 * diff[v, p] / (args.limit * EVAL_W * EVAL_H), 3),
                        "frames_up": int((ious - base > 0.01).sum()), "frames_down": int((base - ious > 0.01).sum())})
    with open(f"{OUT}/accuracy.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\n{'variant':32s} {'post':9s} {'side IoU':>8s} {'focus IoU':>9s} {'P':>6s} {'R':>6s} "
          f"{'diff%':>6s} {'+':>4s} {'-':>4s}")
    for r in results:
        print(f"{r['variant']:32s} {r['post']:9s} {r['sidewalk_iou']:8.4f} {r['focus_iou']:9.4f} {r['focus_p']:6.3f} "
              f"{r['focus_r']:6.3f} {r['focus_vs_current_pct']:6.2f} {r['frames_up']:4d} {r['frames_down']:4d}")


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def timing(args):
    """Median and p95 ms per stage. Frames come from a preloaded set; 4K decode and the 960 shrink are timed apart."""
    out = {}
    paths = [f"{SRC}/{i:06d}.jpg" for i in range(1, 601, 30)]
    t = []
    fulls = []
    for p in paths:
        s = time.perf_counter()
        fulls.append(cv2.imread(p))
        t.append(time.perf_counter() - s)
    out["decode_4K_ms"] = 1000 * np.median(t)
    for name, fn in {"shrink_4K_to_960_area": lambda f: area(f, 960, 540),
                     "shrink_4K_to_512_area": lambda f: area(f, 512, 512),
                     "shrink_4K_to_512_linear": lambda f: lin(f, 512, 512)}.items():
        t = []
        for f in fulls * 3:
            s = time.perf_counter()
            fn(f)
            t.append(time.perf_counter() - s)
        out[name + "_ms"] = 1000 * np.median(t)
    mids = [area(f, 960, 540) for f in fulls]
    for name, fn in {"960_to_512_linear": lambda m: lin(m, 512, 512), "960_to_512_area": lambda m: area(m, 512, 512),
                     "960_to_512_PIL_bilinear": lambda m: pil(m, 512, 512)}.items():
        t = []
        for m in mids * 5:
            s = time.perf_counter()
            fn(m)
            t.append(time.perf_counter() - s)
        out[name + "_ms"] = 1000 * np.median(t)
    print(json.dumps({k: round(v, 2) for k, v in out.items()}, indent=1))

    rows = []
    for device in args.timing_devices:
        model, ids = load_model(device)
        model16 = copy.deepcopy(model).half() if device != "cpu" else None
        post = Post(model.config.num_labels, ids, device)
        mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

        def pre_cpu(m, w, h, half):  # current: normalise in numpy, then upload float32
            return to_input(rgb(lin(m, w, h)), device, half)

        def pre_dev(m, w, h, half):  # upload uint8, swap and normalise on the device
            x = torch.from_numpy(lin(m, w, h)).to(device).permute(2, 0, 1)[None].flip(1)
            x = (x.float() / 255 - mean) / std
            return x.half() if half else x

        configs = [("512x512", 512, 512, False), ("384x384", 384, 384, False), ("448x448", 448, 448, False),
                   ("640x640", 640, 640, False), ("768x768", 768, 768, False), ("672x384", 672, 384, False),
                   ("896x512", 896, 512, False)]
        if model16 is not None:
            configs += [("512x512 fp16", 512, 512, True), ("384x384 fp16", 384, 384, True)]
        for name, w, h, half in configs:
            net = model16 if half else model
            for pre_name, pre in (("pre_numpy", pre_cpu), ("pre_device", pre_dev)):
                rec = {k: [] for k in ("pre", "infer", *POSTS)}
                reps = args.timing_reps if device != "cpu" else max(args.timing_reps // 3, 5)
                for r in range(reps + 5):
                    m = mids[r % len(mids)]
                    sync(device)
                    s0 = time.perf_counter()
                    x = pre(m, w, h, half)
                    sync(device)
                    s1 = time.perf_counter()
                    with torch.no_grad():
                        lg = net(pixel_values=x).logits
                    sync(device)
                    s2 = time.perf_counter()
                    post_t = {}
                    for p in POSTS:
                        s = time.perf_counter()
                        with torch.no_grad():
                            getattr(post, p)(lg)
                        post_t[p] = time.perf_counter() - s
                    if r >= 5:
                        rec["pre"].append(s1 - s0)
                        rec["infer"].append(s2 - s1)
                        for p in POSTS:
                            rec[p].append(post_t[p])
                row = {"device": device, "config": name, "pre": pre_name}
                for k, v in rec.items():
                    row[f"{k}_med"] = round(1000 * np.median(v), 2)
                    row[f"{k}_p95"] = round(1000 * np.percentile(v, 95), 2)
                rows.append(row)
                print(row, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/timing.json", "w") as fh:
        json.dump({"resize": out, "stages": rows}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=N)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--timing-devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--timing-reps", type=int, default=60)
    args = ap.parse_args()
    (timing if args.timing else accuracy)(args)


if __name__ == "__main__":
    main()
