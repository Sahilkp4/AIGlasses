"""Measure the computational load of running the sidewalk SegFormer and YOLO11n together on dataset_flip.

Frames are fed one at a time (batch 1) like a live camera. Modes:
  yolo, seg     each model alone (baselines)
  sequential    both models, one after the other, on every frame
  parallel      both models on every frame at once, in two threads
A background thread samples process CPU %, RSS and MPS memory every 100 ms.

  python3 benchmark_combined.py --device mps --limit 50
"""
import argparse
import csv
import glob
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import psutil
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO

SRC = "dataset_flip"
OUT = "runs/combined_benchmark"
SEG_MODEL = "tobiasc/segformer-b0-finetuned-segments-sidewalk"
YOLO_MODEL = "yolo11n.pt"
W, H = 960, 540
WARMUP = 20
SAMPLE_S = 0.1
MODES = ["yolo", "seg", "sequential", "parallel"]


class Sampler(threading.Thread):
    """Samples CPU %, RSS and MPS memory of this process until stopped."""

    def __init__(self, device):
        super().__init__(daemon=True)
        self.proc, self.mps = psutil.Process(), device == "mps"
        self.cpu, self.rss, self.mps_mem = [], [], []
        self._halt = threading.Event()

    def run(self):
        self.proc.cpu_percent(None)
        ncpu = psutil.cpu_count()
        while not self._halt.wait(SAMPLE_S):
            self.cpu.append(self.proc.cpu_percent(None) / ncpu)
            self.rss.append(self.proc.memory_info().rss / 2**20)
            if self.mps:
                self.mps_mem.append(torch.mps.driver_allocated_memory() / 2**20)

    def stop(self):
        self._halt.set()
        self.join()


class Seg:
    def __init__(self, device, post):
        self.device, self.post = device, post
        self.processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()

    def __call__(self, img):
        t = time.perf_counter()
        inputs = self.processor(images=img[:, :, ::-1], return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            if self.post == "full":  # same as eval_sidewalk.py: upsample 35-class logits, then argmax
                torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear",
                                                align_corners=False).argmax(1).cpu()  # .cpu() forces a sync
            else:  # argmax at native 128x128, then nearest-resize the class map (~97% sidewalk IoU vs full)
                cls = logits.argmax(1).cpu().numpy()[0].astype(np.uint8)
                cv2.resize(cls, (W, H), interpolation=cv2.INTER_NEAREST)
        return (time.perf_counter() - t) * 1000


class Det:
    def __init__(self, device):
        self.device = device
        self.model = YOLO(YOLO_MODEL)

    def __call__(self, img):
        t = time.perf_counter()
        self.model.predict(img, device=self.device, verbose=False)
        return (time.perf_counter() - t) * 1000


def run_mode(mode, frames, seg, det, device, tag):
    pool = ThreadPoolExecutor(2)

    def step(img):
        if mode == "yolo":
            return 0.0, det(img)
        if mode == "seg":
            return seg(img), 0.0
        if mode == "sequential":
            return seg(img), det(img)
        fs, fd = pool.submit(seg, img), pool.submit(det, img)
        return fs.result(), fd.result()

    for p in frames[:WARMUP]:
        step(cv2.resize(cv2.imread(p), (W, H), interpolation=cv2.INTER_AREA))

    rows, sampler = [], Sampler(device)
    sampler.start()
    for p in frames:
        t0 = time.perf_counter()
        img = cv2.resize(cv2.imread(p), (W, H), interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()
        seg_ms, yolo_ms = step(img)
        t2 = time.perf_counter()
        rows.append({"frame": os.path.basename(p), "io_ms": round((t1 - t0) * 1000, 2),
                     "seg_ms": round(seg_ms, 2), "yolo_ms": round(yolo_ms, 2),
                     "model_ms": round((t2 - t1) * 1000, 2), "total_ms": round((t2 - t0) * 1000, 2)})
    sampler.stop()
    pool.shutdown()

    with open(f"{OUT}/per_frame_{tag}_{mode}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    m = np.array([r["model_ms"] for r in rows])
    tot = np.array([r["total_ms"] for r in rows])
    return rows, {
        "device": device, "seg_post": seg.post, "mode": mode, "frames": len(rows),
        "model_ms_mean": m.mean(), "model_ms_p50": np.median(m), "model_ms_p95": np.percentile(m, 95),
        "model_ms_max": m.max(), "model_fps": 1000 / m.mean(), "e2e_fps": 1000 / tot.mean(),
        "io_ms_mean": np.mean([r["io_ms"] for r in rows]),
        "seg_ms_mean": np.mean([r["seg_ms"] for r in rows]),
        "yolo_ms_mean": np.mean([r["yolo_ms"] for r in rows]),
        "cpu_pct_mean": np.mean(sampler.cpu or [0]), "cpu_pct_peak": max(sampler.cpu or [0]),
        "rss_mb_peak": max(sampler.rss or [0]), "mps_mb_peak": max(sampler.mps_mem or [0]),
    }


def plot(results, device, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode, rows in results.items():
        ax.plot([r["model_ms"] for r in rows], lw=0.8, label=mode)
    ax.axhline(1000 / 30, color="k", ls="--", lw=1, label="30 FPS budget (33.3 ms)")
    ax.set(xlabel="frame", ylabel="model latency (ms)", title=f"Per-frame model latency on {device} (seg post: {tag.split('_')[1]})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/latency_{tag}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--seg-post", default="full", choices=["full", "fast"],
                    help="full = upsample logits then argmax (as evaluated); fast = argmax at low res")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N frames")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    frames = sorted(glob.glob(f"{SRC}/*.jpg"))
    if args.limit:
        frames = frames[:args.limit]
    tag = f"{args.device}_{args.seg_post}"
    print(f"Device: {args.device}   seg post: {args.seg_post}   frames: {len(frames)}   models: {SEG_MODEL} + {YOLO_MODEL}")

    seg, det = Seg(args.device, args.seg_post), Det(args.device)
    results, summary = {}, []
    for mode in args.modes:
        print(f"\n[{mode}] running...", flush=True)
        results[mode], s = run_mode(mode, frames, seg, det, args.device, tag)
        summary.append(s)
        print(f"  model {s['model_ms_mean']:.1f} ms/frame ({s['model_fps']:.1f} FPS), "
              f"end-to-end {s['e2e_fps']:.1f} FPS, CPU {s['cpu_pct_mean']:.0f}% avg")

    base = {s["mode"]: s["model_ms_mean"] for s in summary}
    for s in summary:
        if "yolo" in base and "seg" in base and s["mode"] in ("sequential", "parallel"):
            s["vs_sum_of_baselines"] = s["model_ms_mean"] / (base["yolo"] + base["seg"])

    path = f"{OUT}/summary_{tag}.csv"
    keys = list(summary[0]) + ["vs_sum_of_baselines"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows([{k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()} for s in summary])
    plot(results, args.device, tag)

    print(f"\n{'mode':12s}{'mean ms':>9s}{'p95 ms':>9s}{'FPS':>7s}{'e2e FPS':>9s}{'CPU%':>7s}"
          f"{'CPU pk':>8s}{'RSS MB':>8s}{'MPS MB':>8s}{'vs sum':>8s}")
    for s in summary:
        vs = f"{s['vs_sum_of_baselines']:.2f}" if "vs_sum_of_baselines" in s else "-"
        print(f"{s['mode']:12s}{s['model_ms_mean']:9.1f}{s['model_ms_p95']:9.1f}{s['model_fps']:7.1f}"
              f"{s['e2e_fps']:9.1f}{s['cpu_pct_mean']:7.0f}{s['cpu_pct_peak']:8.0f}{s['rss_mb_peak']:8.0f}"
              f"{s['mps_mb_peak']:8.0f}{vs:>8s}")
    print(f"\nCSV: {path}\nPlot: {OUT}/latency_{tag}.png")
    print("Power draw (optional, separate terminal): sudo powermetrics --samplers cpu_power,gpu_power -i 1000")


if __name__ == "__main__":
    main()
