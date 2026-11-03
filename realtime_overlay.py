"""Play dataset_flip as a live camera and run YOLO11n, the sidewalk SegFormer and Depth Anything V2 (metric)
on it in real time.

dataset_flip holds 600 real frames followed by 600 horizontally mirrored copies; only the originals are
played. The real capture rate is unknown, so it is estimated by comparing frame-to-frame motion with
IMG_9858.MOV, a 30 FPS walking clip (cached in fps_estimate.json; --fps overrides).

Frames are released on a wall clock (frame i becomes available at i/fps). Each model runs in its own
thread and, whenever it is free, grabs the newest available frame; frames that arrive while it is busy
are dropped, as they would be on a live feed. Every result is stamped with the frame it was computed on
and the time it finished.

The output video is rendered on a DISPLAY_FPS clock that runs at real-world speed: at time t it shows the
newest camera frame that had arrived and the latest results of each model that had *finished* by t.
Object distances combine the latest YOLO boxes with the latest depth map, as live fusion would.

  python3 realtime_overlay.py --device mps
"""
import argparse
import csv
import glob
import json
import os
import threading
import time

import cv2
import numpy as np
import torch
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          SegformerForSemanticSegmentation)
from ultralytics import YOLO

SRC = "dataset_flip"
OUT = "runs/realtime_overlay"
SEG_MODEL = "tobiasc/segformer-b0-finetuned-segments-sidewalk"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
YOLO_MODEL = "yolo11n.pt"
REF_VIDEO, REF_FPS = "IMG_9858.MOV", 30  # walking clip with a known frame rate, for the fps estimate
N_ORIGINAL = 600  # frames 601-1200 are mirrored copies of 1-600
DISPLAY_FPS = 30
W, H = 960, 540  # 4K source decoded at 1/4 scale
DW, DH = 480, 270  # stored depth map size
WARMUP = 10
SEG_ALPHA = 0.45
LEGEND_TOP = 6
HUD_H = 100
INSET_W, INSET_H, MAX_DEPTH_M = 240, 135, 40


class Camera:
    """Releases preloaded frames on a wall clock."""

    def __init__(self, frames, fps):
        self.frames, self.fps, self.t0 = frames, fps, None

    def start(self):
        self.t0 = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.t0

    def latest(self):
        return min(int(self.now() * self.fps), len(self.frames) - 1)

    def done(self):
        return self.now() >= len(self.frames) / self.fps


class Worker(threading.Thread):
    """Runs `fn` on the newest camera frame whenever free; records (frame, start, end, output)."""

    def __init__(self, name, cam, fn):
        super().__init__(daemon=True, name=name)
        self.cam, self.fn, self.results = cam, fn, []

    def run(self):
        last = -1
        while not self.cam.done():
            idx = self.cam.latest()
            if idx == last:  # nothing new yet; wait for the next frame tick
                time.sleep(max(0.0, (idx + 1) / self.cam.fps - self.cam.now()))
                continue
            t_start = self.cam.now()
            out = self.fn(self.cam.frames[idx])
            self.results.append({"frame": idx, "start": t_start, "end": self.cam.now(), "out": out})
            last = idx


def flow_gray(img):
    return cv2.cvtColor(cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def median_flow(a, b):
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0)
    return float(np.median(np.linalg.norm(f, axis=2)))


def estimate_fps(paths):
    """Capture rate from motion per frame vs the 30 FPS reference clip (assumes similar walking speed and FOV)."""
    cache = f"{OUT}/fps_estimate.json"
    if os.path.exists(cache):
        return json.load(open(cache))["fps"]
    print("Estimating capture rate from frame-to-frame motion...")
    grays = [flow_gray(cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)) for p in paths]
    data = np.median([median_flow(a, b) for a, b in zip(grays, grays[1:])])
    cap, ref, prev = cv2.VideoCapture(REF_VIDEO), [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = flow_gray(f)
        if prev is not None:
            ref.append(median_flow(prev, g))
        prev = g
    moving = np.median([m for m in ref if m > 0.5])  # drop segments where the walker stood still
    fps = round(float(REF_FPS * moving / data), 2)
    json.dump({"fps": fps, "dataset_px_per_frame": float(data), "ref_px_per_frame": float(moving),
               "ref_moving_pairs": int(sum(m > 0.5 for m in ref)), "ref_pairs": len(ref)}, open(cache, "w"), indent=2)
    print(f"  dataset {data:.2f} px/frame vs {REF_VIDEO} {moving:.2f} px/frame at {REF_FPS} FPS -> ~{fps} FPS "
          f"(assumes similar walking speed and field of view)")
    return fps


def make_seg(device):
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()

    def run(img):
        inputs = processor(images=img[:, :, ::-1], return_tensors="pt").to(device)
        with torch.no_grad():
            # argmax at native 128x128, then nearest-resize the class map (~97% sidewalk IoU vs full upsample)
            cls = model(**inputs).logits.argmax(1).cpu().numpy()[0].astype(np.uint8)
        return cv2.resize(cls, (W, H), interpolation=cv2.INTER_NEAREST)

    return run, model.config.id2label


def make_depth(device):
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()

    def run(img):
        inputs = processor(images=img[:, :, ::-1], return_tensors="pt").to(device)
        with torch.no_grad():
            depth = model(**inputs).predicted_depth[0].cpu().numpy()  # meters
        return cv2.resize(depth, (DW, DH), interpolation=cv2.INTER_LINEAR).astype(np.float16)

    return run


def make_yolo(device):
    model = YOLO(YOLO_MODEL)

    def run(img):
        r = model.predict(img, device=device, verbose=False)[0]
        return [(model.names[int(c)], float(p), *map(int, b))
                for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist())]

    return run


def palette(n):
    rng = np.random.default_rng(3)
    pal = rng.integers(40, 255, (n, 3), dtype=np.uint8)
    return pal


def latest_before(results, t, k=0):
    """Advance pointer k to the last result finished by time t; return (k, result or None)."""
    while k + 1 < len(results) and results[k + 1]["end"] <= t:
        k += 1
    return k, (results[k] if results and results[k]["end"] <= t else None)


def box_depth(depth, x1, y1, x2, y2):
    """Median depth (m) over the inner 50% of a box given in W x H pixels."""
    sx, sy = DW / W, DH / H
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
    a, b = int((cx - hw) * sx), int(np.ceil((cx + hw) * sx))
    c, d = int((cy - hh) * sy), int(np.ceil((cy + hh) * sy))
    patch = depth[max(0, c):max(c + 1, d), max(0, a):max(a + 1, b)]
    return float(np.median(patch)) if patch.size else float("nan")


def text(img, s, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(frames, fps, seg_res, yolo_res, depth_res, labels, path):
    pal = palette(len(labels))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), DISPLAY_FPS, (W, H))
    ks = ky = kd = 0
    for j in range(int(np.ceil(len(frames) / fps * DISPLAY_FPS))):
        t = j / DISPLAY_FPS
        cam_idx = min(int(t * fps), len(frames) - 1)
        ks, s = latest_before(seg_res, t, ks)
        ky, y = latest_before(yolo_res, t, ky)
        kd, d = latest_before(depth_res, t, kd)
        out = frames[cam_idx].copy()

        if s is not None:
            cls = s["out"]
            out = cv2.addWeighted(out, 1 - SEG_ALPHA, pal[cls], SEG_ALPHA, 0)
            ids, counts = np.unique(cls, return_counts=True)
            for n, i in enumerate(ids[np.argsort(-counts)][:LEGEND_TOP]):
                yy = HUD_H + 30 + n * 20
                cv2.rectangle(out, (W - 190, yy - 12), (W - 176, yy + 2), pal[i].tolist(), -1)
                text(out, labels[int(i)], (W - 170, yy), 0.45)

        if y is not None:
            for name, conf, x1, y1, x2, y2 in y["out"]:
                label = f"{name} {conf:.2f}"
                if d is not None:
                    label += f"  {box_depth(d['out'], x1, y1, x2, y2):.1f} m"
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                text(out, label, (x1, max(HUD_H + 14, y1 - 4)), 0.45, (0, 255, 255))

        if d is not None:
            dm = np.clip(d["out"].astype(np.float32) / MAX_DEPTH_M, 0, 1)
            inset = cv2.applyColorMap(((1 - dm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # near = warm
            y0 = H - INSET_H - 24
            out[y0:y0 + INSET_H, 10:10 + INSET_W] = cv2.resize(inset, (INSET_W, INSET_H))
            cv2.rectangle(out, (10, y0), (10 + INSET_W, y0 + INSET_H), (255, 255, 255), 1)
            text(out, f"depth  0 m (red) ... {MAX_DEPTH_M} m+ (blue)", (10, H - 8), 0.42)

        cv2.rectangle(out, (0, 0), (W, HUD_H), (0, 0, 0), -1)
        text(out, f"LIVE  t={t:6.2f}s  camera {fps:g} FPS  frame {cam_idx + 1}/{len(frames)}", (10, 20), 0.55)
        rows = [("YOLO11n", y, (0, 255, 255)), ("SegFormer", s, (255, 180, 80)), ("DepthAnyV2", d, (180, 130, 255))]
        for row, (name, r, col) in enumerate(rows):
            msg = "waiting for first result" if r is None else (
                f"result from frame {r['frame'] + 1}  latency {1000 * (r['end'] - r['start']):5.0f} ms  "
                f"overlay age {1000 * (t - r['frame'] / fps):5.0f} ms")
            text(out, f"{name:11s} {msg}", (10, 44 + 22 * row), 0.5, col)
        writer.write(out)
    writer.release()


def summarize(name, res, n_frames, fps):
    lat = np.array([1000 * (r["end"] - r["start"]) for r in res])
    dur = n_frames / fps
    print(f"{name:10s} processed {len(res):4d}/{n_frames} frames ({100 * len(res) / n_frames:5.1f}%)  "
          f"effective {len(res) / dur:5.1f} Hz  latency mean {lat.mean():5.1f} ms  p95 {np.percentile(lat, 95):5.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--fps", type=float, help="camera capture rate (default: estimated from motion)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    paths = sorted(glob.glob(f"{SRC}/*.jpg"))[:N_ORIGINAL]
    fps = args.fps or estimate_fps(paths)
    paths = paths[: args.limit or None]
    print(f"Preloading {len(paths)} frames at {W}x{H} (camera buffer)...")
    frames = [cv2.resize(cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4), (W, H)) for p in paths]

    seg_fn, labels = make_seg(args.device)
    yolo_fn = make_yolo(args.device)
    depth_fn = make_depth(args.device)
    for img in frames[:WARMUP]:
        seg_fn(img); yolo_fn(img); depth_fn(img)

    cam = Camera(frames, fps)
    workers = [Worker("yolo", cam, yolo_fn), Worker("seg", cam, seg_fn), Worker("depth", cam, depth_fn)]
    print(f"Playing {len(frames) / fps:.1f}s of video at {fps:g} FPS on {args.device}...")
    cam.start()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    yolo_res, seg_res, depth_res = (w.results for w in workers)

    summarize("YOLO11n", yolo_res, len(frames), fps)
    summarize("SegFormer", seg_res, len(frames), fps)
    summarize("DepthAnyV2", depth_res, len(frames), fps)

    with open(f"{OUT}/timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "frame", "start_s", "end_s", "latency_ms", "n_detections"])
        for name, res in [("yolo11n", yolo_res), ("segformer", seg_res), ("depth_anything_v2", depth_res)]:
            for r in res:
                w.writerow([name, r["frame"] + 1, f"{r['start']:.4f}", f"{r['end']:.4f}",
                            f"{1000 * (r['end'] - r['start']):.1f}", len(r["out"]) if name == "yolo11n" else ""])

    print("Rendering overlay video...")
    render(frames, fps, seg_res, yolo_res, depth_res, labels, f"{OUT}/realtime_overlay.mp4")
    print(f"Video:    {OUT}/realtime_overlay.mp4 ({len(frames) / fps:.1f}s, real-time speed)\nTimeline: {OUT}/timeline.csv")


if __name__ == "__main__":
    main()
