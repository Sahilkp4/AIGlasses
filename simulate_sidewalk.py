"""Replay dataset_flip as a live camera through sidewalk_system.SidewalkNavigator and score it against the labels.

Two walks: frames 1-600 (A) and their mirrored copies 601-1200 (B), each a fresh navigator session. Frames are
released on a wall clock at the camera rate (FPS, estimated from walking distance between frames; --fps overrides).
The navigator always takes the newest frame that has arrived; frames that arrive while it is busy are dropped.

Scores (sidewalk pixels at 960x540, ground truth from eval_sidewalk.gt_mask):
  live       every camera frame, scored with the newest mask that had finished before the next frame arrived (lag and
             drops count; frames before the first result score as an empty prediction)
  processed  each processed frame against its own labels (no lag)
  baseline   raw model argmax sidewalk (as in eval_sidewalk.py) on the same processed frames, computed offline
The prediction scored is the focus area after the Sobel and colour difference checks, which are on by default. Also:
focus-area coverage of the near-field sidewalk, latency per pipeline stage, CPU time, peak memory and drops.

  python3 simulate_sidewalk.py --limit 60      # smoke test
  python3 simulate_sidewalk.py --video         # full run with overlay videos
  python3 simulate_sidewalk.py --walks A --video   # originals only: walk B just mirrors them
  python3 simulate_sidewalk.py --walks A --no-color --out runs/sidewalk_sobel   # score the Sobel-refined focus area
  python3 simulate_sidewalk.py --walks A --no-sobel --out runs/sidewalk_color   # score the colour-refined focus area
  python3 simulate_sidewalk.py --walks A --no-sobel --no-color --out runs/sidewalk_focus   # score the plain focus area

The overlay video colours SegFormer's regions (focus area, other sidewalk, holes in the focus area), each check's
rejections and their combinations, split by sidewalk versus hole, and outlines the labelled sidewalk; see the legend.
"""
import argparse
import csv
import glob
import json
import os
import resource
import sys
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from eval_sidewalk import EVAL_H, EVAL_W, MODEL, gt_mask
from sidewalk_system import SidewalkNavigator, add_color_args, color_check
from sobel_texture import THRESHOLD, WINDOW, SobelTexture, fill_holes

SRC = "data/dataset_flip"
OUT = "runs/sidewalk_system"
FPS = 2.0  # dataset_flip frames are about one walking step apart: ~1.1 m near frames 20-22, typically ~0.7 m at 1.4 m/s
N_ORIGINAL = 600  # frames 601-1200 are mirrored copies of 1-600
WARMUP = 10
NEAR_FIELD = 0.70  # rows below this fraction of the height are the near field for focus coverage
BASELINE_BATCH = 8
LEGEND_ROW_H = 20
RSS_UNIT = 2 ** 20 if sys.platform == "darwin" else 2 ** 10  # ru_maxrss is bytes on macOS, KiB on Linux
# Overlay classes (BGR). Region classes are blended over the frame; rejections are drawn solid.
FOCUS_COLOR = (0, 200, 0)  # focus area, kept by every check
OTHER_SIDEWALK_COLOR = (170, 170, 0)  # SegFormer sidewalk outside the focus area
HOLE_COLOR = (230, 110, 0)  # not sidewalk, enclosed by the focus area, not rejected
SOBEL_COLOR = (255, 0, 255)  # rejected on sidewalk by Sobel only
COLOR_COLOR = (0, 255, 255)  # rejected on sidewalk by the colour check only
BOTH_COLOR = (0, 0, 255)  # rejected on sidewalk by both
HOLE_SOBEL_COLOR = (150, 0, 120)  # in a hole, Sobel only
HOLE_COLOR_COLOR = (0, 140, 255)  # in a hole, colour check only
HOLE_BOTH_COLOR = (0, 0, 130)  # in a hole, both
LABEL_COLOR = (255, 255, 255)  # outline of labelled sidewalk


class Camera:
    """Releases preloaded frames on a wall clock: frame i is available from i / fps."""

    def __init__(self, n, fps):
        self.n, self.fps, self.t0 = n, fps, None

    def start(self):
        self.t0 = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.t0

    def latest(self):
        return min(int(self.now() * self.fps), self.n - 1)

    def done(self):
        return self.now() >= self.n / self.fps


def run_live(nav, frames, fps):
    """Single consumer on a live feed: grab the newest frame when free, drop the rest."""
    cam, results, last = Camera(len(frames), fps), [], -1
    cam.start()
    while not cam.done():
        idx = cam.latest()
        if idx == last:
            time.sleep(max(0.0, (idx + 1) / fps - cam.now()))
            continue
        start = cam.now()
        r = nav.process(frames[idx])
        r["frame"], r["start"], r["end"] = idx, start, cam.now()
        r["mask"] = np.packbits(r.get("refined_focus", r["focus"]))
        r["focus_mask"], r["sidewalk_mask"] = np.packbits(r["focus"]), np.packbits(r["sidewalk"])
        if "anomaly" in r:
            r["rejected"] = np.packbits(r["anomaly"])
        if "color_anomaly" in r:
            r["color_rejected"] = np.packbits(r["color_anomaly"])
        for k in ("focus", "sidewalk", "anomaly", "color_anomaly", "refined_sidewalk", "refined_focus"):
            r.pop(k, None)
        results.append(r)
        last = idx
    return results


def load_gt(paths):
    cache = f"{OUT}/gt_cache.npz"
    names = np.array([os.path.basename(p) for p in paths])
    if os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["names"], names):
            return z["masks"]
    print(f"Rasterising labels for {len(paths)} frames (cached afterwards)...")
    masks = np.stack([np.packbits(gt_mask(p[:-4] + ".json")) for p in paths])
    np.savez(cache, names=names, masks=masks)
    return masks


def unpack(packed):
    return np.unpackbits(packed, count=EVAL_W * EVAL_H).reshape(EVAL_H, EVAL_W).astype(bool)


def baseline_masks(frames, idxs, device):
    """Raw model sidewalk class, post-processed exactly as eval_sidewalk.py does."""
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device).eval()
    sidewalk = next(k for k, v in model.config.id2label.items() if v == "flat-sidewalk")
    out = {}
    for b in range(0, len(idxs), BASELINE_BATCH):
        chunk = idxs[b:b + BASELINE_BATCH]
        inputs = processor(images=[frames[i][:, :, ::-1] for i in chunk], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.nn.functional.interpolate(logits, size=(EVAL_H, EVAL_W), mode="bilinear",
                                                align_corners=False).argmax(1).cpu().numpy()
        out.update({i: p == sidewalk for i, p in zip(chunk, preds)})
    return out


def counts(pred, gt):
    return np.array([(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()], np.int64)


def prf(c):
    tp, fp, fn = c
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"iou": round(tp / max(tp + fp + fn, 1), 4), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def text(img, s, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def legend(img, items, y, draw=True):
    """Draws colour swatches with labels, wrapping rows; returns the baseline of the last row."""
    x = 10
    for label, color in items:
        width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        if x + 17 + width > img.shape[1]:
            x, y = 10, y + LEGEND_ROW_H
        if draw:
            cv2.rectangle(img, (x, y - 11), (x + 12, y + 1), color, -1)
            cv2.rectangle(img, (x, y - 11), (x + 12, y + 1), (90, 90, 90), 1)
            text(img, label, (x + 17, y), 0.45)
        x += 37 + width
    return y


def overlay_classes(sidewalk, focus, sobel, color):
    """(mask, colour, blended) per overlay class. Every pixel is in at most one class."""
    area = fill_holes(focus)  # what the checks score: the focus area and the holes it encloses
    hole = area & ~sidewalk
    flagged = sobel | color
    s_only, c_only, both = sobel & ~color, color & ~sobel, sobel & color
    return [(focus & ~flagged, FOCUS_COLOR, True),
            (sidewalk & ~focus & ~flagged, OTHER_SIDEWALK_COLOR, True),
            (hole & ~flagged, HOLE_COLOR, True),
            (s_only & ~hole, SOBEL_COLOR, False), (c_only & ~hole, COLOR_COLOR, False), (both & ~hole, BOTH_COLOR, False),
            (s_only & hole, HOLE_SOBEL_COLOR, False), (c_only & hole, HOLE_COLOR_COLOR, False),
            (both & hole, HOLE_BOTH_COLOR, False)]


def render(frames, gts, live, results, fps, path, walk, color_name):
    """Camera frames at real speed with the overlay that was live at each frame's arrival: SegFormer regions, Sobel
    and colour rejections and their combinations, split by whether they fall on sidewalk or in a hole of the focus
    area, and the outline of the labelled sidewalk."""
    has_sobel, has_color = "rejected" in results[0], "color_rejected" in results[0]
    items = [("focus area, kept", FOCUS_COLOR), ("SegFormer sidewalk outside focus", OTHER_SIDEWALK_COLOR),
             ("hole in focus (not sidewalk)", HOLE_COLOR)]
    if has_sobel:
        items.append(("Sobel", SOBEL_COLOR))
    if has_color:
        items.append((color_name, COLOR_COLOR))
    if has_sobel and has_color:
        items.append((f"Sobel + {color_name}", BOTH_COLOR))
    if has_sobel:
        items.append(("hole + Sobel", HOLE_SOBEL_COLOR))
    if has_color:
        items.append((f"hole + {color_name}", HOLE_COLOR_COLOR))
    if has_sobel and has_color:
        items.append((f"hole + Sobel + {color_name}", HOLE_BOTH_COLOR))
    items.append(("labelled sidewalk (outline)", LABEL_COLOR))
    header_h = legend(np.zeros((1, EVAL_W, 3), np.uint8), items, 64, draw=False) + 10
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, (EVAL_W, EVAL_H))
    for i, img in enumerate(frames):
        k = live[i]
        out = img.copy()
        gt = unpack(gts[i])
        if k is not None:
            r = results[k]
            focus, sidewalk = unpack(r["focus_mask"]), unpack(r["sidewalk_mask"])
            sobel = unpack(r["rejected"]) if has_sobel else np.zeros_like(focus)
            color = unpack(r["color_rejected"]) if has_color else np.zeros_like(focus)
            classes = overlay_classes(sidewalk, focus, sobel, color)
            blend = out.copy()
            for mask, c, blended in classes:
                if blended:
                    blend[mask] = c
            out = cv2.addWeighted(out, 0.55, blend, 0.45, 0)
            for mask, c, blended in classes:
                if not blended:
                    out[mask] = c
            age = 1000 * (max(r["end"], i / fps) - r["frame"] / fps)
            n_obj = len(r.get("color_objects", []))
            line1 = (f"focus {100 * focus.mean():.1f}% of frame   Sobel {int(sobel.sum())} px   {color_name} "
                     f"{int(color.sum())} px, {n_obj} objects   latency {r['latency_ms']:.0f} ms")
            line2 = f"from frame {r['frame'] + 1}  age {age:.0f} ms"
        else:
            line1, line2 = "waiting for first result", ""
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, LABEL_COLOR, 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (EVAL_W, header_h), (0, 0, 0), -1)
        text(out, f"walk {walk}  t={i / fps:6.2f}s  frame {i + 1}/{len(frames)}   {line2}", (10, 20), 0.5)
        text(out, line1, (10, 42), 0.5, (0, 255, 255))
        legend(out, items, 64)
        writer.write(out)
    writer.release()


def stage_summary(results):
    """Mean, p95 and share of mean latency for each pipeline stage, plus what the stages do not account for."""
    keys = list(results[0]["timing_ms"])
    lat = np.array([r["latency_ms"] for r in results])
    rows = {k: np.array([r["timing_ms"][k] for r in results]) for k in keys}
    rows["unaccounted"] = lat - sum(rows.values())
    return {k: {"mean": round(float(v.mean()), 2), "p95": round(float(np.percentile(v, 95)), 2),
                "share_pct": round(100 * float(v.mean()) / float(lat.mean()), 1)} for k, v in rows.items()}


def evaluate_walk(name, frames, gts, results, fps, device, video, out, color_name="colour"):
    n = len(frames)
    # live: newest result finished before the next frame arrived
    live, k = [None] * n, -1
    for i in range(n):
        while k + 1 < len(results) and results[k + 1]["end"] <= (i + 1) / fps:
            k += 1
        live[i] = k if k >= 0 else None

    print(f"  computing raw-model baseline on {len(results)} processed frames...")
    base = baseline_masks(frames, [r["frame"] for r in results], device)

    c_live, c_proc, c_base = np.zeros(3, np.int64), np.zeros(3, np.int64), np.zeros(3, np.int64)
    cover_num, cover_den = 0, 0
    rows, by_frame = [], {r["frame"]: r for r in results}
    for i in range(n):
        gt = unpack(gts[i])
        kk = live[i]
        pred = unpack(results[kk]["mask"]) if kk is not None else np.zeros_like(gt)
        cl = counts(pred, gt)
        c_live += cl
        row = {"walk": name, "frame": i + 1, "gt_px": int(cl[0] + cl[2]), "processed": int(i in by_frame),
               "live_from_frame": results[kk]["frame"] + 1 if kk is not None else "",
               "live_age_ms": round(1000 * (i - results[kk]["frame"]) / fps) if kk is not None else "",
               "live_iou": prf(cl)["iou"] if cl.sum() else ""}
        if i in by_frame:
            r = by_frame[i]
            own = unpack(r["mask"])
            cp, cb = counts(own, gt), counts(base[i], gt)
            c_proc += cp
            c_base += cb
            near = gt.copy()
            near[: int(EVAL_H * NEAR_FIELD)] = False
            cover_num += int((near & own).sum())
            cover_den += int(near.sum())
            row.update({"proc_iou": prf(cp)["iou"] if cp.sum() else "", "raw_iou": prf(cb)["iou"] if cb.sum() else "",
                        "latency_ms": r["latency_ms"]})
            row.update(r["timing_ms"])
        rows.append(row)

    if video:
        print("  rendering overlay video...")
        render(frames, gts, live, results, fps, f"{out}/overlay_walk{name}.mp4", name, color_name)

    lat = np.array([r["latency_ms"] for r in results])
    ages = np.array([1000 * (r["end"] - r["frame"] / fps) for r in results])  # capture -> result available
    dur = n / fps
    summary = {
        "frames": n, "processed": len(results), "dropped_pct": round(100 * (1 - len(results) / n), 1),
        "live": prf(c_live), "processed_no_lag": prf(c_proc), "raw_model_baseline": prf(c_base),
        "focus_coverage_near_field": round(cover_num / max(cover_den, 1), 4),
        "latency_ms": {"mean": round(lat.mean(), 1), "p95": round(np.percentile(lat, 95), 1), "max": round(lat.max(), 1)},
        "capture_to_result_ms": {"mean": round(ages.mean(), 1), "p95": round(np.percentile(ages, 95), 1)},
        "update_hz": round(len(results) / dur, 2),
    }
    if "sobel_ms" in results[0]:
        sobel = np.array([r["sobel_ms"] for r in results])
        summary["sobel_ms"] = {"mean": round(sobel.mean(), 2), "p95": round(np.percentile(sobel, 95), 2)}
    if "color_ms" in results[0]:
        ms = np.array([r["color_ms"] for r in results])
        summary["color_ms"] = {"mean": round(ms.mean(), 2), "p95": round(np.percentile(ms, 95), 2)}
    summary["stages_ms"] = stage_summary(results)
    if "color_objects" in results[0]:
        summary["color_objects"] = sum(len(r["color_objects"]) for r in results)
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--fps", type=float, default=FPS, help="camera capture rate")
    ap.add_argument("--limit", type=int, default=0, help="frames per walk")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--walks", default="AB", help="walks to run: A (frames 1-600), B (mirrored 601-1200) or AB")
    ap.add_argument("--out", default=OUT, help="output directory")
    ap.add_argument("--sobel", action=argparse.BooleanOptionalAction, default=True, help="run the Sobel texture check")
    ap.add_argument("--sobel-window", type=int, nargs=2, default=WINDOW, metavar=("W", "H"), help="sliding-window width and height in pixels, odd")
    ap.add_argument("--sobel-threshold", type=float, default=THRESHOLD, help="spreads above normal texture before a pixel is rejected")
    add_color_args(ap)
    args = ap.parse_args()
    color_name = "colour"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    fps = args.fps

    paths = sorted(glob.glob(f"{SRC}/*.jpg"))
    gts_all = load_gt(paths)
    walks = {k: v for k, v in {"A": range(0, N_ORIGINAL), "B": range(N_ORIGINAL, len(paths))}.items()
             if k in args.walks.upper()}
    nav = SidewalkNavigator(args.device, SobelTexture(args.sobel_window, args.sobel_threshold) if args.sobel else None,
                            color_check(args))

    summaries, all_rows = {}, []
    for name, rng in walks.items():
        idx = list(rng)[: args.limit or None]
        print(f"Walk {name}: preloading {len(idx)} frames at {EVAL_W}x{EVAL_H}...")
        frames = [cv2.resize(cv2.imread(paths[i], cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H)) for i in idx]
        for img in frames[:WARMUP]:
            nav.process(img)
        print(f"  playing {len(frames) / fps:.1f}s at {fps:g} FPS on {args.device}...")
        cpu0 = time.process_time()
        results = run_live(nav, frames, fps)
        cpu_ms = 1000 * (time.process_time() - cpu0)  # all threads of this process, including idle waits' overhead
        s, rows = evaluate_walk(name, frames, gts_all[idx], results, fps, args.device, args.video, args.out,
                                color_name)
        s["cpu"] = {"ms_per_processed_frame": round(cpu_ms / len(results), 1),
                    "utilisation_pct_of_one_core": round(100 * cpu_ms / (1000 * len(frames) / fps), 1),
                    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_UNIT, 0)}
        summaries[name] = s
        all_rows += rows
        del frames

    fields = list(dict.fromkeys(k for r in all_rows for k in r))
    with open(f"{args.out}/per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    summary = {"fps": fps, "walks": summaries}
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=2)

    print()
    for name, s in summaries.items():
        print(f"Walk {name}: {s['processed']}/{s['frames']} frames processed ({s['dropped_pct']}% dropped), "
              f"{s['update_hz']} Hz, latency mean {s['latency_ms']['mean']} ms p95 {s['latency_ms']['p95']} ms")
        for k in ("live", "processed_no_lag", "raw_model_baseline"):
            m = s[k]
            print(f"  {k:20s} IoU {m['iou']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")
        print(f"  focus covers {100 * s['focus_coverage_near_field']:.1f}% of near-field sidewalk")
        if "sobel_ms" in s:
            print(f"  sobel mean {s['sobel_ms']['mean']} ms p95 {s['sobel_ms']['p95']} ms")
        if "color_ms" in s:
            print(f"  colour mean {s['color_ms']['mean']} ms p95 {s['color_ms']['p95']} ms, "
                  f"{s.get('color_objects', 0)} objects")
        print("  stage                mean ms    p95 ms   share of latency")
        for k, v in s["stages_ms"].items():
            print(f"  {k:18s} {v['mean']:9.2f} {v['p95']:9.2f} {v['share_pct']:9.1f}%")
        c = s["cpu"]
        print(f"  CPU {c['ms_per_processed_frame']} ms per processed frame "
              f"({c['utilisation_pct_of_one_core']}% of one core over the walk), peak RSS {c['peak_rss_mb']:.0f} MB")
    print(f"\nOutputs in {args.out}/ (summary.json, per_frame.csv{', overlay_walk*.mp4' if args.video else ''})")


if __name__ == "__main__":
    main()
