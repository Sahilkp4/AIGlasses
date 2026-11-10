"""Effect of a new SegFormer front end on the whole model: focus area, Sobel texture check and colour difference check.

Two front ends, the same model weights and the same check settings (the defaults in sobel_texture.py and
color_difference.py), run on the same frames:
  current  512x512 square input, argmax at 1/4 resolution, nearest upsample (sidewalk_system.py before the margin change)
  new      672x384 input (keeps 16:9), margin post-processing: max sidewalk logit minus max other logit at 1/4
           resolution, bilinear upsample of that one channel, > 0 (see eval_segformer_input.py)

Scores, walk A (frames 1-600), labels from eval_sidewalk.gt_mask, frames from eval_targets.load (as the other evals):
  focus            IoU/P/R of the focus area, near-field coverage, holes per frame and their area
  refined          IoU/P/R of the focus area after both checks (what simulate_sidewalk.py scores)
  per check        false flags (share of labelled sidewalk in the focus area flagged, clean frames), flags off label,
                   share of false flags within one Sobel window of the focus edge, objects per minute (colour),
                   catch of each hand-outlined target inside the checked area and its *_seen share, stage time,
                   frames with no reference

  python3 eval_segformer_pipeline.py
  python3 eval_segformer_pipeline.py --limit 60      # smoke test
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from color_difference import ColorDifference
from eval_sidewalk import gt_mask
from eval_targets import NOT_CLEAN, TARGETS, load, target_masks
from sidewalk_system import MODEL_IN, SidewalkNavigator
from simulate_sidewalk import FPS, NEAR_FIELD, SRC
from sobel_texture import WINDOW, SobelTexture, fill_holes

OUT = "runs/segformer_pipeline_eval"
NEW_IN = (672, 384)  # width, height


class FrontEnd(SidewalkNavigator):
    """SidewalkNavigator whose segment() can run either front end and returns the boolean sidewalk mask process() uses.
      "current"  the front end before the margin change: 512x512 input, argmax at 1/4 resolution, nearest upsample
      "new"      the margin method of sidewalk_system.py, at input size NEW_IN (512x512 is sidewalk_system.py as is)"""

    mode = "current"

    def segment(self, frame, timing=None):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        x = self._preprocess(frame, (MODEL_IN, MODEL_IN) if self.mode == "current" else NEW_IN)
        self._sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            lg = self.model(pixel_values=x).logits
            self._sync()
            t2 = time.perf_counter()
            if self.mode == "current":
                cls = F.interpolate(lg.argmax(1, keepdim=True).float(), size=(h, w), mode="nearest")[0, 0]
                out = np.isin(cls.cpu().numpy().astype(np.uint8), self.sidewalk_ids)
            else:
                out = (self.sidewalk_margin(lg, (h, w)) > 0).cpu().numpy()
        if timing is not None:
            timing.update({"preprocess_ms": 1000 * (t1 - t0), "inference_ms": 1000 * (t2 - t1),
                           "postprocess_ms": 1000 * (time.perf_counter() - t2)})
        return out


def prf(tp, fp, fn):
    return {"iou": round(tp / max(tp + fp + fn, 1), 4), "p": round(tp / max(tp + fp, 1), 4),
            "r": round(tp / max(tp + fn, 1), 4)}


class Tally:
    def __init__(self, targets):
        self.c = {}
        self.targets = targets
        self.ms = {"sobel_ms": [], "color_ms": [], "segment_ms": [], "latency_ms": []}

    def add(self, key, v):
        self.c[key] = self.c.get(key, 0) + int(v)

    def frame(self, i, frame, r, gt, clean, edge_band):
        sw, focus = r["sidewalk"], r["focus"]
        area = fill_holes(focus)
        holes = area & ~focus
        for name, m in (("focus", focus), ("refined", r["refined_focus"]), ("sidewalk", sw)):
            self.add(f"{name}_tp", (m & gt).sum())
            self.add(f"{name}_fp", (m & ~gt).sum())
            self.add(f"{name}_fn", (~m & gt).sum())
        near = gt.copy()
        near[: int(gt.shape[0] * NEAR_FIELD)] = False
        self.add("near_hit", (near & focus).sum())
        self.add("near_all", near.sum())
        self.add("focus_px", focus.sum())
        self.add("holes_px", holes.sum())
        self.add("holes_n", cv2.connectedComponents(holes.astype(np.uint8), connectivity=4)[0] - 1)
        self.add("frames", 1)
        self.add("sobel_noref", r["reference"] is None)
        self.add("color_noref", r["color_reference"] is None)
        band = focus & ~cv2.erode(focus.astype(np.uint8), edge_band).astype(bool)
        for chk, flags in (("sobel", r["anomaly"]), ("color", r["color_anomaly"]),
                           ("both", r["anomaly"] | r["color_anomaly"])):
            for k, m in target_masks(self.targets, i, frame).items():
                self.add(f"{chk}_{k}_hit", (m & area & flags).sum())
                self.add(f"{chk}_{k}_in", (m & area).sum())
                self.add(f"{chk}_{k}_all", m.sum())
            if clean:
                self.add(f"{chk}_false", (flags & focus & gt).sum())
                self.add(f"{chk}_false_edge", (flags & band & gt).sum())
                self.add(f"{chk}_off", (flags & ~gt).sum())
        if clean:
            self.add("clean_px", (focus & gt).sum())
            self.add("clean_band_px", (band & gt).sum())
            self.add("clean_frames", 1)
            self.add("color_objects", len(r["color_objects"]))
        t = r["timing_ms"]
        self.ms["sobel_ms"].append(t.get("sobel_ms", 0))
        self.ms["color_ms"].append(t.get("color_ms", 0))
        self.ms["segment_ms"].append(t["preprocess_ms"] + t["inference_ms"] + t["postprocess_ms"])
        self.ms["latency_ms"].append(r["latency_ms"])

    def summary(self):
        c, g = self.c, self.c.get
        s = {"focus": prf(c["focus_tp"], c["focus_fp"], c["focus_fn"]),
             "sidewalk": prf(c["sidewalk_tp"], c["sidewalk_fp"], c["sidewalk_fn"]),
             "refined_focus": prf(c["refined_tp"], c["refined_fp"], c["refined_fn"]),
             "near_field_coverage_pct": round(100 * c["near_hit"] / max(c["near_all"], 1), 1),
             "focus_frac_pct": round(100 * c["focus_px"] / (c["frames"] * 960 * 540), 2),
             "holes_per_frame": round(c["holes_n"] / c["frames"], 2),
             "hole_px_per_frame": round(c["holes_px"] / c["frames"]),
             "edge_band_share_of_clean_focus_pct": round(100 * c["clean_band_px"] / max(c["clean_px"], 1), 1),
             "frames_no_reference": {"sobel": c["sobel_noref"], "color": c["color_noref"]},
             "color_objects_per_min": round(60 * FPS * c["color_objects"] / max(c["clean_frames"], 1), 1)}
        for chk in ("sobel", "color", "both"):
            d = {"false_flags_pct": round(100 * g(f"{chk}_false", 0) / max(c["clean_px"], 1), 3),
                 "false_flags_in_edge_band_pct_of_false": round(100 * g(f"{chk}_false_edge", 0) /
                                                                max(g(f"{chk}_false", 0), 1), 1),
                 "flags_off_label_pct": round(100 * g(f"{chk}_off", 0) /
                                              max(g(f"{chk}_off", 0) + g(f"{chk}_false", 0), 1), 1)}
            for k in self.targets:
                d[f"{k}_catch_pct"] = round(100 * g(f"{chk}_{k}_hit", 0) / max(g(f"{chk}_{k}_in", 0), 1), 1)
                d[f"{k}_seen_pct"] = round(100 * g(f"{chk}_{k}_in", 0) / max(g(f"{chk}_{k}_all", 0), 1), 1)
            s[chk] = d
        s["ms"] = {k: {"mean": round(float(np.mean(v)), 2), "p95": round(float(np.percentile(v, 95)), 2)}
                   for k, v in self.ms.items()}
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()
    cv2.setNumThreads(1)  # as eval_color_difference.py, so check times compare with earlier runs
    targets = {k: v for k, v in json.load(open(TARGETS)).items() if not k.startswith("_")}
    nav = FrontEnd(args.device, SobelTexture(), ColorDifference())
    edge_band = np.ones((2 * (WINDOW[1] // 2) + 1, 2 * (WINDOW[0] // 2) + 1), np.uint8)  # one Sobel window
    tallies = {m: Tally(targets) for m in ("current", "new")}
    agree = []
    for m in tallies:  # warm up both paths
        nav.mode = m
        nav.process(load(1))
    t0 = time.time()
    for i in range(1, args.limit + 1):
        frame, gt = load(i), gt_mask(f"{SRC}/{i:06d}.json")
        clean = not any(i in r for r in NOT_CLEAN)
        focus = {}
        for m, tally in tallies.items():
            nav.mode = m
            r = nav.process(frame)
            tally.frame(i, frame, r, gt, clean, edge_band)
            focus[m] = r["focus"]
        agree.append((focus["current"] & focus["new"]).sum() / max((focus["current"] | focus["new"]).sum(), 1))
        if i % 100 == 0:
            print(f"{i}/{args.limit}  {time.time() - t0:.0f}s", flush=True)
    out = {m: t.summary() for m, t in tallies.items()}
    out["focus_iou_current_vs_new"] = {"mean": round(float(np.mean(agree)), 4),
                                       "p10": round(float(np.percentile(agree, 10)), 4)}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
