"""Sidewalk focus area: segments each frame and marks the largest connected region labelled sidewalk.

Segmentation: tobiasc/segformer-b0-finetuned-segments-sidewalk. A pixel is sidewalk when its top class is one of SIDEWALK,
decided by the margin method: at the model's 1/4 output resolution, the best SIDEWALK logit minus the best other logit,
bilinearly upsampled to the frame and kept where positive. At each output cell this is the same decision as argmax; the
bilinear upsample is how SegFormer upsamples logits in training, and it is faster than argmax plus a nearest upsample
of the class map (eval_segformer_input.py). The focus mask itself never includes non-sidewalk pixels. The checks also score the holes it encloses, and learn their
references from sidewalk pixels only; the refined masks are always the focus or sidewalk mask minus rejections.

The Sobel texture check (sobel_texture.py) and the colour difference check (color_difference.py) are on by default.
The earlier GMM colour check is retired (gmm_color_RETIRED.py) and is not used.

  python3 sidewalk_system.py --source walk.mov        # JSON lines on stdout
  python3 sidewalk_system.py --source 0               # webcam
  python3 sidewalk_system.py --source walk.mov --show           # show the debug panel
  python3 sidewalk_system.py --source walk.mov --debug-video panel.mp4
  python3 sidewalk_system.py --source walk.mov --no-sobel --no-color   # segmentation only
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

import color_difference
from color_difference import ColorDifference
from sobel_texture import THRESHOLD, WINDOW, SobelTexture, debug_panel

MODEL = "tobiasc/segformer-b0-finetuned-segments-sidewalk"
SIDEWALK = ("flat-sidewalk", "flat-cyclinglane", "flat-parkingdriveway")
MODEL_IN = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def add_color_args(ap):
    """Command-line options for the colour check, shared with simulate_sidewalk.py."""
    ap.add_argument("--color", action=argparse.BooleanOptionalAction, default=True,
                    help="run the colour difference check")
    ap.add_argument("--color-window", type=int, default=color_difference.WINDOW, help="colour blur size in pixels, odd")
    ap.add_argument("--color-threshold", type=float, default=color_difference.THRESHOLD,
                    help="colour distance D above which a pixel is rejected")


def color_check(args):
    """The colour difference check set on the command line, or None when it is off."""
    return ColorDifference(args.color_window, args.color_threshold) if args.color else None


class SidewalkNavigator:
    def __init__(self, device=None, sobel=None, color=None):
        self.sobel = sobel  # optional SobelTexture; None runs segmentation only
        self.color = color  # optional ColorDifference
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(self.device).eval()
        names = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.sidewalk_ids = [names[n] for n in SIDEWALK]
        other_ids = [i for i in range(self.model.config.num_labels) if i not in self.sidewalk_ids]
        self._sidewalk_idx = torch.tensor(self.sidewalk_ids, device=self.device)
        self._other_idx = torch.tensor(other_ids, device=self.device)

    def _sync(self):
        """Waits for queued GPU work, so stage timings are not shifted into whichever stage reads results back."""
        if self.device == "mps":
            torch.mps.synchronize()
        elif str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _preprocess(self, frame, size=(MODEL_IN, MODEL_IN)):
        """BGR frame to a normalised RGB tensor of size (width, height) on the device."""
        x = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        x = (x.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(self.device)

    def sidewalk_margin(self, logits, size):
        """Best SIDEWALK logit minus best other logit, bilinearly upsampled to size (height, width). Positive where a
        sidewalk class has the top score."""
        margin = logits[:, self._sidewalk_idx].amax(1, keepdim=True) - logits[:, self._other_idx].amax(1, keepdim=True)
        return torch.nn.functional.interpolate(margin, size=size, mode="bilinear", align_corners=False)[0, 0]

    def segment(self, frame, timing=None):
        """Boolean sidewalk mask at frame resolution (the margin method). timing, if given, receives preprocess_ms,
        inference_ms and postprocess_ms."""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        x = self._preprocess(frame)
        if timing is not None:
            self._sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(pixel_values=x).logits
            if timing is not None:
                self._sync()
            t2 = time.perf_counter()
            mask = self.sidewalk_margin(logits, (h, w)) > 0
        out = mask.cpu().numpy()
        if timing is not None:
            t3 = time.perf_counter()
            timing.update({"preprocess_ms": 1000 * (t1 - t0), "inference_ms": 1000 * (t2 - t1),
                           "postprocess_ms": 1000 * (t3 - t2)})
        return out

    @staticmethod
    def _largest_region(label):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(label.astype(np.uint8), connectivity=4)
        if n <= 1:
            return np.zeros_like(label)
        return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    def process(self, frame):
        """frame: BGR image. Returns the sidewalk label mask, the focus area mask and the latency, plus the
        SobelTexture.verify and ColorDifference.verify results for the stages that are on. The refined masks leave out
        what any stage rejected. latency_ms includes both stages; timing_ms splits it by stage."""
        t0 = time.perf_counter()
        timing = {}
        sidewalk = self.segment(frame, timing)
        t = time.perf_counter()
        focus = self._largest_region(sidewalk)
        out = {"sidewalk": sidewalk, "focus": focus}
        timing["masks_ms"] = 1000 * (time.perf_counter() - t)
        if self.sobel:
            t = time.perf_counter()
            out.update(self.sobel.verify(frame, sidewalk, focus))
            timing["sobel_ms"] = 1000 * (time.perf_counter() - t)
        if self.color:
            t = time.perf_counter()
            out.update(self.color.verify(frame, sidewalk, focus))
            timing["color_ms"] = 1000 * (time.perf_counter() - t)
            t = time.perf_counter()
            rejected = out["color_anomaly"] | out["anomaly"] if self.sobel else out["color_anomaly"]
            out.update({"refined_sidewalk": sidewalk & ~rejected, "refined_focus": focus & ~rejected})
            timing["refine_ms"] = 1000 * (time.perf_counter() - t)
        out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        out["timing_ms"] = {k: round(v, 2) for k, v in timing.items()}  # per stage, measured around each call
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file or camera index")
    ap.add_argument("--device")
    ap.add_argument("--sobel", action=argparse.BooleanOptionalAction, default=True, help="run the Sobel texture check")
    ap.add_argument("--sobel-window", type=int, nargs=2, default=WINDOW, metavar=("W", "H"), help="sliding-window width and height in pixels, odd")
    ap.add_argument("--sobel-threshold", type=float, default=THRESHOLD, help="spreads above normal texture before a pixel is rejected")
    add_color_args(ap)
    ap.add_argument("--show", action="store_true", help="show the debug panel live, q quits (needs --sobel or --color)")
    ap.add_argument("--debug-video", help="write the debug panel to this mp4 (needs --sobel or --color)")
    args = ap.parse_args()
    if (args.show or args.debug_video) and not (args.sobel or args.color):
        ap.error("--show and --debug-video need --sobel or --color")
    nav = SidewalkNavigator(args.device, SobelTexture(args.sobel_window, args.sobel_threshold) if args.sobel else None,
                            color_check(args))
    writer = None
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, round(960 * h / w)), interpolation=cv2.INTER_AREA)
        r = nav.process(frame)
        line = {"frame": i, "focus_frac": round(float(r["focus"].mean()), 3), "latency_ms": r["latency_ms"]}
        if args.sobel:
            line.update({"anomaly_frac": round(float(r["anomaly"].mean()), 3), "sobel_ms": r["sobel_ms"]})
        if args.color:
            line.update({"color_frac": round(float(r["color_anomaly"].mean()), 3),
                         "color_objects": len(r["color_objects"]), "color_ms": r["color_ms"]})
        print(json.dumps(line, separators=(",", ":")), flush=True)
        if args.show or args.debug_video:
            panel = (color_difference.debug_panel if args.color else debug_panel)(frame, r["sidewalk"], r)
            if args.debug_video:
                if writer is None:
                    size = (panel.shape[1], panel.shape[0])
                    writer = cv2.VideoWriter(args.debug_video, cv2.VideoWriter_fourcc(*"avc1"),
                                             cap.get(cv2.CAP_PROP_FPS) or 30, size)
                writer.write(panel)
            if args.show:
                cv2.imshow("debug", panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        i += 1
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    sys.exit(main())
