"""Sidewalk focus area: segments each frame and marks the largest connected region labelled sidewalk.

Segmentation: tobiasc/segformer-b0-finetuned-segments-sidewalk. A pixel is sidewalk when its top class is one of SIDEWALK.

  python3 sidewalk_system.py --source walk.mov        # JSON lines on stdout
  python3 sidewalk_system.py --source 0               # webcam
  python3 sidewalk_system.py --source walk.mov --sobel --show   # add the Sobel texture check and its debug panel
  python3 sidewalk_system.py --source walk.mov --sobel --debug-video panel.mp4
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from sobel_texture import THRESHOLD, WINDOW, SobelTexture, debug_panel

MODEL = "tobiasc/segformer-b0-finetuned-segments-sidewalk"
SIDEWALK = ("flat-sidewalk", "flat-cyclinglane", "flat-parkingdriveway")
MODEL_IN = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class SidewalkNavigator:
    def __init__(self, device=None, sobel=None):
        self.sobel = sobel  # optional SobelTexture; None runs segmentation only
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(self.device).eval()
        names = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.sidewalk_ids = [names[n] for n in SIDEWALK]

    def segment(self, frame):
        """Class map at frame resolution."""
        h, w = frame.shape[:2]
        x = cv2.resize(frame, (MODEL_IN, MODEL_IN), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        x = (x.astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(self.device)
        with torch.no_grad():
            cls = self.model(pixel_values=x).logits.argmax(1, keepdim=True)
            cls = torch.nn.functional.interpolate(cls.float(), size=(h, w), mode="nearest")
        return cls[0, 0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def _largest_region(label):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(label.astype(np.uint8), connectivity=4)
        if n <= 1:
            return np.zeros_like(label)
        return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    def process(self, frame):
        """frame: BGR image. Returns the sidewalk label mask, the focus area mask and the latency, plus the
        SobelTexture.verify results when Sobel is on. latency_ms includes Sobel."""
        t0 = time.perf_counter()
        sidewalk = np.isin(self.segment(frame), self.sidewalk_ids)
        focus = self._largest_region(sidewalk)
        out = {"sidewalk": sidewalk, "focus": focus}
        if self.sobel:
            out.update(self.sobel.verify(frame, sidewalk, focus))
        out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file or camera index")
    ap.add_argument("--device")
    ap.add_argument("--sobel", action="store_true", help="add the Sobel texture check")
    ap.add_argument("--sobel-window", type=int, default=WINDOW, help="sliding-window size in pixels, odd")
    ap.add_argument("--sobel-threshold", type=float, default=THRESHOLD, help="reference SDs before a pixel is rejected")
    ap.add_argument("--show", action="store_true", help="show the Sobel debug panel live, q quits (needs --sobel)")
    ap.add_argument("--debug-video", help="write the Sobel debug panel to this mp4 (needs --sobel)")
    args = ap.parse_args()
    if (args.show or args.debug_video) and not args.sobel:
        ap.error("--show and --debug-video need --sobel")
    nav = SidewalkNavigator(args.device, SobelTexture(args.sobel_window, args.sobel_threshold) if args.sobel else None)
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
        print(json.dumps(line, separators=(",", ":")), flush=True)
        if args.show or args.debug_video:
            panel = debug_panel(frame, r["sidewalk"], r)
            if args.debug_video:
                if writer is None:
                    size = (panel.shape[1], panel.shape[0])
                    writer = cv2.VideoWriter(args.debug_video, cv2.VideoWriter_fourcc(*"avc1"),
                                             cap.get(cv2.CAP_PROP_FPS) or 30, size)
                writer.write(panel)
            if args.show:
                cv2.imshow("sobel", panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        i += 1
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    sys.exit(main())
