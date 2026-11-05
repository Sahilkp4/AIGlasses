"""Sobel texture check: flags focus-area pixels whose edge strength differs from the frame's normal sidewalk texture.

A second stage after segmentation, not a replacement. The reference is learned per frame from the focus area itself,
so it needs no labels and no state between frames. Input masks are never changed: rejected pixels only leave the
separate refined masks.

Tuning:
  WINDOW     Larger averages over more pavement: fewer flags on seams and shadows, but small cracks get diluted, and
             the reference sample shrinks because it is eroded by the same amount. Sized for 960x540 frames.
  THRESHOLD  Lower flags more. On walk A samples 2.0 flagged about 12% of the focus area, 2.5 about 6%, 3.0 about 3%.
"""
import time

import cv2
import numpy as np

WINDOW = 15  # sliding-window size in pixels, odd
THRESHOLD = 2.5  # NOT MATCH when the local mean is more than this many reference SDs from the reference mean
MIN_REFERENCE_PX = 500  # with fewer clean sample pixels the reference is not trusted and nothing is flagged

KEPT_COLOR = (0, 200, 0)
REJECTED_COLOR = (255, 0, 255)


def gradient_magnitude(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


class SobelTexture:
    def __init__(self, window=WINDOW, threshold=THRESHOLD):
        if window < 3 or window % 2 == 0:
            raise ValueError(f"window must be odd and at least 3, got {window}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.window, self.threshold = window, threshold

    def _reference(self, local, focus):
        """Mean and SD of the local means deep inside the focus area, leaving out what they would reject themselves."""
        core = cv2.erode(focus.astype(np.uint8), np.ones((self.window, self.window), np.uint8)).astype(bool)
        sample = local[core]
        if sample.size < MIN_REFERENCE_PX:
            return None
        clean = sample[np.abs(sample - sample.mean()) <= self.threshold * sample.std()]
        if clean.size < MIN_REFERENCE_PX:
            return None
        return {"mean": float(clean.mean()), "sd": float(clean.std()), "px": int(clean.size)}

    def verify(self, frame, sidewalk, focus):
        """frame: BGR image; sidewalk, focus: boolean masks. Returns the anomaly mask and the refined masks."""
        t0 = time.perf_counter()
        anomaly, reference = np.zeros_like(focus), None
        x, y, w, h = cv2.boundingRect(focus.astype(np.uint8))
        if w and h:
            margin = self.window // 2 + 1  # windows at the focus edge see real pixels, not a padded border
            x0, y0 = max(x - margin, 0), max(y - margin, 0)
            x1, y1 = min(x + w + margin, frame.shape[1]), min(y + h + margin, frame.shape[0])
            box = (slice(y0, y1), slice(x0, x1))
            local = cv2.blur(gradient_magnitude(frame[box]), (self.window, self.window))
            reference = self._reference(local, focus[box])
            if reference is not None:
                differs = np.abs(local - reference["mean"]) > self.threshold * reference["sd"]
                anomaly[box] = focus[box] & differs
        return {"anomaly": anomaly, "refined_sidewalk": sidewalk & ~anomaly, "refined_focus": focus & ~anomaly,
                "reference": reference, "sobel_ms": round(1000 * (time.perf_counter() - t0), 2)}


def _tile(img, title, size):
    out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def debug_panel(frame, sidewalk, result):
    """2x2 comparison, same size as frame: original mask, anomalies, refined mask, frame with anomalies highlighted."""
    h, w = frame.shape[:2]
    size = (w // 2, h // 2)
    anomaly, refined = result["anomaly"], result["refined_sidewalk"]

    def paint(mask, color):
        out = np.zeros_like(frame)
        out[mask] = color
        return out

    highlight = frame.copy()
    highlight[refined] = KEPT_COLOR
    highlight = cv2.addWeighted(frame, 0.65, highlight, 0.35, 0)
    highlight[anomaly] = REJECTED_COLOR

    ref = result["reference"]
    stats = f"ref {ref['mean']:.0f} +/- {ref['sd']:.0f}" if ref else "no reference"
    top = np.hstack([_tile(paint(sidewalk, KEPT_COLOR), "original sidewalk", size),
                     _tile(paint(anomaly, REJECTED_COLOR), f"Sobel anomalies ({stats})", size)])
    bottom = np.hstack([_tile(paint(refined, KEPT_COLOR), "refined sidewalk", size),
                        _tile(highlight, "kept green, rejected magenta", size)])
    return np.vstack([top, bottom])
