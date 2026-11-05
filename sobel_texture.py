"""Sobel texture check: flags focus-area pixels whose edge strength is above the normal sidewalk texture for their row.

A second stage after segmentation, not a replacement. The reference is learned per frame from the focus area itself,
so it needs no labels and no state between frames. Input masks are never changed: rejected pixels only leave the
separate refined masks.

Sobel runs on log brightness, so dark ground is not scored lower for being dark. Texture is averaged over focus pixels
only, so kerbs and cars do not leak into windows at the focus edge. Normal texture is a smooth curve over image rows:
it follows the change with distance, but a fault running across the path cannot set its own row's normal. Only texture
stronger than normal is flagged.

Tuning:
  WINDOW     Wide and short, because perspective flattens ground features. Fits cracks across the path and may dilute
             cracks running along it, which are untested. Sized for 960x540 frames.
  THRESHOLD  Lower flags more. On clean walk A samples 2.3 flagged 3.0% of labelled sidewalk, 2.45 2.4%, 2.6 2.0%. Long
             thin false flags are the main remaining cost.
"""
import time

import cv2
import numpy as np

WINDOW = (31, 5)  # sliding-window width and height in pixels, odd
THRESHOLD = 2.45  # NOT MATCH when the local log texture is more than this many spreads above normal for its row
MIN_REFERENCE_PX = 500  # with fewer clean sample pixels the reference is not trusted and nothing is flagged
LOG_SCALE = 46  # brings log brightness back to about the 0-255 range
EDGE_ERODE = 5  # focus-edge pixels left out of the window average, as the edge itself is a strong gradient
REFERENCE_ERODE = 15  # the reference sample stays this deep inside the focus area
BAND_ROWS = 20  # rows per reference band
MIN_BAND_PX = 150  # bands with fewer sample pixels are skipped
CURVE_DEGREE = 2  # low enough that one band cannot bend the curve towards itself
MIN_SPREAD = 0.05  # floor on the spread, in log texture units
MIN_WEIGHT = 0.02  # floor on the share of a window inside the focus area

KEPT_COLOR = (0, 200, 0)
REJECTED_COLOR = (255, 0, 255)


def gradient_magnitude(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    log = (np.log1p(gray.astype(np.float32)) * LOG_SCALE).astype(np.float32)
    gx = cv2.Sobel(log, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


class SobelTexture:
    def __init__(self, window=WINDOW, threshold=THRESHOLD):
        if any(size < 3 or size % 2 == 0 for size in window):
            raise ValueError(f"window sizes must be odd and at least 3, got {window}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.window, self.threshold = tuple(window), threshold

    def _texture(self, frame, focus):
        """Log of the windowed edge strength, averaged over focus pixels only."""
        inside = cv2.erode(focus.astype(np.uint8), np.ones((EDGE_ERODE, EDGE_ERODE), np.uint8)).astype(np.float32)
        weight = np.maximum(cv2.blur(inside, self.window), MIN_WEIGHT)
        return np.log1p(cv2.blur(gradient_magnitude(frame) * inside, self.window) / weight)

    @staticmethod
    def _reference(texture, focus, top):
        """Normal texture and spread for each row, from bands deep inside the focus area. top is the first row, so
        bands stay on the same grid wherever the focus area sits."""
        core = cv2.erode(focus.astype(np.uint8), np.ones((REFERENCE_ERODE, REFERENCE_ERODE), np.uint8)).astype(bool)
        rows = np.nonzero(core)[0] + top
        sample = texture[core]
        if sample.size < MIN_REFERENCE_PX:
            return None
        centres, medians, spreads = [], [], []
        for start in range(top // BAND_ROWS * BAND_ROWS, top + len(texture), BAND_ROWS):
            band = sample[(rows >= start) & (rows < start + BAND_ROWS)]
            if band.size >= MIN_BAND_PX:
                median = np.median(band)
                centres.append(start + BAND_ROWS / 2)
                medians.append(median)
                spreads.append(1.4826 * np.median(np.abs(band - median)))  # scaled to match an SD
        if len(centres) < 2:
            return None
        y = np.arange(top, top + len(texture), dtype=np.float32)
        if len(centres) > CURVE_DEGREE + 1:
            y = np.clip(y, centres[0], centres[-1])
            normal = np.polyval(np.polyfit(centres, medians, CURVE_DEGREE), y)
            spread = np.polyval(np.polyfit(centres, spreads, CURVE_DEGREE), y)
        else:
            normal, spread = np.interp(y, centres, medians), np.interp(y, centres, spreads)
        return {"normal": normal, "spread": np.maximum(spread, MIN_SPREAD), "bands": len(centres), "px": int(sample.size)}

    def verify(self, frame, sidewalk, focus):
        """frame: BGR image; sidewalk, focus: boolean masks. Returns the anomaly mask and the refined masks."""
        t0 = time.perf_counter()
        anomaly, reference = np.zeros_like(focus), None
        x, y, w, h = cv2.boundingRect(focus.astype(np.uint8))
        if w and h:
            margin_x, margin_y = (size // 2 + 1 for size in self.window)  # windows at the focus edge see real pixels
            x0, y0 = max(x - margin_x, 0), max(y - margin_y, 0)
            x1, y1 = min(x + w + margin_x, frame.shape[1]), min(y + h + margin_y, frame.shape[0])
            box = (slice(y0, y1), slice(x0, x1))
            texture = self._texture(frame[box], focus[box])
            reference = self._reference(texture, focus[box], y0)
            if reference is not None:
                score = (texture - reference["normal"][:, None]) / reference["spread"][:, None]
                anomaly[box] = focus[box] & (score > self.threshold)
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
    stats = f"{ref['bands']} reference bands" if ref else "no reference"
    top = np.hstack([_tile(paint(sidewalk, KEPT_COLOR), "original sidewalk", size),
                     _tile(paint(anomaly, REJECTED_COLOR), f"Sobel anomalies ({stats})", size)])
    bottom = np.hstack([_tile(paint(refined, KEPT_COLOR), "refined sidewalk", size),
                        _tile(highlight, "kept green, rejected magenta", size)])
    return np.vstack([top, bottom])
