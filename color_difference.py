"""Colour difference check: flags focus-area pixels whose Lab a, b colour is far from the frame's median sidewalk colour.

The colour stage of the full model, after segmentation and alongside the Sobel texture check. It replaced the GMM colour
check (now gmm_color_RETIRED.py). The reference is one robust colour per frame: the median of a and b over the focus
area, with the spread of each channel estimated from its median absolute deviation (MAD). Each pixel gets the score

    D = sqrt(((a - med_a) / (sigma_a + EPS))^2 + ((b - med_b) / (sigma_b + EPS))^2),   sigma = 1.4826 * MAD

and is flagged when D > threshold. L is left out, so only colour differences are found: potholes, manhole covers and grey
objects are not. Input masks are never changed, and nothing is kept between frames. Holes in the focus area
(non-sidewalk regions it fully encloses) are scored too, but only sidewalk-labelled pixels feed the reference.

Everything stays uint8: the median and MAD come from 256-bin histograms, and D^2 is read from a lookup table per
channel, so the cost is a colour conversion, a blur and a few table lookups.

Tuning:
  WINDOW         Size of the box blur applied to a, b before scoring; 1 scores raw pixels. Larger suppresses JPEG chroma
                 noise and seams but dilutes thin objects. On all 600 frames of walk A, 3 beat 1, 5, 7, 9 and 11.
                 Sized for 960x540 frames.
  THRESHOLD      Lower flags more. Read together with EPS: the cut-off is THRESHOLD * (sigma + EPS) Lab units per channel.
  EPS            Added to each sigma in Lab units. Pavement a often has a MAD of a single step (sigma ~1.5), so with a
                 small EPS any slight tint flags; EPS 4 makes the cut-off about 3.5 * (1.5 + 4) ~ 19 units in a, close
                 to a fixed colour distance that widens on varied pavement. Walk A, all 600 frames, window 3: EPS 1 at
                 T 6 flagged 0.38% of clean sidewalk for 86.6% catch of cone and racks; EPS 4 at T 3.5 flagged 0.09% for
                 86.0%, with the best mask IoU of any setting tried (the GMM: 0.43% for 91.1%).
  MIN_OBJECT_PX  Smaller flagged blobs are dropped. 50 kept more of the cone than 100 at little extra noise; 25 added
                 objects without catching more.
"""
import time

import cv2
import numpy as np

from sobel_texture import KEPT_COLOR, REJECTED_COLOR, _tile, fill_holes

WINDOW = 3  # box-blur size in pixels, odd; 1 means no blur
THRESHOLD = 3.5  # flagged when D is above this
EPS = 4.0  # added to each channel's sigma, Lab units
MAD_TO_SD = 1.4826  # MAD of a normal distribution times this is its SD
MIN_REFERENCE_PX = 500  # with fewer reference pixels the median is not trusted and nothing is flagged
MIN_OBJECT_PX = 50  # flagged blobs smaller than this are dropped

COLOR_REJECTED_COLOR = (0, 255, 255)

_LEVELS = np.arange(256, dtype=np.float64)


def _hist_median(counts, values):
    """Median of a sample given as counts per value (values need not be sorted)."""
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(counts[order])
    n = cdf[-1]
    lo = values[order][np.searchsorted(cdf, (n - 1) // 2, side="right")]
    hi = values[order][np.searchsorted(cdf, n // 2, side="right")]
    return (lo + hi) / 2


def robust_stats(channel):
    """Median and MAD-based SD of a uint8 sample."""
    counts = np.bincount(channel, minlength=256)
    median = _hist_median(counts, _LEVELS)
    mad = _hist_median(counts, np.abs(_LEVELS - median))
    return median, MAD_TO_SD * mad


class ColorDifference:
    def __init__(self, window=WINDOW, threshold=THRESHOLD, eps=EPS, min_object_px=MIN_OBJECT_PX):
        if window < 1 or window % 2 == 0:
            raise ValueError(f"window must be odd and at least 1, got {window}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.window, self.threshold, self.eps, self.min_object_px = window, threshold, eps, min_object_px

    def _reference(self, ab, labelled):
        """Median colour and SDs of the local colours deep inside the sidewalk-labelled focus area. The erosion matches
        the blur, so no reference window reaches a non-sidewalk pixel."""
        core = labelled.astype(np.uint8)
        if self.window > 1:
            core = cv2.erode(core, np.ones((self.window, self.window), np.uint8))
        core = core.astype(bool)
        px = int(core.sum())
        if px < MIN_REFERENCE_PX:
            return None
        (ma, sa), (mb, sb) = robust_stats(ab[:, :, 0][core]), robust_stats(ab[:, :, 1][core])
        return {"median": (ma, mb), "sigma": (sa, sb), "px": px}

    def verify(self, frame, sidewalk, focus):
        """frame: BGR image; sidewalk, focus: boolean masks. Returns the colour anomaly mask and its object boxes."""
        t0 = time.perf_counter()
        anomaly, objects, reference = np.zeros_like(focus), [], None
        x, y, w, h = cv2.boundingRect(focus.astype(np.uint8))
        if w and h:
            margin = self.window // 2 + 1  # windows at the focus edge see real pixels, not a padded border
            x0, y0 = max(x - margin, 0), max(y - margin, 0)
            x1, y1 = min(x + w + margin, frame.shape[1]), min(y + h + margin, frame.shape[0])
            box = (slice(y0, y1), slice(x0, x1))
            ab = cv2.cvtColor(frame[box], cv2.COLOR_BGR2LAB)[:, :, 1:]
            if self.window > 1:
                ab = cv2.blur(ab, (self.window, self.window))
            reference = self._reference(ab, focus[box] & sidewalk[box])
            if reference is not None:
                area = fill_holes(focus[box])  # holes are scored, but never feed the reference
                (ma, mb), (sa, sb) = reference["median"], reference["sigma"]
                lut_a = (((_LEVELS - ma) / (sa + self.eps)) ** 2).astype(np.float32)
                lut_b = (((_LEVELS - mb) / (sb + self.eps)) ** 2).astype(np.float32)
                d2 = lut_a[ab[:, :, 0]] + lut_b[ab[:, :, 1]]
                flagged = (area & (d2 > self.threshold ** 2)).astype(np.uint8)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(flagged, connectivity=8)
                big = 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= self.min_object_px)
                keep = np.zeros(n, bool)
                keep[big] = True
                anomaly[box] = keep[lab]
                objects = [(int(stats[i, 0]) + x0, int(stats[i, 1]) + y0, int(stats[i, 2]), int(stats[i, 3]))
                           for i in big]  # x, y, w, h in frame coordinates
        return {"color_anomaly": anomaly, "color_objects": objects, "color_reference": reference,
                "color_ms": round(1000 * (time.perf_counter() - t0), 2)}


def debug_panel(frame, sidewalk, result):
    """2x2 comparison, same size as frame: original mask, colour anomalies, refined mask, frame with rejections
    highlighted. Sobel rejections are shown too when that stage ran."""
    h, w = frame.shape[:2]
    size = (w // 2, h // 2)
    anomaly, refined, sobel = result["color_anomaly"], result["refined_sidewalk"], result.get("anomaly")

    def paint(mask, color):
        out = np.zeros_like(frame)
        out[mask] = color
        return out

    highlight = frame.copy()
    highlight[refined] = KEPT_COLOR
    highlight = cv2.addWeighted(frame, 0.65, highlight, 0.35, 0)
    if sobel is not None:
        highlight[sobel] = REJECTED_COLOR
    highlight[anomaly] = COLOR_REJECTED_COLOR
    for x, y, bw, bh in result["color_objects"]:
        cv2.rectangle(highlight, (x, y), (x + bw, y + bh), COLOR_REJECTED_COLOR, 1)

    ref = result["color_reference"]
    stats = "no reference" if not ref else "median a {:.0f}, b {:.0f}".format(*ref["median"])
    title = "kept green, colour yellow" + (", Sobel magenta" if sobel is not None else "")
    top = np.hstack([_tile(paint(sidewalk, KEPT_COLOR), "original sidewalk", size),
                     _tile(paint(anomaly, COLOR_REJECTED_COLOR), f"colour anomalies ({stats})", size)])
    bottom = np.hstack([_tile(paint(refined, KEPT_COLOR), "refined sidewalk", size), _tile(highlight, title, size)])
    return np.vstack([top, bottom])
