"""RETIRED (November 8, 2026): replaced in the full model by the median/MAD colour difference check in
color_difference.py, which matched its mask accuracy at about 1/20 of the cost (see Test Results/Color Difference Test
Summary.md). Nothing in the pipeline imports this module; it is kept only so eval_gmm_RETIRED.py can reproduce results.

GMM colour check: flags focus-area pixels whose Lab a, b colour fits none of the frame's normal sidewalk colours.

A second stage after segmentation, not a replacement. The GMM is learned per frame from the focus area itself, so it
needs no labels and no state between frames. Input masks are never changed. L is left out, so brightness alone cannot
flag a pixel, and only colour differences are found: potholes, manhole covers and grey objects are not. Holes in the
focus area (non-sidewalk regions it fully encloses) are scored too, but only sidewalk-labelled pixels feed the GMM.

Tuning:
  WINDOW         Larger averages over more pavement: fewer flags on seams, but thin objects get diluted, and the
                 reference sample shrinks because it is eroded by the same amount. Sized for 960x540 frames.
  THRESHOLD      Lower flags more. On walk A samples 3 flagged 0.9% of clean sidewalk, 4 about 0.3%, 5 about 0.1%.
  MIN_WEIGHT     An object larger than this share of the sample can become a component of its own and count as normal.
  MIN_OBJECT_PX  Smaller flagged blobs are dropped.
"""
import time

import cv2
import numpy as np

from sobel_texture import KEPT_COLOR, REJECTED_COLOR, _tile, fill_holes

WINDOW = 9  # sliding-window size in pixels, odd
THRESHOLD = 4.0  # NOT MATCH when the local colour is more than this many SDs from every kept component
COMPONENTS = 4  # lets sunlit pavement, cool shadow, paint and tactile strips each count as normal
SAMPLE_PX = 2000  # most pixels used to fit the GMM
MIN_REFERENCE_PX = 500  # with fewer clean sample pixels the GMM is not trusted and nothing is flagged
MIN_WEIGHT = 0.05  # lighter components are ignored, so a small object cannot be learned as sidewalk
MIN_SD = 1.0  # added to every component's SD in Lab units, so uniform pavement does not flag on JPEG chroma noise
MIN_OBJECT_PX = 100  # flagged blobs smaller than this are dropped

COLOR_REJECTED_COLOR = (0, 255, 255)
BOTH_REJECTED_COLOR = (255, 255, 0)


def ab_features(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float32)


def distance(x, means, precisions):
    """Mahalanobis distance from each row of x (..., 2) to the nearest component."""
    d2 = [((x - m) @ p * (x - m)).sum(-1) for m, p in zip(means, precisions)]
    return np.sqrt(np.min(d2, axis=0))


class GmmColor:
    def __init__(self, window=WINDOW, threshold=THRESHOLD):
        if window < 3 or window % 2 == 0:
            raise ValueError(f"window must be odd and at least 3, got {window}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.window, self.threshold = window, threshold

    @staticmethod
    def _fit(sample):
        """Means and inverse covariances of the components heavy enough to be sidewalk."""
        em = cv2.ml.EM_create()
        em.setClustersNumber(COMPONENTS)
        em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_GENERIC)
        em.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 20, 1e-2))
        em.trainEM(sample)
        keep = np.flatnonzero(em.getWeights()[0] >= MIN_WEIGHT)
        covs = em.getCovs()
        return (em.getMeans()[keep].astype(np.float32),
                np.array([np.linalg.inv(covs[i] + MIN_SD ** 2 * np.eye(2)) for i in keep], np.float32))

    def _reference(self, local, labelled):
        """GMM of the local colours deep inside the sidewalk-labelled focus area, refitted without what it would reject
        itself. The erosion matches the blur, so no reference window reaches a non-sidewalk pixel."""
        core = cv2.erode(labelled.astype(np.uint8), np.ones((self.window, self.window), np.uint8)).astype(bool)
        sample = local[core]
        if len(sample) < MIN_REFERENCE_PX:
            return None
        cv2.setRNGSeed(0)  # EM starts from a random split: the same frame always gets the same GMM
        rng = np.random.default_rng(0)
        sample = sample[rng.permutation(len(sample))[:SAMPLE_PX]]
        clean = sample[distance(sample, *self._fit(sample)) <= self.threshold]
        if len(clean) < MIN_REFERENCE_PX:
            return None
        means, precisions = self._fit(clean)
        return {"means": means, "precisions": precisions, "px": len(clean)}

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
            local = cv2.blur(ab_features(frame[box]), (self.window, self.window))
            reference = self._reference(local, focus[box] & sidewalk[box])
            if reference is not None:
                differs = distance(local, reference["means"], reference["precisions"]) > self.threshold
                anomaly[box] = fill_holes(focus[box]) & differs  # holes are scored, but never feed the reference
                n, lab, stats, _ = cv2.connectedComponentsWithStats(anomaly.astype(np.uint8), connectivity=8)
                big = 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= MIN_OBJECT_PX)
                anomaly = np.isin(lab, big)
                objects = [tuple(int(v) for v in stats[i, :4]) for i in big]  # x, y, w, h
        return {"color_anomaly": anomaly, "color_objects": objects, "color_reference": reference,
                "gmm_ms": round(1000 * (time.perf_counter() - t0), 2)}


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
    stats = "no reference" if not ref else f"{len(ref['means'])} components" if "means" in ref else "median reference"
    title = "kept green, colour yellow" + (", Sobel magenta" if sobel is not None else "")
    top = np.hstack([_tile(paint(sidewalk, KEPT_COLOR), "original sidewalk", size),
                     _tile(paint(anomaly, COLOR_REJECTED_COLOR), f"colour anomalies ({stats})", size)])
    bottom = np.hstack([_tile(paint(refined, KEPT_COLOR), "refined sidewalk", size), _tile(highlight, title, size)])
    return np.vstack([top, bottom])
