"""Shared helpers for the colour and texture evaluations on walk A: frame loading and hand-outlined targets.

Targets are in gmm_targets.json (named after the first stage they were outlined for, the retired GMM check): polygons at
960x540 for the cone, pothole, bike racks and a manhole cover.
"""
import cv2
import numpy as np

from eval_sidewalk import EVAL_H, EVAL_W, SRC

TARGETS = "gmm_targets.json"
NOT_CLEAN = (range(125, 141), range(440, 481))  # the labels call the cone and racks sidewalk here
RACK_MIN_B = 150  # OpenCV 8-bit Lab: rack outlines are loose, so only clearly yellow pixels count as rack


def load(i):
    return cv2.resize(cv2.imread(f"{SRC}/{i:06d}.jpg", cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))


def fill(polygons):
    mask = np.zeros((EVAL_H, EVAL_W), np.uint8)
    cv2.fillPoly(mask, [np.array(p, np.int32) for p in polygons], 1)
    return mask.astype(bool)


def target_masks(targets, i, frame):
    """Masks of the targets outlined on frame i. The cone stands in the pothole, so it is cut out of it."""
    out = {name: fill(frames[str(i)]) for name, frames in targets.items() if str(i) in frames}
    if "rack" in out:
        out["rack"] &= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 2] >= RACK_MIN_B
    if "pothole" in out and "cone" in out:
        out["pothole"] &= ~out["cone"]
    return out
