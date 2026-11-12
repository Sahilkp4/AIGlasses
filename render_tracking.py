"""Review overlay for feature tracking: runs the full pipeline on a range of dataset_flip frames and draws, per frame,
the previous frame (left) and the current frame (right) side by side.

  left   previous frame with the tracks as they stood after it (yellow tentative, red guaranteed, dashed coasting)
  right  current frame: Sobel flags magenta, colour flags yellow (tinted); each match drawn from the track's old centre
         (white dot) to the box it matched, green when accepted and red when lost as an outlier; a cyan cross at
         the predicted centre under the voted motion; the voted motion as cyan arrows down the left edge (expected
         displacement at that row); and the tracks after this frame

  python3 render_tracking.py                      # frames 1-23 -> runs/tracking_review/
Writes frame_NNNN.png per frame plus review_F-L.mp4 (H.264, QuickTime) and review_F-L.webm (VP8, plays in VS Code and
browsers, which lack H.264). Videos run at VIDEO_FPS with each review frame held for 1 / --fps seconds, since some
players mishandle a 1 FPS stream.
  python3 render_tracking.py --first 1 --last 60
  python3 render_tracking.py --angles 60 105      # side by side: current frame under each DOWN_ANGLE_DEG
                                                  # -> runs/tracking_review_angles/
  python3 render_tracking.py --arrows             # previous boxes with an arrow to the box each matched
                                                  # -> runs/tracking_arrows/
  python3 render_tracking.py --current --last 600 --fps 2   # only this frame's tracked boxes, at the camera rate
                                                  # -> runs/tracking_current/ (JPEG stills)
  python3 render_tracking.py --status             # every tracked box by status, coasting and discarded included
                                                  # -> runs/tracking_status/ (JPEG stills)

--arrows view: left is the previous frame with its boxes (white, dashed if coasting). Right is the current frame with
the same previous boxes in white, this frame's boxes (yellow tentative, red guaranteed, labelled id:streak), and an
arrow from each previous box to the current box it was paired with, drawn between the middles of their bottom edges
(the tracker measures motion on box edges, not centres): green accepted, red rejected as an outlier. A thin cyan box
is where the tracker predicted each paired box would be.
A previous box with no pair gets a grey cross: tentative ones are forgotten, guaranteed ones coast (dashed next frame).
Orange boxes are guaranteed obstacles discarded by the size check (no growth over their last 3 frames, on sidewalk).
"""
import argparse
import glob
import os

import cv2
import numpy as np

from color_difference import ColorDifference
from eval_sidewalk import EVAL_H, EVAL_W
import feature_tracking
from feature_tracking import (N_COLUMNS, N_REGIONS, FeatureTracker, cell_text, column_lefts, draw_tracks,
                              motion_text, region_tops)
from sidewalk_system import SidewalkNavigator
from sobel_texture import SobelTexture

SRC = "data/dataset_flip"
OUT = "runs/tracking_review"
VIDEO_FPS = 10
OK_COLOR, LOST_COLOR, PRED_COLOR = (0, 220, 0), (0, 0, 255), (255, 255, 0)
PREV_COLOR, NOMATCH_COLOR, DISCARD_COLOR = (255, 255, 255), (160, 160, 160), (0, 140, 255)
LEGEND_H = 28


def text(img, s, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def current_view(frame, r):
    out = frame.copy()
    tint = out.copy()
    tint[r["anomaly"]] = (255, 0, 255)
    tint[r["color_anomaly"]] = (0, 255, 255)
    out = cv2.addWeighted(out, 0.5, tint, 0.5, 0)
    H, W = out.shape[:2]
    top = cv2.boundingRect(r["focus"].astype(np.uint8))[1] if r["focus"].any() else 0
    if r["motion"] is not None:  # expected displacement of a ground object at each row, under each column's cell
        m = np.array(r["motion"])
        for c, left in enumerate(column_lefts(W)):
            x = int(left) + 30
            for y in range(top + 20, H, 40):
                region = FeatureTracker._region(y, H)
                d = m[region * N_COLUMNS + c] * FeatureTracker._region_scale(y, region, top, H, 1)
                cv2.arrowedLine(out, (x, y), (int(x + d[0]), int(y + d[1])), PRED_COLOR, 2, tipLength=0.2)
    draw_band_lines(out, r)
    draw_tracks(out, r)
    for mt in r["track_matches"]:
        a, b, p = (tuple(int(v) for v in mt[k]) for k in ("from", "to", "predicted"))
        cv2.arrowedLine(out, a, b, OK_COLOR if mt["ok"] else LOST_COLOR, 2, tipLength=0.15)
        cv2.circle(out, a, 3, (255, 255, 255), -1)
        cv2.drawMarker(out, p, PRED_COLOR, cv2.MARKER_CROSS, 10, 1)
    return out


def draw_band_lines(img, r=None):
    """Thin grey lines of the tracker's grid (regions R1 at the bottom to R4, columns L C R). With a tracker result r,
    each cell is labelled in its bottom-right corner with its motion: dy at the region's bottom edge, dx when not ~0,
    and v / l / d for vote, last vote or default."""
    H, W = img.shape[:2]
    tops, lefts = region_tops(H), column_lefts(W)
    for k, t in enumerate(tops[:-1]):
        cv2.line(img, (0, int(round(t))), (W, int(round(t))), (150, 150, 150), 1)
    for left in lefts[1:]:
        cv2.line(img, (int(round(left)), 0), (int(round(left)), H), (150, 150, 150), 1)
    for region in range(N_REGIONS):
        bottom = H if region == 0 else int(round(tops[region - 1]))
        for c in range(N_COLUMNS):
            right = int(round(lefts[c + 1])) if c + 1 < N_COLUMNS else W
            label = f"R{region + 1}{'LCR'[c]}"
            if r is not None:
                k = region * N_COLUMNS + c
                label += " " + cell_text(r["motion"][k], r["motion_source"][k])
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
            text(img, label, (right - size - 4, bottom - 5), (190, 190, 190), 0.38)


def _box(img, box, color, dashed=False, width=1):
    x, y, w, h = (int(v) for v in box)
    if dashed:
        feature_tracking._dashed_rect(img, (x, y), (x + w, y + h), color)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, width)


def _bottom_mid(box):
    x, y, w, h = box
    return int(x + w / 2), int(y + h)


def arrows_view(frame, r, prev_tracks):
    """Current frame: previous boxes (white), current boxes (yellow/red) and an arrow per pairing."""
    out = frame.copy()
    tint = out.copy()
    tint[r["anomaly"]] = (255, 0, 255)
    tint[r["color_anomaly"]] = (0, 255, 255)
    out = cv2.addWeighted(out, 0.6, tint, 0.4, 0)
    for t in prev_tracks:
        _box(out, t["box"], PREV_COLOR, t["coasting"])
    for t in r["tracks"]:
        if not t["coasting"]:  # this frame's boxes
            color = feature_tracking.GUARANTEED_COLOR if t["state"] == "guaranteed" else feature_tracking.TENTATIVE_COLOR
            _box(out, t["box"], color, width=2)
            text(out, f"{t['id']}:{t['streak']}", (int(t["box"][0]), max(int(t["box"][1]) - 3, 10)), color, 0.35)
    paired = {m["track"] for m in r["track_matches"]}
    for t in prev_tracks:
        if t["id"] not in paired:
            cv2.drawMarker(out, _bottom_mid(t["box"]), NOMATCH_COLOR, cv2.MARKER_TILTED_CROSS, 9, 2)
    for m in r["track_matches"]:
        fb = m["from_box"]
        dx, dy = m["predicted"][0] - m["from"][0], m["predicted"][1] - m["from"][1]
        _box(out, (fb[0] + dx, fb[1] + dy, fb[2], fb[3]), PRED_COLOR)  # predicted box
        a, b = _bottom_mid(m["from_box"]), _bottom_mid(m["to_box"])
        cv2.arrowedLine(out, a, b, OK_COLOR if m["ok"] else LOST_COLOR, 2, tipLength=0.2)
    for d in r.get("track_discarded", []):  # guaranteed but not growing on sidewalk: given back
        _box(out, d["box"], DISCARD_COLOR, d["coasting"], width=2)
        text(out, f"{d['id']} discarded", (int(d["box"][0]), int(d["box"][1] + d["box"][3]) + 12), DISCARD_COLOR, 0.35)
    strip = np.zeros((LEGEND_H, out.shape[1], 3), np.uint8)  # legend below the frame, so it hides no boxes
    y, x = LEGEND_H - 10, 10
    for label, color in (("previous box", PREV_COLOR), ("current tentative", feature_tracking.TENTATIVE_COLOR),
                         ("current guaranteed", feature_tracking.GUARANTEED_COLOR), ("accepted", OK_COLOR),
                         ("rejected", LOST_COLOR), ("predicted", PRED_COLOR), ("x no pair", NOMATCH_COLOR),
                         ("discarded (no growth, on sidewalk)", DISCARD_COLOR)):
        text(strip, label, (x, y), color, 0.42)
        x += 12 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    return np.vstack([out, strip])


def render_arrows(args, paths, videos):
    """Full model (segmentation, Sobel, colour, tracking): previous frame left, arrows view right."""
    nav = SidewalkNavigator(None, SobelTexture(), ColorDifference(), FeatureTracker())
    prev, prev_tracks = None, []
    for k, path in enumerate(paths):
        n = args.first + k
        frame = cv2.resize(cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))
        r = nav.process(frame, n - 1)
        right = arrows_view(frame, r, prev_tracks)
        if prev is None:
            left = np.zeros_like(frame)
            text(left, "no previous frame", (10, 60))
        else:
            left = prev.copy()
            for t in prev_tracks:
                _box(left, t["box"], PREV_COLOR, t["coasting"])
            text(left, f"frame {n - 1}: its boxes (white, dashed = coasting)", (10, 20))
        print(f"frame {n}: " + header(right, r, n))
        left = np.vstack([left, np.zeros((LEGEND_H, EVAL_W, 3), np.uint8)])
        panel = np.hstack([left, np.full((EVAL_H + LEGEND_H, 6, 3), 255, np.uint8), right])
        cv2.imwrite(f"{args.out}/frame_{n:04d}.png", panel)
        videos.write(panel)
        prev, prev_tracks = frame, r["tracks"]


def render_current(args, paths, videos):
    """Full model; each frame with only its current boxes from the tracker: yellow tentative, red guaranteed, labelled
    id:streak. Previous, coasting and discarded boxes are not drawn."""
    nav = SidewalkNavigator(None, SobelTexture(), ColorDifference(), FeatureTracker())
    counts = {"tentative": 0, "guaranteed": 0}
    for k, path in enumerate(paths):
        n = args.first + k
        frame = cv2.resize(cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))
        r = nav.process(frame, n - 1)
        out = frame.copy()
        current = [t for t in r["tracks"] if not t["coasting"]]
        for t in sorted(current, key=lambda t: t["state"] == "guaranteed"):  # guaranteed drawn on top
            color = feature_tracking.GUARANTEED_COLOR if t["state"] == "guaranteed" else feature_tracking.TENTATIVE_COLOR
            _box(out, t["box"], color, width=2)
            text(out, f"{t['id']}:{t['streak']}", (int(t["box"][0]), max(int(t["box"][1]) - 4, 12)), color, 0.4)
            counts[t["state"]] += 1
        n_g = sum(t["state"] == "guaranteed" for t in current)
        text(out, f"frame {n}   {len(current) - n_g} tentative, {n_g} guaranteed   "
                  f"motion {motion_text(r)}", (10, 20))
        strip = np.zeros((LEGEND_H, EVAL_W, 3), np.uint8)
        text(strip, "current tentative (id:streak)", (10, LEGEND_H - 10), feature_tracking.TENTATIVE_COLOR, 0.42)
        text(strip, "current guaranteed (id:streak)", (260, LEGEND_H - 10), feature_tracking.GUARANTEED_COLOR, 0.42)
        panel = np.vstack([out, strip])
        cv2.imwrite(f"{args.out}/frame_{n:04d}.jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        videos.write(panel)
        if n % 50 == 0:
            print(f"frame {n}/{args.first + len(paths) - 1}", flush=True)
    return counts


def render_status(args, paths, videos):
    """Full model; every box the tracker holds after each frame, by status: yellow tentative, red guaranteed (both
    matched or new this frame), dashed red coasting (guaranteed, unmatched, still cut from the walkable area), orange
    discarded by the size check this frame (given back). Flags are tinted underneath."""
    nav = SidewalkNavigator(None, SobelTexture(), ColorDifference(), FeatureTracker())
    counts = {"tentative": 0, "guaranteed": 0, "coasting": 0, "discarded": 0}
    for k, path in enumerate(paths):
        n = args.first + k
        frame = cv2.resize(cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))
        r = nav.process(frame, n - 1)
        out = frame.copy()
        tint = out.copy()
        tint[r["anomaly"]] = (255, 0, 255)
        tint[r["color_anomaly"]] = (0, 255, 255)
        out = cv2.addWeighted(out, 0.7, tint, 0.3, 0)
        draw_band_lines(out, r)
        frame_counts = dict.fromkeys(counts, 0)
        order = {"tentative": 0, "coasting": 1, "guaranteed": 2}  # guaranteed drawn on top
        for t in sorted(r["tracks"], key=lambda t: order["coasting" if t["coasting"] else t["state"]]):
            status = "coasting" if t["coasting"] else t["state"]
            color = feature_tracking.TENTATIVE_COLOR if status == "tentative" else feature_tracking.GUARANTEED_COLOR
            _box(out, t["box"], color, dashed=t["coasting"], width=2)
            label = f"{t['id']}:{t['streak']}" + (" coast" if t["coasting"] else "") \
                + (" split" if t["id"] in r.get("track_split", []) else "")
            text(out, label, (int(t["box"][0]), max(int(t["box"][1]) - 4, 12)), color, 0.4)
            frame_counts[status] += 1
        for d in r.get("track_discarded", []):
            _box(out, d["box"], DISCARD_COLOR, dashed=d["coasting"], width=2)
            text(out, f"{d['id']} discarded", (int(d["box"][0]), int(d["box"][1] + d["box"][3]) + 12),
                 DISCARD_COLOR, 0.4)
            frame_counts["discarded"] += 1
        for s in counts:
            counts[s] += frame_counts[s]
        v = r["track_vote"]
        text(out, f"frame {n}   " + ", ".join(f"{c} {s}" for s, c in frame_counts.items())
             + f", {len(r.get('track_split', []))} split", (10, 20))
        text(out, f"motion (v vote, l last vote, d default): {motion_text(r)}", (10, 40), scale=0.42)
        text(out, f"vote: {v['support']} of {v['voters']} voters, weight {v.get('weight', 0)}   "
                  f"matches {sum(m['ok'] for m in r['track_matches'])} ok, {r['track_lost']} rejected, "
                  f"{r.get('track_backward', 0)} off-direction", (10, 58), scale=0.42)
        strip = np.zeros((LEGEND_H, EVAL_W, 3), np.uint8)
        x = 10
        for label, color in (("tentative (id:streak)", feature_tracking.TENTATIVE_COLOR),
                             ("guaranteed", feature_tracking.GUARANTEED_COLOR),
                             ("coasting (dashed, still removed)", feature_tracking.GUARANTEED_COLOR),
                             ("discarded (given back)", DISCARD_COLOR), ("Sobel flags", (255, 0, 255)),
                             ("colour flags", (0, 255, 255))):
            text(strip, label, (x, LEGEND_H - 10), color, 0.42)
            x += 16 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        panel = np.vstack([out, strip])
        cv2.imwrite(f"{args.out}/frame_{n:04d}.jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        videos.write(panel)
        print(f"frame {n}: " + ", ".join(f"{c} {s}" for s, c in frame_counts.items())
              + f", split {r.get('track_split', [])}, split votes excluded {v.get('excluded_split', 0)}"
              + f"   motion {motion_text(r)}", flush=True)
        if n % 50 == 0:
            print(f"frame {n}/{args.first + len(paths) - 1}", flush=True)
    return counts


def header(img, r, n, label=""):
    v = r["track_vote"]
    n_ok = sum(m["ok"] for m in r["track_matches"])
    n_g = sum(t["state"] == "guaranteed" for t in r["tracks"])
    text(img, f"{label}frame {n}   vote: {v['support']} of {v['voters']} voters (streak {v['top_streak']}"
              f"{', ' + str(v['excluded_discarded']) + ' discarded' if v.get('excluded_discarded') else ''})"
              f"   motion {motion_text(r)}   "
              f"tracks {len(r['tracks'])}, "
              f"{n_g} guaranteed", (10, 20))
    text(img, f"matches {n_ok} ok (green), {r['track_lost']} lost (red), {r.get('track_backward', 0)} off-direction"
              f"   forgotten {r['track_safe']} safe + {r.get('track_dropped', 0)}   "
              f"discarded {len(r.get('track_discarded', []))}", (10, 42))
    return f"vote {v}, motion {motion_text(r)}, {n_ok} ok, {r['track_lost']} lost, {r.get('track_backward', 0)} " \
           f"off-direction, {len(r.get('track_discarded', []))} size-discarded, {len(r['tracks'])} tracks " \
           f"({n_g} guaranteed)"


class Videos:
    """mp4 and webm writers, opened on the first panel, each panel held for hold video frames."""

    def __init__(self, stem, hold):
        self.stem, self.hold, self.writers = stem, hold, None

    def write(self, panel):
        if self.writers is None:
            size = (panel.shape[1], panel.shape[0])
            self.writers = [cv2.VideoWriter(f"{self.stem}.{ext}", cv2.VideoWriter_fourcc(*cc), VIDEO_FPS, size)
                            for ext, cc in (("mp4", "avc1"), ("webm", "VP80"))]
        for w in self.writers:
            for _ in range(self.hold):
                w.write(panel)

    def release(self):
        for w in self.writers or []:
            w.release()


def compare_angles(args, paths, videos):
    """One tracker per DOWN_ANGLE_DEG, all fed the same Sobel and colour flags; panels side by side."""
    nav = SidewalkNavigator(None, SobelTexture(), ColorDifference(), None)
    trackers = [FeatureTracker() for _ in args.angles]
    for k, path in enumerate(paths):
        n = args.first + k
        frame = cv2.resize(cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))
        base = nav.process(frame, n - 1)
        flagged = base["anomaly"] | base["color_anomaly"]
        views = []
        for angle, tr in zip(args.angles, trackers):
            feature_tracking.DOWN_ANGLE_DEG = angle  # read at call time by FeatureTracker._forward
            r = {**base, **tr.update(base["sidewalk"], base["focus"], flagged, n - 1)}
            view = current_view(frame, r)
            print(f"frame {n} [{angle:g} deg]: " + header(view, r, n, f"DOWN_ANGLE_DEG {angle:g}   "))
            views.append(view)
        divider = np.full((EVAL_H, 6, 3), 255, np.uint8)
        panel = np.hstack(sum(([v, divider] for v in views), [])[:-1])
        cv2.imwrite(f"{args.out}/frame_{n:04d}.png", panel)
        videos.write(panel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1, help="first frame, 1-based")
    ap.add_argument("--last", type=int, default=23, help="last frame, 1-based, inclusive")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fps", type=float, default=1.0, help="review frames per second in the videos")
    ap.add_argument("--angles", type=float, nargs="+", help="compare these DOWN_ANGLE_DEG values side by side")
    ap.add_argument("--arrows", action="store_true", help="previous boxes with an arrow to the box each matched")
    ap.add_argument("--current", action="store_true", help="only each frame's current tentative and guaranteed boxes")
    ap.add_argument("--status", action="store_true",
                    help="every tracked box by status: tentative, guaranteed, coasting, discarded")
    args = ap.parse_args()
    if args.status and args.out == OUT:
        args.out = "runs/tracking_status"
    if args.angles and args.out == OUT:
        args.out = OUT + "_angles"
    if args.arrows and args.out == OUT:
        args.out = "runs/tracking_arrows"
    if args.current and args.out == OUT:
        args.out = "runs/tracking_current"
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(f"{SRC}/*.jpg"))[args.first - 1: args.last]
    hold = max(1, round(VIDEO_FPS / args.fps))
    stem = f"{args.out}/review_{args.first}-{args.last}"
    videos = Videos(stem, hold)
    if args.current or args.status:
        counts = (render_status if args.status else render_current)(args, paths, videos)
        videos.release()
        print(f"box-frames drawn: {counts}")
        print(f"Outputs in {args.out}/ (frame_*.jpg, {stem.split('/')[-1]}.mp4 and .webm)")
        return
    if args.angles or args.arrows:
        (compare_angles if args.angles else render_arrows)(args, paths, videos)
        videos.release()
        print(f"Outputs in {args.out}/ (frame_*.png, {stem.split('/')[-1]}.mp4 and .webm)")
        return
    nav = SidewalkNavigator(None, SobelTexture(), ColorDifference(), FeatureTracker())
    prev, prev_r = None, None
    for k, path in enumerate(paths):
        n = args.first + k
        frame = cv2.resize(cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4), (EVAL_W, EVAL_H))
        r = nav.process(frame, n - 1)
        right = current_view(frame, r)
        if prev is None:
            left = np.zeros_like(frame)
            text(left, "no previous frame", (10, 60))
        else:
            left = draw_tracks(prev.copy(), prev_r)
        text(left, f"frame {n - 1}: tracks after it" if prev is not None else "", (10, 20))
        print(f"frame {n}: " + header(right, r, n))
        panel = np.hstack([left, right])
        cv2.imwrite(f"{args.out}/frame_{n:04d}.png", panel)
        videos.write(panel)
        prev, prev_r = frame, r
    videos.release()
    print(f"Outputs in {args.out}/ (frame_*.png, {stem.split('/')[-1]}.mp4 and .webm)")


if __name__ == "__main__":
    main()
