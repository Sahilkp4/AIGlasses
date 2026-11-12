"""Feature tracking: links flagged obstacle blobs across processed frames to confirm or dismiss them.

The stage after the Sobel texture check and the colour difference check. They flag pixels frame by frame with no memory;
this stage keeps a sliding window of two processed frames (the tracks from the last frame and the blobs of this one):

  Box geometry, not centres. A blob's centre jumps whenever it merges with neighbouring flags, splits or grows, so
  no step uses it. How box b moved from box a is measured on its edges: vertically by the change of the bottom edge
  (where an obstacle meets the ground, the point the cell motion describes), sideways by the gap between the two
  boxes' horizontal ranges (0 when they overlap). The same measure gives the residual from a predicted box. The
  outward drift of the side columns is not measured but comes from each cell's straight-walk direction (step 2).

  0. Straight-walk prior. The wearer is assumed to always walk forward in a straight line, so fixed ground objects
     move away from the point the wearer walks towards (horizontal centre, horizon row HORIZON_FRAC of the height):
     down in the centre column, down and outward in the side columns. Each grid cell (step 2) has a straight-walk
     direction, the direction of its initial default. Any (track, box) pairing whose displacement points more than
     DOWN_ANGLE_DEG away from the direction of the track's cell is treated as a wrong match or a false feature and
     discarded before voting and before matching, so the track can pair with a box that does move that way.
     Displacements shorter than MIN_DIR_PX pass, since their direction is noise. When the wearer walks backwards or
     sideways nothing matches: the tracks coast or are dropped, and new boxes start new tracks once forward walking
     resumes.
  1. Boxes. The union of the Sobel and colour anomaly masks is split into 8-connected blobs; blobs under MIN_OBJECT_PX
     are dropped, and each kept blob gets a bounding box.
  2. Motion vote per grid cell (a "band" in the code). Walking forward, fixed ground objects move faster the lower
     they sit in the image (about with the square of the distance below the horizon), and they drift outward from the
     point the wearer walks towards, so one motion cannot fit the whole frame. The frame is split into N_REGIONS
     horizontal regions, sized by REGION_FRACS from the bottom up: region 1 is the bottom 35% of the frame, region 2
     the next 18%, region 3 the next 9% and region 4 the top 38% (rows 351-540, 254-351, 205-254 and 0-205 at
     960x540). Each region is split into N_COLUMNS columns, sized by COLUMN_FRACS from the left: 25%, 50% and 25% of
     the width (x 0-240, 240-720 and 720-960). That gives N_BANDS = 12 grid cells, each with its own motion, outlier
     spread, direction and default, estimated independently. A box belongs to the cell holding the majority of its
     area (ties: the lower region); a track's motion comes from the cell of its box in the previous frame. A cell's
     motion is in pixels per camera frame at its region's bottom edge y_ref; inside the region it is scaled by
     s(y) = ((y - top) / (y_ref - top))^2 for a bottom row y (top: first row of the focus area, a horizon proxy; s
     floored at MIN_BAND_SCALE, and 1 for a region wholly above the focus area) and by the frame gap; the column does
     not change the scale. The voters are the established tracks (seen in at least track_frames consecutive frames,
     not coasting) whose box was in the cell. Each voter is matched first (step 3) with its cell's default motion as
     its prediction, and its down move (change of the bottom edge) divided by s(y) and the gap is its vote on the
     cell's speed; a cell's direction stays its straight-walk direction and is never voted. A vote
     weighs streak / track_frames, so a box tracked across 6 frames counts double one tracked across 3. A voter votes
     only when its matched box is in the same cell as its previous box: a box that moved to another cell votes again
     once it has been in one cell in two frames in a row. A voter the size check (step 7) would discard this frame
     does not vote, nor one that split (step 5), nor one at or heading past the frame's bottom edge (its old or
     matched box ending within EDGE_MARGIN_PX of it, or, with EDGE_PREDICTED, its predicted box reaching past it):
     the box is clipped there, so its measured move is cut short. With at least MIN_PAIRS votes the cell's speed is
     their weighted median, and the cell's outlier spread for step 4 is the weighted MAD-based spread of their down
     moves from it. A speed that is not downward is no information. The speed is then bounded around the cell's
     default speed, held to [1 / VOTE_BOUND, VOTE_BOUND] times it, and the cell's motion becomes its straight-walk
     direction at that speed, which is also its new default. One wrong vote can then move a cell by at most VOTE_BOUND, which also keeps predictions wide
     enough to reach the true box, while a true lasting change is followed within a few frames (1.5: a 3x change in
     about 3). A cell without information uses its default: its last bounded vote, or until its first vote its
     straight-walk default, set from the first frame's size: default_forward[region] down and
     default_forward[region] * (x - centre) / (y_ref - horizon) sideways at the cell's column centre x, so the side
     columns point outward, and lower regions get bigger moves, sideways and down, at smaller angles from straight
     down (at 960x540: R1 side columns 157 px out and 120 down, 52 degrees; R2 46 out and 11 down, 76 degrees;
     regions above the horizon straight down). A track's cell motion is then used for its matching, outlier test and
     coasting alike; a coasting box is re-assigned to a cell every frame.
  3. Match. Every live track's box is moved by the expected motion and padded by MATCH_PAD_PX + MATCH_PAD_FRAC times
     the expected displacement (the prediction's uncertainty). A current box sharing at least MIN_OVERLAP of the
     smaller of the two boxes with the padded prediction is a candidate, so a box that grew or merged around it still
     matches the box it grew from. Candidates are ranked by shared area in pixels (ties: smaller bottom-edge
     residual): when a box splits, the largest piece keeps the track, and when several old boxes merge into one, the
     largest of them keeps it, instead of a small fragment that happens to lie wholly inside. A box that does not
     overlap the padded prediction by at least MIN_OVERLAP is never a candidate. Matching is one-to-one.
  4. Outliers. A pair whose residual from the predicted box (bottom edge, horizontal gap) is more than OUTLIER_K
     spreads (MAD-based, floored), or whose displacement points more than MAX_ANGLE_DEG away from the expected motion,
     is lost: the old track coasts and the new box starts a new track. Unmatched boxes start new tracks too.
  5. Confirmation. A track seen in track_frames consecutive processed frames (TRACK_FRAMES = 3) is a guaranteed obstacle
     and stays one. Splits: when at least MIN_SPLIT_PIECES boxes smaller than a track's box overlap its padded
     predicted area (the matching test), the flagged area broke into pieces. The track keeps the largest piece, but
     that piece has other edges than the whole, so its move would look like motion away from the pieces that split
     off: the track does not vote this frame, its streak restarts at 1 (a guaranteed track stays guaranteed) and its
     size history restarts with the piece. The other pieces start new tracks.
  6. Unmatched tracks. A tentative track with no match is forgotten at once (counted as safe when at least SAFE_FRAC
     of its predicted box is sidewalk-labelled and unflagged, otherwise as dropped); it never coasts. A guaranteed
     track coasts with the expected motion until its predicted box is entirely outside the frame, and that box stays
     removed from the refined masks meanwhile. Each frame a coasting track is discarded (forgotten, its box no longer
     removed) when at least SAFE_FRAC of its expected box is clean sidewalk (sidewalk-labelled, with no Sobel or
     colour flags) OR when its box height and width both did not grow over its last track_frames matched frames:
     either the predicted spot shows nothing there, or the object had stopped approaching before it was lost.
  7. Size check (matched tracks). Walking towards a real obstacle makes its box grow. A guaranteed track whose box
     height and box width both did not grow at any step over its last track_frames matched frames, and whose matched
     box is sidewalk-labelled (at least SAFE_FRAC of it), is discarded as a false obstacle: the track is forgotten and
     its pixels are given back to the refined masks this frame. Growth in either dimension keeps the track, so a
     false band that widens by merging with flags along its row is kept too.

Output mask tracked_anomaly: every kept current blob (removed at once, whether tentative or guaranteed) plus the
predicted boxes of coasting guaranteed tracks. Blobs under MIN_OBJECT_PX are given back to the sidewalk.

Motion is kept per camera frame and scaled by the gap between processed frames, so frames the live loop drops are
accounted for when process() is given frame indices.

Known limits:
  - Walking backwards, sideways or turning breaks the straight-walk prior: matches are discarded until the wearer walks
    forward again. Head sway that shifts the whole image sideways for a step does the same.
  - A guaranteed obstacle that never leaves the frame (standing still, motion ~ 0) or a false one keeps coasting.
  - One motion per cell: the outward drift grows with the distance from the frame centre, but every box in a side
    column gets its cell's drift. The straight-walk defaults assume the wearer walks towards the frame centre at a
    fixed horizon; turning the head breaks them until the voters re-estimate the cells.
  - The sideways move follows the bottom edge's middle, so a box that grows or merges to one side appears to move
    that way by half the growth.
  - The in-band scale relies on the focus top as the horizon; when the focus area shrinks or breaks up, the top moves
    and every scale with it.
  - With MIN_PAIRS 1 a single established box can set the motion for its whole band; if it is a false obstacle that
    happens to move consistently, every other match in the band is judged against its motion.
  - Twelve cells split the voters twelve ways: many cells rarely vote and run on their default, and a cell's
    motion may rest on one or two boxes.
  - Each cell's last bounded vote is reused until its next one, however old it is. Wrong votes that keep pointing
    the same way still walk the default by VOTE_BOUND per frame.
  - The bound also slows the response to a true sudden change (stopping, or a frame-rate change): a region takes
    about log(change) / log(VOTE_BOUND) frames to follow it.
  - Without a distance fallback, a box that moves further than its padded prediction (a poor prediction at low frame
    rates, or a thin box) is not matched: its track is forgotten or coasts, and the box starts a new track.
  - Obstacles that move on their own (people, bikes) are outliers by design, so they never become guaranteed; their
    current flags are still removed each frame.
  - Overlap matching follows a box that grows or merges, but a merged box is one track: two obstacles that merge are
    tracked as one, and a box that splits keeps its track on the piece overlapping most; the rest start new tracks.

Tuning:
  TRACK_FRAMES   Consecutive frames before a track is guaranteed (--track-frames); also the streak of a vote of
                 weight 1.
  MATCH_PAD_PX, MATCH_PAD_FRAC   Padding of the predicted box for overlap matching: MATCH_PAD_PX plus MATCH_PAD_FRAC
                 times the expected displacement, on every side.
  MIN_OVERLAP    Least shared area, as a share of the smaller box, for a match candidate.
  MIN_SPLIT_PIECES  Smaller boxes in a track's predicted area that mean it split (see step 5).
  MIN_PAIRS      Votes needed for a voted motion; with fewer the default (the last vote) is used.
  OUTLIER_K, MIN_SPREAD_PX, REL_SPREAD   Outlier cut-off: OUTLIER_K * max(MAD spread, MIN_SPREAD_PX,
                 REL_SPREAD * expected displacement) pixels.
  DOWN_ANGLE_DEG Straight-walk cone: largest angle from the cell's straight-walk direction for any match or vote.
  HORIZON_FRAC   Horizon row, as a share of the height, for the straight-walk defaults and directions; 264 / 540 is
                 the median focus top on dataset_flip, which matched the horizon fitted to the ground motion.
  MAX_ANGLE_DEG, MIN_DIR_PX   Direction test, only applied when both expected and actual motion are at least MIN_DIR_PX.
  REGION_FRACS   Region heights as shares of the frame, bottom up; N_REGIONS is their number.
  COLUMN_FRACS   Column widths as shares of the frame, left to right; N_COLUMNS is their number. Each of the
                 N_BANDS = N_REGIONS x N_COLUMNS grid cells has its own motion, spread and default.
  MIN_BAND_SCALE Floor on the in-region scale s(y). Without it a track at the focus top divides its displacement by ~0.
  VOTE_BOUND     Largest factor a region's vote may differ from its default, per frame. 1.5 gave the closest coasting
                 moves on dataset_flip frames 1-23 (homography truth); smaller steadies more but follows slower.
  EDGE_MARGIN_PX, EDGE_PREDICTED   Voters at or heading past the frame's bottom edge are left out (see step 2).
                 Flags often stop a few rows short of the edge, hence the margin.
  DEFAULT_BAND_PX  Initial forward motion of each region (1-4, bottom up, seeding all its columns), straight down, in pixels per camera frame at
                 the region's bottom edge (--forward-px); used until the region's first successful vote (and again
                 after reset()); also its voters' first prediction. 120 and 11 are the median true ground motion at
                 rows 540 and 351 on dataset_flip frames 3-23 (ground-plane homography), whose frames are about one
                 walking step apart; regions 3 and 4 lie above the horizon there and hold no sidewalk (2). At a live
                 30 FPS they should be about 15 times smaller.
  SAFE_FRAC      Share of a box that must be sidewalk: clean sidewalk for a lost tentative track to count as safe, and
                 sidewalk-labelled for the size check to discard a non-growing guaranteed track.
  MAX_TRACKS     Memory guard: beyond this, the oldest coasting tentative tracks are forgotten first.
"""
import argparse
import time

import cv2
import numpy as np

TRACK_FRAMES = 3  # consecutive matched frames before a track is a guaranteed obstacle
MIN_OBJECT_PX = 50  # flagged blobs smaller than this are dropped (same as the colour check)
MIN_PAIRS = 1  # votes needed for a voted motion; with fewer the default (last voted) motion is used
MATCH_PAD_PX = 4.0  # padding of the predicted box for overlap matching, pixels
MATCH_PAD_FRAC = 0.5  # plus this share of the expected displacement
MIN_OVERLAP = 0.2  # least shared area, as a share of the smaller box, for an overlap candidate
MIN_SPLIT_PIECES = 2  # this many smaller boxes in a track's predicted area mean it split: streak reset, no vote
OUTLIER_K = 3.0  # outlier cut-off in spreads
MIN_SPREAD_PX = 4.0  # floor on the spread, pixels
REL_SPREAD = 0.25  # floor on the spread as a share of the expected displacement
MAX_ANGLE_DEG = 60.0  # pairs further than this from the expected direction are lost
MIN_DIR_PX = 3.0  # direction is only tested when expected and actual displacement are at least this
DOWN_ANGLE_DEG = 60.0  # straight-walk prior: matches and votes must point within this of straight down
REGION_FRACS = (0.35, 0.18, 0.09, 0.38)  # region heights as shares of the frame, bottom up: regions 1, 2, 3, 4
N_REGIONS = len(REGION_FRACS)  # horizontal regions
COLUMN_FRACS = (0.25, 0.5, 0.25)  # column widths as shares of the frame, left to right
N_COLUMNS = len(COLUMN_FRACS)
N_BANDS = N_REGIONS * N_COLUMNS  # grid cells with independent motion; cell = region * N_COLUMNS + column
MIN_BAND_SCALE = 0.02  # floor on the in-region scale s(y)
EDGE_MARGIN_PX = 8  # a voter whose box ends this close to the frame's bottom edge does not vote
EDGE_PREDICTED = True  # nor one whose predicted box (under the region's default) reaches past the bottom edge
VOTE_BOUND = 1.5  # a region's vote is bounded to [1 / VOTE_BOUND, VOTE_BOUND] times the length of its default
HORIZON_FRAC = 264 / 540  # straight-walk defaults: horizon row as a share of the height (dataset_flip focus top)
DEFAULT_BAND_PX = (120.0, 11.0, 2.0, 2.0)  # forward motion of each region (1-4, bottom up) until its first vote,
# px per camera frame at the region's bottom edge
SAFE_FRAC = 0.5  # share of a predicted box that must be clean sidewalk for a lost tentative track to be safe
MAX_TRACKS = 200  # memory guard on live tracks

TENTATIVE_COLOR = (0, 230, 230)
GUARANTEED_COLOR = (0, 0, 255)


def _x_gap(a, b):
    """Signed horizontal gap from box a's x-range to box b's, 0 when they overlap. Boxes are (x, y, w, h)."""
    if b[0] > a[0] + a[2]:
        return b[0] - (a[0] + a[2])
    if b[0] + b[2] < a[0]:
        return b[0] + b[2] - a[0]
    return 0.0


def _move(a, b):
    """How box b moved from box a: (horizontal gap, change of the bottom edge). Edge-based, so a box that grows,
    merges or changes shape sideways does not appear to move sideways. (Measuring the sideways move by the bottom
    edge's middle instead cut accepted matches on dataset_flip frames 1-23 from 160 to 92: flagged blobs change shape
    every frame.) The sideways drift comes from each cell's straight-walk direction instead (step 2)."""
    return np.array([_x_gap(a, b), (b[1] + b[3]) - (a[1] + a[3])], dtype=float)


def _shift(box, d):
    return (box[0] + d[0], box[1] + d[1], box[2], box[3])


def _pad(box, p):
    return (box[0] - p, box[1] - p, box[2] + 2 * p, box[3] + 2 * p)


def _overlap(a, b):
    """(shared area of boxes a and b in pixels, that area as a share of the smaller box)."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    return w * h, w * h / max(min(a[2] * a[3], b[2] * b[3]), 1.0)


def region_tops(height):
    """Top row of each region, bottom up (region 1 first): REGION_FRACS stacked from the bottom of the frame."""
    tops, cum = [], 0.0
    for f in REGION_FRACS:
        cum += f
        tops.append(max(height * (1 - cum), 0.0))
    return tops


def column_lefts(width):
    """Left column of each grid column, left to right: COLUMN_FRACS stacked from the left edge of the frame."""
    lefts, cum = [], 0.0
    for f in COLUMN_FRACS:
        lefts.append(width * cum)
        cum += f
    return lefts


def _weighted_median(values, weights):
    """Weighted median of values (n,) or, per column, of values (n, k). Equal weights give np.median."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.ndim == 2:
        return np.array([_weighted_median(values[:, j], weights) for j in range(values.shape[1])])
    order = np.argsort(values)
    v, cw = values[order], np.cumsum(weights[order])
    half = cw[-1] / 2
    i = int(np.searchsorted(cw, half))
    if np.isclose(cw[i], half) and i + 1 < len(v):  # weight splits exactly here: average the two middle values
        return float((v[i] + v[i + 1]) / 2)
    return float(v[i])


def _bottom_mid(box):
    """Middle of the bottom edge, for drawing only."""
    return [round(float(box[0] + box[2] / 2), 1), round(float(box[1] + box[3]), 1)]


class Track:
    __slots__ = ("id", "box", "streak", "guaranteed", "coasting", "age", "heights", "widths")

    def __init__(self, tid, box):
        self.id, self.box, self.streak, self.guaranteed, self.coasting, self.age = tid, box, 1, False, False, 0
        self.heights, self.widths = [box[3]], [box[2]]  # box sizes of the matched frames, newest last

    @property
    def bottom(self):
        return self.box[1] + self.box[3]

    def as_dict(self):
        return {"id": self.id, "box": [int(round(v)) for v in self.box], "streak": self.streak,
                "state": "guaranteed" if self.guaranteed else "tentative", "coasting": self.coasting}


class FeatureTracker:
    def __init__(self, track_frames=TRACK_FRAMES, min_object_px=MIN_OBJECT_PX, default_forward=DEFAULT_BAND_PX):
        if track_frames < 1:
            raise ValueError(f"track_frames must be at least 1, got {track_frames}")
        if len(default_forward) != N_REGIONS:
            raise ValueError(f"default_forward needs {N_REGIONS} values (regions 1-{N_REGIONS}, bottom up), "
                             f"got {default_forward}")
        self.track_frames, self.min_object_px = track_frames, min_object_px
        self.region_px = [float(v) for v in default_forward]
        self.reset()

    def reset(self):
        """Forgets all tracks and every cell's last vote, for a new session. The straight-walk defaults are set from
        the first frame's size."""
        self.tracks, self.last_index, self.next_id = [], None, 0
        self.size, self.initial_motions, self.walk_dirs = None, None, None
        self.default_motions, self.default_sources = None, ["default"] * N_BANDS

    def _init_defaults(self, height, width):
        """Straight-walk defaults of every cell for this frame size. Walking straight ahead, ground points move away
        from the point the wearer walks towards (horizontal centre, horizon row HORIZON_FRAC * height): at the
        cell's reference point (column centre, region bottom edge) the move is region_px down and
        region_px * (x - centre) / (y_ref - horizon) sideways, so side columns point outward and the sideways pixels
        grow lower in the frame while the angle from straight down shrinks. Regions wholly above the horizon point
        straight down. walk_dirs (unit vectors) are the straight-walk directions the cone is centred on."""
        tops, lefts = region_tops(height), column_lefts(width) + [float(width)]
        horizon, centre = HORIZON_FRAC * height, width / 2
        motions = []
        for r in range(N_REGIONS):
            ref = height if r == 0 else tops[r - 1]
            for c in range(N_COLUMNS):
                x_mid = (lefts[c] + lefts[c + 1]) / 2
                dx = self.region_px[r] * (x_mid - centre) / (ref - horizon) if ref > horizon else 0.0
                motions.append([dx, self.region_px[r]])
        self.initial_motions = np.array(motions)
        self.walk_dirs = self.initial_motions / np.linalg.norm(self.initial_motions, axis=1, keepdims=True)
        self.default_motions, self.default_sources = self.initial_motions.copy(), ["default"] * N_BANDS
        self.size = (height, width)

    def _new_track(self, box):
        self.next_id += 1
        return Track(self.next_id, box)

    @staticmethod
    def _region(bottom, height):
        """Region index (0 = region 1, the bottom one) of a box with this bottom edge: a box spanning two regions
        belongs to the lower one."""
        row = bottom - 1
        for k, t in enumerate(region_tops(height)):
            if row >= t:
                return k
        return N_REGIONS - 1

    @staticmethod
    def _column(x_mid, width):
        """Column index (0 = left) of a horizontal position."""
        col = 0
        for k, left in enumerate(column_lefts(width)):
            if x_mid >= left:
                col = k
        return col

    @classmethod
    def _cell(cls, box, height, width):
        """Grid cell holding the majority of a box's area (within the frame); ties go to the lower region. A box wholly
        outside the frame falls back to the cell of its bottom edge's middle."""
        x, y, w, h = box
        tops, lefts = region_tops(height), column_lefts(width) + [float(width)]
        best, best_area = None, 0.0
        for r in range(N_REGIONS):
            bottom = height if r == 0 else tops[r - 1]
            rows = min(y + h, bottom) - max(y, tops[r])
            if rows <= 0:
                continue
            for c in range(N_COLUMNS):
                cols = min(x + w, lefts[c + 1]) - max(x, lefts[c])
                if cols > 0 and rows * cols > best_area:
                    best, best_area = r * N_COLUMNS + c, rows * cols
        if best is None:
            return cls._region(y + h, height) * N_COLUMNS + cls._column(x + w / 2, width)
        return best

    @staticmethod
    def _region_scale(bottom, region, top, height, gap):
        """In-region scale s(y) of a bottom edge, times the frame gap: ((y - top) / (y_ref - top))^2 with y_ref the
        region's bottom edge, so the cell motion is the move at y_ref; 1 for a region wholly above the focus area.
        Columns do not change it."""
        ref = height if region == 0 else region_tops(height)[region - 1]
        if ref <= top:
            return float(gap)
        return max(((max(bottom - top, 0.0)) / (ref - top)) ** 2, MIN_BAND_SCALE) * gap

    def _expected(self, motions, box, top, height, width, gap):
        """Expected displacement in pixels of a ground object with this box, under per-cell motions."""
        cell = self._cell(box, height, width)
        return motions[cell] * self._region_scale(box[1] + box[3], cell // N_COLUMNS, top, height, gap)

    @staticmethod
    def _consistent(residual, d, expected, spread):
        """Whether a pair agrees with the expected motion: its residual from the predicted box within the outlier
        cut-off and, when both are long enough, displacement d in about the same direction as expected."""
        e = np.linalg.norm(expected)
        if np.linalg.norm(residual) > OUTLIER_K * max(spread, MIN_SPREAD_PX, REL_SPREAD * e):
            return False
        n = np.linalg.norm(d)
        if e >= MIN_DIR_PX and n >= MIN_DIR_PX:
            cos = float(np.dot(d, expected) / (n * e))
            if cos < np.cos(np.radians(MAX_ANGLE_DEG)):
                return False
        return True

    @staticmethod
    def _frac(mask, box):
        """Share of box (x, y, w, h, clipped to the frame) that is set in mask."""
        H, W = mask.shape
        x, y, w, h = box
        x0, y0, x1, y1 = max(int(round(x)), 0), max(int(round(y)), 0), min(int(round(x + w)), W), min(int(round(y + h)), H)
        region = mask[y0:y1, x0:x1]
        return float(region.mean()) if region.size else 0.0

    def _not_growing(self, t, new_box=None):
        """Whether neither the box height nor the box width grew at any step over the last track_frames matched frames
        (including new_box, if given, as the newest). Growth in either dimension keeps the track."""
        heights = (t.heights + ([new_box[3]] if new_box is not None else []))[-self.track_frames:]
        widths = (t.widths + ([new_box[2]] if new_box is not None else []))[-self.track_frames:]
        if len(heights) < self.track_frames:
            return False
        flat = lambda s: all(b <= a for a, b in zip(s, s[1:]))
        return flat(heights) and flat(widths)

    @staticmethod
    def _forward(d, walk_dir):
        """Straight-walk prior: whether displacements d (..., 2) point within DOWN_ANGLE_DEG of the cell's straight-walk
        direction walk_dir (a unit vector), or are too short to have a direction."""
        n = np.linalg.norm(d, axis=-1)
        return (n < MIN_DIR_PX) | (d @ walk_dir >= n * np.cos(np.radians(DOWN_ANGLE_DEG)))

    @staticmethod
    def _padded_prediction(t, e):
        """(predicted box, padded predicted box) of track t under expected displacement e: the area a match must
        overlap."""
        predicted = _shift(t.box, e)
        return predicted, _pad(predicted, MATCH_PAD_PX + MATCH_PAD_FRAC * float(np.linalg.norm(e)))

    @staticmethod
    def _split(t, boxes, e):
        """Whether track t split: at least MIN_SPLIT_PIECES current boxes smaller in area than its box overlap its
        padded predicted area by at least MIN_OVERLAP (the matching test). The piece it is matched to then has other
        edges than the whole, so its move is not a real move."""
        _, padded = FeatureTracker._padded_prediction(t, e)
        area = t.box[2] * t.box[3]
        pieces = sum(1 for b in boxes if b[2] * b[3] < area and _overlap(padded, b)[1] >= MIN_OVERLAP)
        return pieces >= MIN_SPLIT_PIECES

    def _match(self, idx, boxes, expected, H, W):
        """One-to-one match of the tracks at indices idx to boxes that overlap each padded predicted box by at least
        MIN_OVERLAP, most shared pixels first; pairings failing the straight-walk prior are discarded. Returns (pairs
        as (track index, box index), number discarded by the prior)."""
        candidates, n_backward = [], 0
        for ti in idx:
            t = self.tracks[ti]
            predicted, padded = self._padded_prediction(t, expected[ti])
            for bi, b in enumerate(boxes):
                shared, share = _overlap(padded, b)
                if share < MIN_OVERLAP:
                    continue  # does not overlap the expected position: never a candidate
                key = (-shared, float(np.linalg.norm(_move(predicted, b))))  # most shared pixels, then nearest
                if self._forward(_move(t.box, b), self.walk_dirs[self._cell(t.box, H, W)]):
                    candidates.append((key, ti, bi))
                else:  # moves up or sideways: a wrong match or a false feature under the straight-walk prior
                    n_backward += 1
        candidates.sort()
        pairs, used_t, used_b = [], set(), set()
        for _, ti, bi in candidates:
            if ti not in used_t and bi not in used_b:
                used_t.add(ti)
                used_b.add(bi)
                pairs.append((ti, bi))
        return pairs, n_backward

    def _vote(self, boxes, sidewalk, top, H, W, gap):
        """Per-band motion from the established tracks of each band, each matched with its band's default prediction
        and weighted by streak / track_frames; voters the size check would discard are left out. Returns (motion or
        None per band, spread px per band, vote info: totals plus "bands", one entry per band)."""
        voters = [ti for ti, t in enumerate(self.tracks) if not t.coasting and t.streak >= self.track_frames]
        bands = [{"band": k, "region": k // N_COLUMNS + 1, "column": "LCR"[k % N_COLUMNS], "voted": False, "voters": 0, "top_streak": 0, "support": 0, "weight": 0.0,
                  "excluded_discarded": 0, "excluded_edge": 0, "excluded_split": 0, "excluded_cell": 0}
                 for k in range(N_BANDS)]
        for ti in voters:
            b = bands[self._cell(self.tracks[ti].box, H, W)]
            b["voters"] += 1
            b["top_streak"] = max(b["top_streak"], self.tracks[ti].streak)
        motions, spreads = [None] * N_BANDS, [0.0] * N_BANDS
        votes = [[] for _ in range(N_BANDS)]  # per band: (displacement, scale, weight)
        if voters and boxes:
            default = [self._expected(self.default_motions, t.box, top, H, W, gap) for t in self.tracks]
            pairs, _ = self._match(voters, boxes, default, H, W)
            for ti, bi in pairs:
                t, b = self.tracks[ti], boxes[bi]
                k = self._cell(t.box, H, W)
                # at or heading past the frame's bottom edge: the box is clipped there, so its move is cut short
                if (max(t.bottom, b[1] + b[3]) >= H - EDGE_MARGIN_PX
                        or (EDGE_PREDICTED and t.bottom + default[ti][1] >= H)):
                    bands[k]["excluded_edge"] += 1
                    continue
                if self._split(t, boxes, default[ti]):  # split into pieces: the piece's move is not a real move
                    bands[k]["excluded_split"] += 1
                    continue
                if self._cell(b, H, W) != k:  # moved to another cell: votes only after two frames in the same cell
                    bands[k]["excluded_cell"] += 1
                    continue
                if t.guaranteed and self._not_growing(t, b) and self._frac(sidewalk, b) >= SAFE_FRAC:
                    bands[k]["excluded_discarded"] += 1  # about to be discarded as a false obstacle: no vote
                    continue
                votes[k].append((_move(t.box, b), self._region_scale(t.bottom, k // N_COLUMNS, top, H, gap),
                                 t.streak / self.track_frames))  # tracked across 6 frames: double one tracked across 3
        for k, v in enumerate(votes):
            bands[k]["support"], bands[k]["weight"] = len(v), round(float(sum(w for *_, w in v)), 2)
            if len(v) < MIN_PAIRS:
                continue
            # speed-only vote: each cell keeps its straight-walk direction; the voters set how far it moves, as the
            # down move at the region's bottom edge
            disp, scale, weight = np.array([d for d, _, _ in v]), np.array([s for _, s, _ in v]), [w for *_, w in v]
            speed = _weighted_median(disp[:, 1] / scale, weight)
            if speed <= 0:  # not towards the camera: no usable information
                bands[k]["not_forward"] = True
                continue
            # spread: how far the voters' down moves disagree with their own median, before the bound
            spreads[k] = 1.4826 * _weighted_median(np.abs(disp[:, 1] - speed * scale), weight)
            # bound the vote around the cell's default: one wrong vote moves it by at most VOTE_BOUND, while a true
            # lasting change is followed within a few frames as each bounded vote becomes the new default
            default_speed = float(self.default_motions[k][1])
            bands[k]["raw"] = round(float(speed), 2)
            bounded = min(max(speed, default_speed / VOTE_BOUND), default_speed * VOTE_BOUND)
            if not np.isclose(bounded, speed):
                bands[k]["bounded"] = True
            motions[k] = self.initial_motions[k] * (bounded / self.initial_motions[k][1])
            bands[k]["voted"] = True
        info = {"voted": any(b["voted"] for b in bands), "eligible": len(voters), "voters": len(voters),
                "top_streak": max(b["top_streak"] for b in bands), "support": sum(b["support"] for b in bands),
                "weight": round(sum(b["weight"] for b in bands), 2),
                "excluded_discarded": sum(b["excluded_discarded"] for b in bands),
                "excluded_edge": sum(b["excluded_edge"] for b in bands),
                "excluded_split": sum(b["excluded_split"] for b in bands),
                "excluded_cell": sum(b["excluded_cell"] for b in bands), "bands": bands}
        return motions, spreads, info

    def update(self, sidewalk, focus, flagged, frame_index=None):
        """sidewalk, focus, flagged: boolean masks of one frame (flagged: the union of the checks' anomalies).
        frame_index: camera frame number, so the motion is scaled by the frames dropped since the last call."""
        t0 = time.perf_counter()
        H, W = flagged.shape
        gap = 1
        if frame_index is not None and self.last_index is not None and frame_index > self.last_index:
            gap = frame_index - self.last_index
        self.last_index = frame_index
        top = cv2.boundingRect(focus.astype(np.uint8))[1] if focus.any() else 0
        if self.size != (H, W):  # first frame, or a new frame size: straight-walk defaults for this size
            self._init_defaults(H, W)

        # 1. boxes of the kept blobs
        n, lab, stats, _ = cv2.connectedComponentsWithStats(flagged.astype(np.uint8), connectivity=8)
        big = 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= self.min_object_px)
        keep = np.zeros(n, bool)
        keep[big] = True
        boxes = [tuple(float(v) for v in stats[i, :4]) for i in big]

        # 2. weighted motion vote per band; a band without one uses its last successful vote (or its initial forward
        # motion before its first vote)
        voted, spreads, vote_info = self._vote(boxes, sidewalk, top, H, W, gap)
        motions, sources = self.default_motions.copy(), list(self.default_sources)
        for k, m in enumerate(voted):
            if m is not None:
                motions[k], sources[k] = m, "vote"
                self.default_motions[k], self.default_sources[k] = m, "last vote"
        before = [(t.id, t.box) for t in self.tracks]

        # 3. one-to-one match of every track under its band's motion
        expected = [self._expected(motions, t.box, top, H, W, gap) for t in self.tracks]
        predicted = [_shift(t.box, e) for t, e in zip(self.tracks, expected)]
        pairs, n_backward = self._match(range(len(self.tracks)), boxes, expected, H, W)
        used_b = {bi for _, bi in pairs}

        # 4. outliers against the band's motion, with the band voters' spread
        accepted, lost = [], []
        for k, (ti, bi) in enumerate(pairs):
            ok = self._consistent(_move(predicted[ti], boxes[bi]), _move(self.tracks[ti].box, boxes[bi]),
                                  expected[ti], spreads[self._cell(self.tracks[ti].box, H, W)])
            (accepted if ok else lost).append(k)
        lost_set = set(lost)
        # accepted tracks that split into several smaller pieces (checked on the old box, before it is replaced)
        split_t = {pairs[k][0] for k in accepted if self._split(self.tracks[pairs[k][0]], boxes, expected[pairs[k][0]])}
        matches = [{"track": before[ti][0], "ok": k not in lost_set, "split": ti in split_t,  # points: bottom-edge middles, for drawing
                    "from": _bottom_mid(before[ti][1]), "predicted": _bottom_mid(predicted[ti]),
                    "to": _bottom_mid(boxes[bi]), "move": [round(float(v), 1) for v in _move(before[ti][1], boxes[bi])],
                    "from_box": [int(round(v)) for v in before[ti][1]], "to_box": [int(v) for v in boxes[bi]]}
                   for k, (ti, bi) in enumerate(pairs)]

        # 5. track states
        next_tracks, matched_t, discarded = [], set(), []
        for k in accepted:
            ti, bi = pairs[k]
            t = self.tracks[ti]
            t.box, t.coasting, t.age = boxes[bi], False, 0
            t.heights = (t.heights + [boxes[bi][3]])[-self.track_frames:]
            t.widths = (t.widths + [boxes[bi][2]])[-self.track_frames:]
            if ti in split_t:  # split: count from 1 again (a guaranteed track stays guaranteed), growth starts over
                t.streak, t.heights, t.widths = 1, [boxes[bi][3]], [boxes[bi][2]]
            else:
                t.streak += 1
            t.guaranteed = t.guaranteed or t.streak >= self.track_frames
            matched_t.add(ti)
            if t.guaranteed and self._not_growing(t) and self._frac(sidewalk, t.box) >= SAFE_FRAC:
                keep[big[bi]] = False  # 7. size check at the current position: its pixels go back to the refined masks
                discarded.append(t)
                continue
            next_tracks.append(t)
        for k in lost:  # the old track coasts below; its box starts over
            next_tracks.append(self._new_track(boxes[pairs[k][1]]))
        for bi in range(len(boxes)):
            if bi not in used_b:
                next_tracks.append(self._new_track(boxes[bi]))
        for t in next_tracks:  # a new track may already be confirmed when track_frames is 1
            t.guaranteed = t.guaranteed or t.streak >= self.track_frames

        # 6. unmatched tracks: tentative ones are forgotten, guaranteed ones coast
        safe, dropped, coast_mask = 0, 0, np.zeros_like(flagged, dtype=bool)
        clean = sidewalk & ~flagged
        for ti, t in enumerate(self.tracks):
            if ti in matched_t:
                continue
            if not t.guaranteed:
                x, y, w, h = t.box
                shift = self._expected(motions, t.box, top, H, W, gap)
                if self._frac(clean, (x + shift[0], y + shift[1], w, h)) >= SAFE_FRAC:
                    safe += 1
                else:
                    dropped += 1
                continue
            shift = self._expected(motions, t.box, top, H, W, gap)  # cell re-read every frame as it moves
            x, y, w, h = t.box
            t.box, t.coasting = (x + shift[0], y + shift[1], w, h), True
            t.age += 1
            bx, by = t.box[0], t.box[1]
            if bx >= W or bx + w <= 0 or by >= H or by + h <= 0:
                continue  # entirely outside the frame: forgotten
            # coasting check at the expected position: discarded when it lands on clean sidewalk (labelled sidewalk,
            # no Sobel or colour flags) OR when it had stopped growing before it was lost
            if self._frac(clean, t.box) >= SAFE_FRAC or self._not_growing(t):
                discarded.append(t)
                continue
            x0, y0 = max(int(round(t.box[0])), 0), max(int(round(t.box[1])), 0)
            x1, y1 = min(int(round(t.box[0] + w)), W), min(int(round(t.box[1] + h)), H)
            coast_mask[y0:y1, x0:x1] = True
            next_tracks.append(t)
        if len(next_tracks) > MAX_TRACKS:  # drop the oldest coasting tentative tracks first
            spare = sorted((t for t in next_tracks if t.coasting and not t.guaranteed), key=lambda t: -t.age)
            drop = {id(t) for t in spare[: len(next_tracks) - MAX_TRACKS]}
            next_tracks = [t for t in next_tracks if id(t) not in drop]
        self.tracks = next_tracks

        blobs = keep[lab]  # after the size check gave discarded obstacles back
        return {"tracked_anomaly": blobs | coast_mask, "tracks": [t.as_dict() for t in self.tracks],
                "motion": [[round(float(v), 2) for v in m] for m in motions], "motion_source": sources,  # per band
                "track_pairs": len(pairs), "track_lost": len(lost), "track_safe": safe, "track_matches": matches,
                "track_backward": n_backward, "track_dropped": dropped,
                "track_split": sorted(before[ti][0] for ti in split_t),  # self.tracks is already the next list
                "track_discarded": [{"id": t.id, "box": [int(round(v)) for v in t.box], "coasting": t.coasting}
                                    for t in discarded],
                "track_vote": vote_info,
                "tracking_ms": round(1000 * (time.perf_counter() - t0), 2)}


REGION_NAMES = tuple(f"R{k + 1}" for k in range(N_REGIONS))  # R1 is the bottom region
BAND_NAMES = tuple(f"{r}{c}" for r in REGION_NAMES for c in "LCR"[:N_COLUMNS])  # cells: R1L, R1C, R1R, R2L, ...
SOURCE_LETTER = {"vote": "v", "last vote": "l", "default": "d"}


def cell_text(motion, source):
    """Short label of one cell's motion: dy at its region's bottom edge, dx when not ~0, and v / l / d for vote, last
    vote or default."""
    dx, dy = motion
    return f"{dy:.0f}" + (f",{dx:+.0f}" if abs(dx) >= 0.5 else "") + SOURCE_LETTER.get(source, "?")


def motion_text(result):
    """One line with every cell's motion, grouped by region, bottom up: R1 L.. C.. R.. | R2 ..."""
    parts = []
    for r, name in enumerate(REGION_NAMES):
        cells = range(r * N_COLUMNS, (r + 1) * N_COLUMNS)
        parts.append(name + " " + " ".join(f"{'LCR'[k % N_COLUMNS]}{cell_text(result['motion'][k], result['motion_source'][k])}"
                                           for k in cells))
    return " | ".join(parts)


def add_tracking_args(ap):
    """Command-line options for feature tracking, shared by sidewalk_system.py and simulate_sidewalk.py."""
    ap.add_argument("--tracking", action=argparse.BooleanOptionalAction, default=True,
                    help="track flagged obstacles across frames (needs --sobel or --color)")
    ap.add_argument("--track-frames", type=int, default=TRACK_FRAMES,
                    help="consecutive frames before a tracked obstacle is guaranteed")
    ap.add_argument("--forward-px", type=float, nargs=N_REGIONS, default=list(DEFAULT_BAND_PX),
                    metavar=REGION_NAMES,
                    help="forward motion of each region, bottom up (R1 = bottom 35%%), seeding all its columns, until "
                         "each cell's first vote (later frames without a vote reuse the cell's last vote), px per "
                         "camera frame at the region's bottom edge")


def tracker(args):
    """The feature tracker set on the command line, or None when it is off or there is nothing to track."""
    if not args.tracking or not (args.sobel or args.color):
        return None
    return FeatureTracker(args.track_frames, default_forward=args.forward_px)


def _dashed_rect(img, p0, p1, color, dash=6):
    (x0, y0), (x1, y1) = p0, p1
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
        for s in range(0, length, 2 * dash):
            e = min(s + dash, length)
            pa = (int(a[0] + (b[0] - a[0]) * s / max(length, 1)), int(a[1] + (b[1] - a[1]) * s / max(length, 1)))
            pb = (int(a[0] + (b[0] - a[0]) * e / max(length, 1)), int(a[1] + (b[1] - a[1]) * e / max(length, 1)))
            cv2.line(img, pa, pb, color, 1)


def draw_tracks(img, result, scale=1.0, offset=(0, 0), current_only=False, width=1):
    """Draws the tracks of result onto img in place: yellow tentative, red guaranteed, dashed while coasting.
    current_only leaves out coasting tracks, so only this frame's boxes are drawn."""
    tracks = [t for t in result.get("tracks", []) if not (current_only and t["coasting"])]
    for t in sorted(tracks, key=lambda t: t["state"] == "guaranteed"):  # guaranteed drawn on top
        x, y, w, h = t["box"]
        p0 = (int(offset[0] + x * scale), int(offset[1] + y * scale))
        p1 = (int(offset[0] + (x + w) * scale), int(offset[1] + (y + h) * scale))
        color = GUARANTEED_COLOR if t["state"] == "guaranteed" else TENTATIVE_COLOR
        if t["coasting"]:
            _dashed_rect(img, p0, p1, color)
        else:
            cv2.rectangle(img, p0, p1, color, width)
        cv2.putText(img, f"{t['id']}:{t['streak']}", (p0[0], max(p0[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color, 1, cv2.LINE_AA)
    return img
