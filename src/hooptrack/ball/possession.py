"""Who has the ball, frame by frame, from the tracked ball and the tracked players (possession chapter, P01).

Rules (all fixed before measuring, from what a possession looks like in the image)
--------------------------------------------------------------------------------
1. **Within reach**: a player can have the ball if it is inside his *reach box*: his box widened by
   ``reach_x`` of its width on each side (arms out) and raised by ``reach_up`` of its height (ball over the head); its
   bottom is the feet (a dribble touches the floor at the feet).
2. **Moving with him**: a ball flying across a player (a pass going by) is inside his reach for a frame or two but
   moves fast; a held or dribbled ball moves with the player. Over ``speed_window`` frames, the ball must move less
   than ``max_speed`` ball diameters per frame relative to the player's box centre.
3. **Closest**: among the players that pass 1 and 2, the one whose "hands" (``hands_y`` of his height from the top)
   are closest to the ball, in units of his own size.
4. **In time**: a holder must last ``min_run`` frames, short drops of ``max_drop`` frames inside one player's
   possession are filled (the ball hidden for a moment), and frames without a ball keep the last holder for at most
   ``max_no_ball`` frames.

Output per frame: the player's track id, ``AIR`` (a ball, nobody has it) or ``UNKNOWN`` (no ball). Events: the frames
where the holder changes (``catch``: AIR -> player, ``release``: player -> AIR, a direct player -> player change counts
as both).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AIR, UNKNOWN = -1, -2


@dataclass
class PossessionConfig:
    reach_x: float = 0.35         # of the box width, each side
    reach_up: float = 0.25        # of the box height, above the head
    hands_y: float = 0.45         # "hands" height, from the top of the box
    max_speed: float = 1.5        # ball diameters per frame, relative to the player
    speed_window: int = 3
    min_run: int = 4
    max_drop: int = 3             # "air" this short between two stretches of the SAME player: still his
    max_drop_switch: int = 0      # P01b: "air" this short between two DIFFERENT players: held (half each); 0 = off
    max_no_ball: int = 10
    # P04: frames without a ball keep the last state (holder or air) for as long as the ball is not seen, instead of
    # becoming "no ball" after ``max_no_ball`` frames; frames before the first ball take the first state. A ball we do
    # not see did not change hands as far as we know. False = P01.
    keep_state: bool = False


def raw_holder(ball: tuple[float, float, float], ids: np.ndarray, boxes: np.ndarray, cfg: PossessionConfig) -> int:
    """Rules 1 and 3 for one frame: the closest player within reach, or AIR."""
    x, y, _ = ball
    best, best_d = AIR, np.inf
    for i, (x1, y1, x2, y2) in zip(ids, np.asarray(boxes, float), strict=True):
        w, h = x2 - x1, y2 - y1
        if not (x1 - cfg.reach_x * w <= x <= x2 + cfg.reach_x * w and y1 - cfg.reach_up * h <= y <= y2):
            continue
        d = np.hypot((x - (x1 + x2) / 2) / w, (y - (y1 + cfg.hands_y * h)) / h)
        if d < best_d:
            best, best_d = int(i), d
    return best


def possession(ball: dict[int, tuple], players: dict[int, tuple[np.ndarray, np.ndarray]], n_frames: int,
               cfg: PossessionConfig | None = None) -> np.ndarray:
    """``ball`` {frame: (cx, cy, d, ...)}, ``players`` {frame: (ids, TLBR)} -> holder per frame (index = frame - 1)."""
    cfg = cfg or PossessionConfig()
    raw = np.full(n_frames, UNKNOWN)
    for f in range(1, n_frames + 1):
        if f in ball:
            ids, boxes = players.get(f, (np.zeros(0, int), np.zeros((0, 4))))
            raw[f - 1] = raw_holder(ball[f][:3], ids, boxes, cfg)
    # rule 2: the ball must move with the holder (relative speed in ball diameters per frame)
    for f in range(1, n_frames + 1):
        h = raw[f - 1]
        if h < 0:
            continue
        speeds = []
        for g in range(max(1, f - cfg.speed_window), min(n_frames, f + cfg.speed_window)):
            if g in ball and g + 1 in ball and g in players and g + 1 in players:
                c = []
                for t in (g, g + 1):
                    ids, boxes = players[t]
                    k = np.flatnonzero(ids == h)
                    if not len(k):
                        break
                    b = boxes[k[0]]
                    c.append((ball[t][0] - (b[0] + b[2]) / 2, ball[t][1] - (b[1] + b[3]) / 2))
                if len(c) == 2:
                    speeds.append(np.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]) / max(ball[g][2], 1.0))
        if speeds and np.median(speeds) > cfg.max_speed:
            raw[f - 1] = AIR
    # rule 4: time
    out = raw.copy()
    last, gap = UNKNOWN, 0
    for k in range(n_frames):                               # no ball: keep the holder a little
        if out[k] == UNKNOWN:
            gap += 1
            if cfg.keep_state and last != UNKNOWN:          # P04: keep any state, however long
                out[k] = last
            elif last >= 0 and gap <= cfg.max_no_ball:
                out[k] = last
        else:
            last, gap = out[k], 0
    if cfg.keep_state:                                      # P04: before the first ball, the first state
        known = np.flatnonzero(out != UNKNOWN)
        if len(known):
            out[:known[0]] = out[known[0]]
    runs = _runs(out)
    for s, e, v in runs:                                    # short drops inside one player's possession
        if v == AIR and e - s <= cfg.max_drop and s > 0 and e < n_frames and out[s - 1] >= 0 and out[s - 1] == out[e]:
            out[s:e] = out[s - 1]
        elif (v == AIR and e - s <= cfg.max_drop_switch and s > 0 and e < n_frames and out[s - 1] >= 0
              and out[e] >= 0 and out[s - 1] != out[e]):  # P01b: a hand-off or a flip between two close players
            mid = (s + e) // 2
            out[s:mid], out[mid:e] = out[s - 1], out[e]
    for s, e, v in _runs(out):                              # too short to be a possession: the ball is in the air
        if v >= 0 and e - s < cfg.min_run:
            out[s:e] = AIR
    return out


def _runs(a: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end exclusive, value) of each run of equal values."""
    if not len(a):
        return []
    cut = np.flatnonzero(np.diff(a)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(a)]
    return [(int(s), int(e), int(a[s])) for s, e in zip(starts, ends, strict=True)]


def events(holder: np.ndarray) -> list[tuple[int, str, int, int]]:
    """(frame, "catch" | "release", from, to) at every change of holder; UNKNOWN stretches are skipped."""
    out, prev = [], None
    for k, v in enumerate(holder):
        if v == UNKNOWN:
            continue
        if prev is not None and v != prev:
            if prev >= 0:
                out.append((k + 1, "release", int(prev), int(v)))
            if v >= 0:
                out.append((k + 1, "catch", int(prev), int(v)))
        prev = v
    return out


# ============================================================================ P02: air from the ball's own motion
# Why: P01 decided "air" from the distance between the ball and the player boxes, frame by frame: too noisy (the
# boxes jitter, two players overlap, the holder flips between an attacker and his defender), 3x too many changes.
# A pass or a shot is visible in the ball itself: it flies, fast and for several frames; a held or dribbled ball
# moves with a player. So here "air" = the ball moves faster than ``v_air`` ball diameters per frame, after removing
# the camera's own motion (the median motion of the players' boxes), for at least ``min_air`` frames. Who has the ball
# is decided once per held stretch (the player most often within reach in it), not frame by frame.


@dataclass
class MotionPossessionConfig:
    v_air: float = 1.5            # ball diameters per frame, camera motion removed
    min_air: int = 4              # frames a flight must last
    smooth: int = 3               # median over this many frames of the speed
    max_no_ball: int = 10         # frames without a ball that keep the current state
    reach: PossessionConfig | None = None   # reach box for choosing the holder (P01's)


def _camera_shift(players: dict, f: int) -> tuple[float, float]:
    """Median displacement of the player boxes kept from frame f-1 to f (a proxy of the camera's pan)."""
    if f - 1 not in players or f not in players:
        return 0.0, 0.0
    (i0, b0), (i1, b1) = players[f - 1], players[f]
    common = np.intersect1d(i0, i1)
    if not len(common):
        return 0.0, 0.0
    c0 = {int(i): ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for i, b in zip(i0, b0, strict=True)}
    c1 = {int(i): ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for i, b in zip(i1, b1, strict=True)}
    d = np.array([(c1[int(i)][0] - c0[int(i)][0], c1[int(i)][1] - c0[int(i)][1]) for i in common])
    return float(np.median(d[:, 0])), float(np.median(d[:, 1]))


def possession_motion(ball: dict[int, tuple], players: dict[int, tuple[np.ndarray, np.ndarray]], n_frames: int,
                      cfg: MotionPossessionConfig | None = None) -> np.ndarray:
    """Holder per frame (index = frame - 1): a player's id, AIR, or UNKNOWN (no ball for too long)."""
    cfg = cfg or MotionPossessionConfig()
    reach = cfg.reach or PossessionConfig()
    speed = np.full(n_frames, np.nan)
    for f in range(2, n_frames + 1):
        if f in ball and f - 1 in ball:
            sx, sy = _camera_shift(players, f)
            dx, dy = ball[f][0] - ball[f - 1][0] - sx, ball[f][1] - ball[f - 1][1] - sy
            speed[f - 1] = np.hypot(dx, dy) / max(ball[f][2], 1.0)
    sm = np.full(n_frames, np.nan)
    for k in range(n_frames):
        w = speed[max(0, k - cfg.smooth // 2):k + cfg.smooth // 2 + 1]
        w = w[~np.isnan(w)]
        if len(w):
            sm[k] = np.median(w)
    state = np.full(n_frames, UNKNOWN)
    state[~np.isnan(sm)] = 0                               # 0 = held, AIR = flying
    state[(sm > cfg.v_air)] = AIR
    for s, e, v in _runs(state):                           # a flight must last
        if v == AIR and e - s < cfg.min_air:
            state[s:e] = 0
    last, gap = UNKNOWN, 0                                 # no ball for a moment: keep the state
    for k in range(n_frames):
        if state[k] == UNKNOWN:
            gap += 1
            if last != UNKNOWN and gap <= cfg.max_no_ball:
                state[k] = last
        else:
            last, gap = state[k], 0
    out = state.copy()
    for s, e, v in _runs(state):                           # one holder per held stretch: the most frequent
        if v != 0:
            continue
        votes: dict[int, int] = {}
        for k in range(s, e):
            f = k + 1
            if f in ball:
                ids, boxes = players.get(f, (np.zeros(0, int), np.zeros((0, 4))))
                h = raw_holder(ball[f][:3], ids, boxes, reach)
                if h >= 0:
                    votes[h] = votes.get(h, 0) + 1
        out[s:e] = max(votes, key=votes.get) if votes else 10 ** 6   # held, holder not found (still "held")
    return out
