"""Side-by-side video of two (or more) finished runs on one clip, with identity switches marked (029).

The README numbers (IDF1, HOTA, ID switches) say HOW MUCH better a run is; this video shows WHERE. Every run is
drawn on the same frames, one tile each, left to right. Each box has its run's id and a colour per id.

Identity switches are found with the ground truth, the way the CLEAR MOT metrics count them (``clear_matches``: a
player keeps last frame's match while it still overlaps, the rest is matched by IoU >= 0.5; the totals agree with
the ``IDSW`` of TrackEval); when a ground-truth player is matched to a
DIFFERENT run id than the last time he was matched, that run has switched his identity. Each switch is one of
three kinds, which are very different errors (``classify``):

* ``new``      the player gets an id the run had never used: a FRAGMENT (he is now counted twice). Yellow.
* ``mate``     he gets an id that belonged to a TEAM-MATE: two players of the same kit swapped. Orange.
* ``opponent`` he gets an id that belonged to a player of the OTHER team: a cross-team swap. Red.

The box is drawn in that colour for one second, and the tile header counts each kind so far. Teams come from
the jersey colours of the ground-truth players (``hooptrack.eval.gt_teams``, as the 016 benchmark), which can be
wrong for a few players: a switch whose previous owner has no team is counted as ``mate``-or-``opponent`` unknown
(``unknown``, grey). ``scripts/switch_report.py`` gives the same counts per minute for every clip.

Usage (repo root):
    python scripts/render_compare.py --config configs/player.yaml --clip v_5ekaksddqrc_c003 `
        --runs 011b=outputs/runs/011b_no_referee_filter/segmentation 020=outputs/runs/020_siglip_k2_15 `
        --out outputs/videos/compare_v_5ekaksddqrc_c003.mp4
A run may be ``<run>/<clip>/results/<clip>.txt`` or ``<run>/results/<clip>.txt`` (as ``scripts/eval_hota.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hota import result_file  # noqa: E402  (same run layouts)

from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import list_sequences  # noqa: E402
from hooptrack.eval.gt_teams import player_teams  # noqa: E402
from hooptrack.tracking.matching import iou_matrix, linear_assignment  # noqa: E402
from hooptrack.viz import id_color  # noqa: E402

RED = (0, 0, 255)
KIND_COLOUR = {"new": (0, 220, 255), "mate": (0, 140, 255), "opponent": (0, 0, 255), "unknown": (160, 160, 160)}
KINDS = ("new", "mate", "opponent", "unknown")


def load_mot(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """MOT results file -> {frame: (ids, TLBR boxes)}."""
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    out = {}
    for f in np.unique(raw[:, 0]).astype(int):
        r = raw[raw[:, 0] == f]
        out[f] = (r[:, 1].astype(int), np.c_[r[:, 2:4], r[:, 2:4] + r[:, 4:6]])
    return out


def switch_events(pred: dict, gt: dict, length: int, teams: dict[int, int] | None = None,
                  iou_min: float = 0.5) -> dict[int, dict[int, str]]:
    """{frame: {run id: kind}} for every identity switch (kinds in the module docstring)."""
    events: dict[int, dict[int, str]] = {}
    for r in switch_records(pred, gt, length, teams, iou_min):
        events.setdefault(r["frame"], {})[r["run_id"]] = r["kind"]
    return events


def switch_records(pred: dict, gt: dict, length: int, teams: dict[int, int] | None = None,
                   iou_min: float = 0.5) -> list[dict]:
    """One dict per identity switch: frame, ground-truth player, new run id, previous run id, kind.

    ``last``  : ground-truth player -> the run id he was last matched to (a change = a switch);
    ``owner`` : run id -> the ground-truth player it was last matched to (who "had" that id before).
    """
    teams = teams or {}
    last: dict[int, int] = {}
    owner: dict[int, int] = {}
    records: list[dict] = []
    for f in range(1, length + 1):
        if f not in pred or f not in gt:
            continue
        pids, pboxes = pred[f]
        gids, gboxes = gt[f]
        gboxes = np.asarray(gboxes, float)
        iou = iou_matrix(gboxes, pboxes)
        matches = clear_matches(gids, pids, iou, last, iou_min)
        for g, p in matches:
            gid, rid = int(gids[g]), int(pids[p])
            if gid in last and last[gid] != rid:
                records.append({"frame": f, "gt_id": gid, "run_id": rid, "previous_run_id": last[gid],
                                "kind": classify(gid, owner.get(rid), teams)})
            last[gid] = rid
            owner[rid] = gid
    return records


def clear_matches(gids, pids, iou: np.ndarray, last: dict[int, int], iou_min: float) -> list[tuple[int, int]]:
    """Ground truth <-> run boxes of one frame, as the CLEAR MOT metrics match them (MOTA's ID switches).

    1. A ground-truth player KEEPS the run id he was matched to in his last frame if that id is still on him
       (IoU >= ``iou_min``). Without this rule, two overlapping players (a defender on an attacker) can be
       matched one way in a frame and the other way in the next with no change in the tracker's output, and each
       flip would be counted as two switches.
    2. The remaining boxes are matched by the Hungarian algorithm on IoU (pairs below ``iou_min`` refused).
    """
    col = {int(p): j for j, p in enumerate(pids)}
    matches, used_g, used_p = [], set(), set()
    for g, gid in enumerate(gids):
        j = col.get(last.get(int(gid), -1))
        if j is not None and j not in used_p and iou[g, j] >= iou_min:
            matches.append((g, j))
            used_g.add(g)
            used_p.add(j)
    rest_g = [g for g in range(len(gids)) if g not in used_g]
    rest_p = [j for j in range(len(pids)) if j not in used_p]
    if rest_g and rest_p:
        m, _, _ = linear_assignment(1.0 - iou[np.ix_(rest_g, rest_p)], 1.0 - iou_min)
        matches += [(rest_g[g], rest_p[j]) for g, j in m]
    return matches


def classify(gid: int, previous_owner: int | None, teams: dict[int, int]) -> str:
    """Kind of switch: the player ``gid`` now carries a run id last carried by ``previous_owner``."""
    if previous_owner is None or previous_owner == gid:
        return "new"                                   # an id never seen on anyone else: a fragment
    a, b = teams.get(gid), teams.get(previous_owner)
    if a is None or b is None or a < 0 or b < 0:
        return "unknown"
    return "mate" if a == b else "opponent"


def draw_tile(img, name, frame_pred, flashing: dict[int, str], counts: dict[str, int], ids_so_far: int,
              scale: float):
    """One run's tile: boxes, ids, coloured boxes for recent switches, header with the counters per kind."""
    out = img.copy()
    if frame_pred is not None:
        for rid, (x1, y1, x2, y2) in zip(*frame_pred, strict=True):
            kind = flashing.get(int(rid))
            col = KIND_COLOUR[kind] if kind else id_color(int(rid))
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, col, 6 if kind else 2)
            label = f"#{rid} {kind.upper()}" if kind else f"#{rid}"
            cv2.putText(out, label, (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    out = cv2.resize(out, None, fx=scale, fy=scale)
    cv2.rectangle(out, (0, 0), (out.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(out, f"{name} | ids used: {ids_so_far}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    x = 8
    for k in KINDS:                                    # coloured counters: new / mate / opponent / unknown
        txt = f"{k}: {counts[k]}"
        cv2.putText(out, txt, (x, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, KIND_COLOUR[k], 2)
        x += cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] + 18
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=path pairs, drawn left to right")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="size of each tile (0.5 = half the frame)")
    ap.add_argument("--flash", type=float, default=1.0, help="seconds a switch stays red")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = apply_overrides(cfg, ["data.only=null"])["data"]
    seqs = [s for s in list_sequences(d["root"], d["split"], d.get("sport"), d["splits_dir"]) if s.name == args.clip]
    if not seqs:
        raise SystemExit(f"clip {args.clip} not found in split {d['split']}")
    seq = seqs[0]
    gt = seq.load_gt()
    teams = player_teams(seq)                          # ground-truth player -> kit (jersey colour)
    runs = dict(r.split("=", 1) for r in args.runs)
    preds, events = {}, {}
    for name, path in runs.items():
        f = result_file(Path(path), seq.name)
        if f is None:
            raise SystemExit(f"{name}: no results for {seq.name} in {path}")
        preds[name] = load_mot(f)
        events[name] = switch_events(preds[name], gt, seq.length, teams)
        kinds = [k for v in events[name].values() for k in v.values()]
        print(f"{name}: {len(kinds)} identity switches | " + ", ".join(f"{k} {kinds.count(k)}" for k in KINDS))

    flash = int(round(args.flash * seq.fps))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer, seen = None, {n: set() for n in runs}
    counts = {n: dict.fromkeys(KINDS, 0) for n in runs}
    for f in range(1, seq.length + 1):
        img = cv2.imread(str(seq.frame_path(f)))
        tiles = []
        for name in runs:
            for kind in events[name].get(f, {}).values():
                counts[name][kind] += 1
            fp = preds[name].get(f)
            if fp is not None:
                seen[name].update(int(i) for i in fp[0])
            recent: dict[int, str] = {}
            for k in range(max(1, f - flash + 1), f + 1):          # switches of the last `flash` frames
                recent.update(events[name].get(k, {}))
            tiles.append(draw_tile(img, name, fp, recent, counts[name], len(seen[name]), args.scale))
        canvas = np.hstack(tiles)
        if writer is None:
            writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), seq.fps,
                                     (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
    writer.release()
    print(f"Saved {args.out} ({seq.length} frames)")


if __name__ == "__main__":
    main()
