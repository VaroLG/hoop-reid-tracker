"""Homography for every frame of a clip from a few anchors (court chapter, C04).

1. Each anchor (``anchors_<sequence>.json``) is fitted: hand-placed points ->
   homography -> polished with the painted lines.
2. From each anchor, a **chain** is propagated forwards to the next anchor (or
   the end) and backwards to the previous one (or frame 1), with optical flow on
   the floor + line refinement (``hooptrack.court.propagate``). Between two
   anchors there are therefore TWO independent chains.
3. **Validation**: where two chains overlap, their disagreement (pixels and
   metres on the court) measures the error with no ground truth at all. Two
   chains started from different hand annotations, carried hundreds of frames
   through camera motion, can only agree if both are right.
4. **Merge**: one homography per frame, highest line agreement wins
   (``merge_chains``).

Each chain is saved as it finishes and skipped if it already exists, so the
work can be split over several runs (``--only a300f``).

Usage:
    python scripts/court/propagate_homographies.py \
        --anchors experiments/court/C04_anchor_homographies/anchors_v_00HRwkvvjtQ_c001.json \
        --out outputs/court_homographies
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from _clip import Clip

from hooptrack.court.anchors import AnchorFile, fit_anchor
from hooptrack.court.propagate import Chain, ChainStep, disagreement, merge_chains, track_homography, trusted


def chain_plan(af: AnchorFile, length: int) -> list[tuple[str, int, int]]:
    """(name, from anchor frame, to frame) for every chain."""
    fr = sorted(a.frame for a in af.anchors)
    plan = []
    for i, f in enumerate(fr):
        plan.append((f"a{f}b", f, fr[i - 1] if i else 1))
        plan.append((f"a{f}f", f, fr[i + 1] if i + 1 < len(fr) else length))
    return [p for p in plan if p[1] != p[2]]


def save_chain(path: Path, ch: Chain) -> None:
    np.savez(path, frames=np.array([s.frame for s in ch.steps]), H=np.array([s.H for s in ch.steps]),
             agreement=np.array([s.agreement for s in ch.steps]),
             flow_inliers=np.array([s.flow_inliers for s in ch.steps]), anchor_frame=ch.anchor_frame)


def load_chain(path: Path) -> Chain:
    path = Path(path)
    z = np.load(path)
    steps = [ChainStep(int(f), H, float(a), int(n))
             for f, H, a, n in zip(z["frames"], z["H"], z["agreement"], z["flow_inliers"], strict=True)]
    return Chain(path.stem, int(z["anchor_frame"]), steps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/court_homographies"))
    ap.add_argument("--only", nargs="*", help="run only these chains (e.g. a300f a900b)")
    ap.add_argument("--force", action="store_true", help="recompute chains that already exist")
    args = ap.parse_args()

    meta = json.loads(args.anchors.read_text())
    af = AnchorFile.load(args.anchors)
    spec, below = af.spec, meta.get("ignore_below")
    clip = Clip(af.sequence)
    out = args.out / af.sequence
    out.mkdir(parents=True, exist_ok=True)

    # 1. anchors
    anchor_H, anchor_A, anchor_log = {}, {}, {}
    for a in af.anchors:
        H, a0, a1 = fit_anchor(clip.frame(a.frame), a, spec, clip.people(a.frame), below)
        anchor_H[a.frame], anchor_A[a.frame] = H, a1
        anchor_log[a.frame] = {"points": len(a.points), "agreement_hand": round(a0, 3),
                               "agreement_refined": round(a1, 3)}
        print(f"anchor {a.frame}: {len(a.points)} points, line agreement {a0:.2f} -> {a1:.2f} after refinement")

    # 2. chains
    plan = chain_plan(af, clip.length)
    for name, f0, f1 in plan:
        path = out / f"{name}.npz"
        if (path.exists() and not args.force) or (args.only and name not in args.only):
            continue
        t0 = time.time()
        steps = track_homography(clip.frames(f0, f1), anchor_H[f0], spec, clip.people, below)
        ch = Chain(name, f0, [ChainStep(f0, anchor_H[f0], anchor_A[f0], 0), *steps])
        save_chain(path, ch)
        print(f"chain {name}: {f0} -> {f1}, {len(steps)} frames in {time.time() - t0:.0f} s, "
              f"agreement median {np.median([s.agreement for s in steps]):.2f}")

    missing = [n for n, *_ in plan if not (out / f"{n}.npz").exists()]
    if missing:
        print("chains still to run:", missing)
        return

    # 3. validation between independent chains (forward from one anchor vs backward from the next)
    raw = {n: load_chain(out / f"{n}.npz") for n, *_ in plan}
    chains = {n: trusted(c) for n, c in raw.items()}        # drop what follows a loss of tracking
    trust = {n: {"frames": len(raw[n].steps), "trusted_until": c.steps[-1].frame} for n, c in chains.items()}
    fr = sorted(anchor_H)
    consistency = {}
    for a, b in zip(fr[:-1], fr[1:], strict=True):
        fa = {s.frame: s.H for s in chains[f"a{a}f"].steps}
        fb = {s.frame: s.H for s in chains[f"a{b}b"].steps}
        common = sorted(set(fa) & set(fb) - {a, b})
        if not common:
            consistency[f"a{a}f vs a{b}b"] = {"frames": 0}
            continue
        d = np.array([disagreement(fa[f], fb[f], spec, (1280, below or 720)) for f in common])
        consistency[f"a{a}f vs a{b}b"] = {
            "frames": len(common),
            "median_px": round(float(np.nanmedian(d[:, 0])), 1),
            "p90_px": round(float(np.nanpercentile(d[:, 0], 90)), 1),
            "median_m": round(float(np.nanmedian(d[:, 1])), 3), "p90_m": round(float(np.nanpercentile(d[:, 1], 90)), 3),
        }

    # 4. merge
    best = merge_chains(list(chains.values()), [tuple(r) for r in meta.get("nearest_ranges", [])])
    frames = np.array(list(best))
    agree = np.array([s.agreement for s in best.values()])
    owner = {id(s): n for n, ch in chains.items() for s in ch.steps}
    chosen = {f: owner[id(s)] for f, s in best.items()}
    np.savez(out / "homographies.npz", frames=frames, H=np.array([s.H for s in best.values()]),
             agreement=agree, chain=np.array([chosen[int(f)] for f in frames]), league=af.league)
    report = {
        "sequence": af.sequence, "league": af.league, "anchors": anchor_log, "chains": trust,
        "chain_consistency": consistency,
        "frames": int(len(frames)), "clip_length": clip.length,
        "frames_per_chain": {n: int((np.array([chosen[int(f)] for f in frames]) == n).sum()) for n in chains},
        "agreement_median": round(float(np.median(agree)), 3),
        "agreement_share_ge_0.5": round(float((agree >= 0.5).mean()), 3),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
