"""Homographies for a whole clip with NO annotation: automatic anchors + chain validation (court chapter, C05).

C04 needed a person to place ~20 points on 3 frames. Here those anchors are
found by the machine and *proved* by the video itself:

1. **Candidates.** Every ``stride`` frames, ``search.search_lines`` proposes the
   best court homography from the straight lines of that frame alone. Many
   frames give none (no line pair in each family) — that is fine.
2. **Confirmation.** A single frame can be fooled (C03); a video is much
   harder to fool. Consecutive candidates are linked by two chains
   (``propagate.track_homography``): forwards from the earlier one, backwards
   from the later one. If they meet at the same court (median disagreement on
   the visible court below ``agree_m``), BOTH candidates are confirmed: two
   independent single-frame guesses, carried through camera motion, landed on
   the same answer. Candidates that agree with no neighbour are dropped and
   their neighbours are linked directly; this repeats until stable.
3. **Merge.** Confirmed anchors play the role of C04's hand anchors: every
   frame takes the trusted chain with the best line agreement.

No step uses ground truth; the C04 hand-made homographies are only used to
MEASURE the result (scripts/court/auto_homographies.py).

Nothing here imports the tracker: frames, people boxes and the floor hull
come in as callables from the calling script.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .leagues import CourtSpec
from .propagate import Chain, ChainStep, disagreement, merge_chains, track_homography, trusted
from .refine import line_evidence
from .search import Candidate, search_lines

FrameFn = Callable[[int], np.ndarray]
BoxesFn = Callable[[int], np.ndarray]


@dataclass
class AutoResult:
    candidates: dict[int, Candidate]
    confirmed: list[int]
    links: dict[tuple[int, int], dict] = field(default_factory=dict)     # (a, b) -> agreement stats
    homographies: dict[int, ChainStep] = field(default_factory=dict)
    chains: dict[str, Chain] = field(default_factory=dict)
    conflict: list[int] = field(default_factory=list)       # frames where two unconfirmed chains disagree


def evidence(img: np.ndarray, people: np.ndarray | None, hull: np.ndarray | None, ignore_below: int | None,
             evidence_kw: dict | None = None):
    ev = line_evidence(img, hull, people, **(evidence_kw or {}))
    if ignore_below is not None:
        ev.valid[ignore_below:, :] = 0
    return ev


def find_candidates(frame: FrameFn, n_frames: int, spec: CourtSpec, people: BoxesFn | None = None,
                    hull: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
                    ignore_below: int | None = None, stride: int = 25, min_score: float = 100.0,
                    log: Callable[[str], None] | None = None, cache: dict | None = None,
                    on_step: Callable[[], None] | None = None,
                    evidence_kw: dict | None = None) -> dict[int, Candidate]:
    """Best single-frame candidate every ``stride`` frames (only those scoring >= ``min_score``).

    ``cache`` (frame -> best Candidate or None) lets a long run be split into
    several shorter ones; ``on_step`` is called after each new frame (e.g. to save it).
    """
    cache = {} if cache is None else cache
    for f in range(1, n_frames + 1, stride):
        if f not in cache:
            img = frame(f)
            boxes = people(f) if people else None
            ev = evidence(img, boxes, hull(f, img) if hull else None, ignore_below, evidence_kw)
            cands = search_lines(ev, spec, img.shape[1::-1])
            cache[f] = cands[0] if cands else None
            if log:
                log(f"frame {f}: " + (f"score {cands[0].score:.0f}" if cands else "no candidate"))
            if on_step:
                on_step()
    return {f: c for f, c in cache.items() if c is not None and c.score >= min_score and (f - 1) % stride == 0}


def _chain(frame: FrameFn, a: int, b: int, H: np.ndarray, spec: CourtSpec, people, ignore_below, agreement: float,
           evidence_kw: dict | None = None):
    step = 1 if b >= a else -1
    steps = track_homography(((f, frame(f)) for f in range(a, b + step, step)), H, spec, people, ignore_below,
                             evidence_kw=evidence_kw)
    return Chain(f"a{a}{'f' if step > 0 else 'b'}", a, [ChainStep(a, H, agreement, 0), *steps])


def link(frame: FrameFn, a: int, b: int, Ha: np.ndarray, Hb: np.ndarray, spec: CourtSpec, people=None,
         ignore_below=None, agreement=(1.0, 1.0), image_size=(1280, 720),
         evidence_kw: dict | None = None) -> tuple[dict, Chain, Chain]:
    """Chains a -> b and b -> a; their disagreement over the frames both trust."""
    fwd = trusted(_chain(frame, a, b, Ha, spec, people, ignore_below, agreement[0], evidence_kw))
    bwd = trusted(_chain(frame, b, a, Hb, spec, people, ignore_below, agreement[1], evidence_kw))
    fa = {s.frame: s.H for s in fwd.steps}
    fb = {s.frame: s.H for s in bwd.steps}
    common = sorted(set(fa) & set(fb))
    size = (image_size[0], ignore_below or image_size[1])
    d = np.array([disagreement(fa[f], fb[f], spec, size)[1] for f in common]) if common else np.array([np.nan])
    stats = {"frames": len(common), "median_m": float(np.nanmedian(d)) if len(d) else float("nan"),
             "covered": len(common) / (abs(b - a) + 1)}
    return stats, fwd, bwd


def confirm(frame: FrameFn, n_frames: int, cands: dict[int, Candidate], spec: CourtSpec, people=None,
            ignore_below=None, agree_m: float = 0.5, min_cover: float = 0.8, image_size=(1280, 720),
            log: Callable[[str], None] | None = None, cache: dict | None = None,
            on_step: Callable[[], None] | None = None, evidence_kw: dict | None = None) -> AutoResult:
    """Keep the candidates confirmed by a neighbour; build every chain between confirmed anchors.

    ``cache`` maps (a, b) or ("end", a, b) to already computed links / chains (resumable runs).
    """
    alive = sorted(cands)
    cache = {} if cache is None else cache
    links, chains = {}, {}
    changed = True
    while changed and len(alive) >= 2:
        changed = False
        for a, b in zip(alive[:-1], alive[1:], strict=True):
            if (a, b) in links:
                continue
            if (a, b) not in cache:
                cache[(a, b)] = link(frame, a, b, cands[a].H, cands[b].H, spec, people, ignore_below,
                                     (cands[a].agreement, cands[b].agreement), image_size, evidence_kw)
                if on_step:
                    on_step()
            stats, fwd, bwd = cache[(a, b)]
            stats = dict(stats)
            stats["ok"] = bool(stats["median_m"] < agree_m and stats["covered"] >= min_cover)
            links[(a, b)] = stats
            chains[(a, b)] = (fwd, bwd)
            if log:
                log(f"link {a}-{b}: {stats['median_m']:.2f} m over {stats['frames']} frames -> "
                    f"{'agree' if stats['ok'] else 'disagree'}")
        ok_nodes = {n for (a, b), s in links.items() if s["ok"] and a in alive and b in alive for n in (a, b)}
        new_alive = [n for n in alive if n in ok_nodes]
        if new_alive != alive:
            alive, changed = new_alive, True
    res = AutoResult(cands, alive, links)
    if not alive:
        return res
    # chains used for the final merge: between consecutive confirmed anchors, plus the two ends of the clip
    use, conflict = [], set()
    size = (image_size[0], ignore_below or image_size[1])
    for a, b in zip(alive[:-1], alive[1:], strict=True):
        fwd, bwd = chains[(a, b)]
        use += [fwd, bwd]
        if not links[(a, b)]["ok"]:
            # Both ends are confirmed by their OTHER neighbours, but these two chains do not meet
            # (fast pan, few lines): where they both reach a frame and disagree, at least one is
            # wrong and nothing says which. Those frames get no homography (the tracker falls back
            # to the colour mask) rather than a coin flip.
            fa = {st.frame: st.H for st in fwd.steps}
            fb = {st.frame: st.H for st in bwd.steps}
            conflict |= {f for f in set(fa) & set(fb)
                         if not disagreement(fa[f], fb[f], spec, size)[1] < agree_m}
    first, last = alive[0], alive[-1]
    for a, b in ((first, 1), (last, n_frames)):
        if a == b:
            continue
        if ("end", a, b) not in cache:
            cache[("end", a, b)] = trusted(_chain(frame, a, b, cands[a].H, spec, people, ignore_below,
                                                  cands[a].agreement, evidence_kw))
            if on_step:
                on_step()
        use.append(cache[("end", a, b)])
    res.chains = {f"{c.name}": c for c in use}
    merged = merge_chains(use)
    res.conflict = sorted(conflict)
    res.homographies = {f: st for f, st in merged.items() if f not in conflict}
    return res
