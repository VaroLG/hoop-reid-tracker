"""049 (scripts/sam2_track.py): the association of boxes to SAM 2 masks, with synthetic masks (no GPU)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sam2_track as st  # noqa: E402


def mask(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), bool)
    m[y1:y2, x1:x2] = True
    return m


def test_share_of_a_mask_inside_a_box():
    m = mask(100, 200, 20, 20, 60, 80)[None]                    # 40 x 60 mask
    share = st.mask_box_share(m, np.array([[0, 0, 100, 100], [40, 0, 200, 100], [150, 0, 200, 100]]), step=1)
    assert share[0, 0] == 1.0 and abs(share[0, 1] - 0.5) < 1e-6 and share[0, 2] == 0.0


def test_identities_follow_the_masks_through_a_crossing():
    """Two players cross: the detector gives the boxes in either order, the masks follow the people."""
    h, w = 120, 300
    book = st.Book(min_share=0.5, max_lost=10)
    a0, b0 = np.array([20, 20, 60, 100]), np.array([200, 20, 240, 100])
    book.step(1, np.array([a0, b0]), [], None)                  # frame 1: two new players
    prompts = book.prompts(1)
    assert len(prompts) == 2 and book.pending == {}
    id_a, id_b = (tid for tid, b in prompts.items() if b[0] == 20), (tid for tid, b in prompts.items() if b[0] == 200)
    id_a, id_b = next(id_a), next(id_b)
    for f, xa, xb in ((2, 80, 160), (3, 120, 120), (4, 160, 80), (5, 200, 20)):   # A moves right, B left
        boxes = np.array([[xb, 20, xb + 40, 100], [xa, 20, xa + 40, 100]])         # listed in the other order
        masks = np.stack([mask(h, w, xa, 25, xa + 40, 95), mask(h, w, xb, 25, xb + 40, 95)])
        book.step(f, boxes, [id_a, id_b], masks)
    last = {tid: b for (g, tid, b, _) in book.rows if g == 5}
    assert last[id_a][0] == 200 and last[id_b][0] == 20          # each id stayed on its player


def test_unclaimed_box_becomes_a_pending_track_then_a_player():
    book = st.Book(min_share=0.5, max_lost=10)
    book.step(1, np.array([[0, 0, 10, 20]]), [], None)
    new = np.array([[50, 0, 60, 20]])
    book.step(2, np.vstack([[0, 0, 10, 20], new]), list(book.prompts(1)), np.stack([mask(30, 80, 0, 0, 10, 20)]))
    book.step(3, np.vstack([[0, 0, 10, 20], new + 2]), list(book.players), np.stack([mask(30, 80, 0, 0, 10, 20)]))
    ids3 = [tid for (g, tid, _, _) in book.rows if g == 3]
    ids2 = [tid for (g, tid, _, _) in book.rows if g == 2]
    assert ids2 == ids3 and len(set(ids3)) == 2                 # the new box kept its pending id
    assert set(book.prompts(3)) == set(ids3) and book.pending == {}
