"""049 · SAM 2 carries each player's identity: a mask per player propagated through the clip (GPU, Colab).

Why: the online tracker (011b) keeps identities with boxes, motion and appearance, and swaps them when players
overlap; 038 then cuts every track at every crossing and re-assigns the pieces by appearance. 046 showed that
fixing the crossings AFTER the fact does not help that assignment. This experiment changes the tracker itself: a
segmentation mask follows each player frame to frame (SAM 2 video, as SAM2MOT / SAMIDARE do), so a box in a crossing
belongs to the player whose mask it contains, not to the closest prediction.

What it does, per clip, with the SAME boxes as the online run (``--run``, e.g. 011b: players only, on the court),
so that only the identity mechanism changes (as 024-026 compared Deep-EIoU on the same boxes):

1. the clip is processed in windows of ``--window`` frames that share their first / last frame;
2. at the start of a window, every player being followed is prompted to SAM 2 with his box in that frame, plus a
   new player for every box nobody owns;
3. SAM 2 propagates the masks through the window; in every frame the boxes are given to the masks they contain
   (share of the mask inside the box, Hungarian, at least ``--min-share``);
4. a box no mask claims continues a short IoU track (``pending``) and becomes a SAM 2 player at the next window;
   a player whose mask finds no box for ``--max-lost`` frames is dropped.

Output: ``<out>/results/<clip>.txt`` (MOT), the layout of ``hooptrack track``: ``scripts/eval_hota.py`` scores it
and ``scripts/stitch_tracklets.py`` (038) can run on top.

Model: SAM 2.1 (``facebook/sam2.1-hiera-small`` by default) through Hugging Face ``transformers``; Apache-2.0.

Usage (repo root, a GPU; see notebooks/colab_sam2_tracking.ipynb):
    python scripts/sam2_track.py --run outputs/runs/011b/segmentation --out outputs/runs/049_sam2 --window 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import load_config  # noqa: E402
from hooptrack.data.sportsmot import load_mot_txt, write_mot_txt  # noqa: E402
from hooptrack.tracking.matching import iou_matrix  # noqa: E402


# ------------------------------------------------------------------------------------- association (no torch)
def mask_box_share(masks: np.ndarray, boxes: np.ndarray, step: int = 4) -> np.ndarray:
    """(players, boxes): the share of each player's mask that lies inside each box, and 0 where the mask covers
    less than 5 % of the box (a sliver of a mask in a big box is not a match). ``masks``: (P, H, W) bool."""
    if len(masks) == 0 or len(boxes) == 0:
        return np.zeros((len(masks), len(boxes)))
    m = masks[:, ::step, ::step].astype(np.float32)                    # coarse grid: plenty for shares
    ii = np.pad(m.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))       # integral images
    area = ii[:, -1, -1]
    h, w = m.shape[1:]
    out = np.zeros((len(m), len(boxes)))
    for j, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, float) / step):
        a, b = int(np.clip(np.floor(x1), 0, w)), int(np.clip(np.ceil(x2), 0, w))
        c, d = int(np.clip(np.floor(y1), 0, h)), int(np.clip(np.ceil(y2), 0, h))
        inside = ii[:, d, b] - ii[:, c, b] - ii[:, d, a] + ii[:, c, a]
        box_area = max(1.0, (b - a) * (d - c))
        out[:, j] = np.where(inside / box_area >= 0.05, inside / np.maximum(area, 1.0), 0.0)
    return out


def assign(share: np.ndarray, min_share: float) -> dict[int, int]:
    """{box index: player index}: one box per player and one player per box, best shares first (Hungarian)."""
    if share.size == 0:
        return {}
    rows, cols = linear_sum_assignment(-share)
    return {int(c): int(r) for r, c in zip(rows, cols, strict=True) if share[r, c] >= min_share}


class Book:
    """Who owns which box: SAM 2 players (followed by masks) and pending tracks (short IoU tracks of the boxes no
    mask claims, which become SAM 2 players at the next window)."""

    def __init__(self, min_share: float, max_lost: int, pending_iou: float = 0.3) -> None:
        self.min_share, self.max_lost, self.pending_iou = min_share, max_lost, pending_iou
        self.next_id = 1
        self.players: dict[int, int] = {}        # id -> frames since its mask last owned a box
        self.pending: dict[int, np.ndarray] = {}  # id -> its last box
        self.last_box: dict[int, np.ndarray] = {}
        self.rows: list[tuple[int, int, np.ndarray, float]] = []

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def step(self, f: int, boxes: np.ndarray, player_ids: list[int], masks: np.ndarray | None) -> None:
        """Frame ``f``: give its boxes to the players' masks, the rest to pending tracks, and write the rows."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        owner: dict[int, int] = {}
        if masks is not None and len(player_ids):
            for j, k in assign(mask_box_share(masks, boxes), self.min_share).items():
                owner[j] = player_ids[k]
        for pid in player_ids:                                   # lost counters
            self.players[pid] = 0 if pid in owner.values() else self.players.get(pid, 0) + 1
        free = [j for j in range(len(boxes)) if j not in owner]  # boxes no mask claims: pending tracks
        if free and self.pending:
            pids = list(self.pending)
            iou = iou_matrix(boxes[free], np.array([self.pending[p] for p in pids]))
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols, strict=True):
                if iou[r, c] >= self.pending_iou:
                    owner[free[r]] = pids[c]
        for j in range(len(boxes)):
            if j not in owner:
                owner[j] = self.new_id()
                self.pending[owner[j]] = boxes[j]
        for j, tid in owner.items():
            if tid in self.pending:
                self.pending[tid] = boxes[j]
            self.last_box[tid] = boxes[j]
            self.rows.append((f, tid, boxes[j], 1.0))

    def prompts(self, f: int) -> dict[int, np.ndarray]:
        """Players to prompt at the start of the window that begins at frame ``f`` (already assigned): every
        player and pending track that owns a box in that frame; players lost too long are dropped."""
        here = {tid: b for (g, tid, b, _) in self.rows if g == f}
        for pid in [p for p, lost in self.players.items() if lost > self.max_lost and p not in here]:
            del self.players[pid]
        for tid in here:                                          # pending tracks become SAM 2 players
            self.pending.pop(tid, None)
            self.players.setdefault(tid, 0)
        return here


# ------------------------------------------------------------------------------------------------- SAM 2 (GPU)
class Sam2Windows:
    """SAM 2.1 video, one short session per window (prompts on its first frame, propagated to its last)."""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.torch, self.device = torch, device
        self.dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device, dtype=self.dtype).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)

    def run(self, frames: list[np.ndarray], prompts: dict[int, np.ndarray]) -> dict[int, tuple[list[int], np.ndarray]]:
        """{window frame index: (player ids, (P, H, W) bool masks)} for every frame after the first."""
        torch = self.torch
        h, w = frames[0].shape[:2]
        ids = list(prompts)
        with torch.inference_mode():
            session = self.processor.init_video_session(video=frames, inference_device=self.device,
                                                        torch_dtype=self.dtype)
            boxes = [[[float(v) for v in prompts[i]] for i in ids]]
            self.processor.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=ids,
                                                           input_boxes=boxes)
            self.model(inference_session=session, frame_idx=0)
            out = {}
            for o in self.model.propagate_in_video_iterator(session):
                if o.frame_idx == 0:
                    continue
                m = self.processor.post_process_masks([o.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                out[int(o.frame_idx)] = ([int(i) for i in session.obj_ids], m[:, 0].cpu().numpy().astype(bool))
        return out


def track_clip(seq, boxes_by_frame: dict, sam, window: int, min_share: float, max_lost: int) -> list:
    import cv2

    book = Book(min_share, max_lost)
    first = 1
    book.step(first, boxes_by_frame.get(first, (None, np.zeros((0, 4))))[1], [], None)  # frame 1: all new
    s = first
    while s < seq.length:
        e = min(seq.length, s + window)
        prompts = book.prompts(s)
        frames = [cv2.cvtColor(cv2.imread(str(seq.frame_path(f))), cv2.COLOR_BGR2RGB) for f in range(s, e + 1)]
        res = sam.run(frames, prompts) if prompts else {}
        for k in range(1, e - s + 1):
            f = s + k
            ids, masks = res.get(k, ([], None))
            book.step(f, boxes_by_frame.get(f, (None, np.zeros((0, 4))))[1], ids, masks)
        s = e
    return book.rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="online run whose BOXES are used: <run>/results/<clip>.txt")
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.split=train")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every clip of --run")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--window", type=int, default=30, help="frames per SAM 2 session")
    ap.add_argument("--min-share", type=float, default=0.5, help="share of a mask inside a box to own it")
    ap.add_argument("--max-lost", type=int, default=50, help="frames a mask may own no box before it is dropped")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seqs = {s.name: s for s in _sequences(load_config(args.config, args.set + ["data.only=null"]))}
    clips = args.clips or sorted(p.stem for p in (args.run / "results").glob("*.txt"))
    sam = Sam2Windows(args.model, args.device)
    timing = {}
    for clip in clips:
        dst = args.out / "results" / f"{clip}.txt"
        if dst.is_file():
            print(f"{clip}: done")
            continue
        t0 = time.time()
        rows = track_clip(seqs[clip], load_mot_txt(args.run / "results" / f"{clip}.txt"), sam, args.window,
                          args.min_share, args.max_lost)
        write_mot_txt(dst, rows)
        timing[clip] = round(time.time() - t0, 1)
        print(f"{clip}: {len({r[1] for r in rows})} ids, {timing[clip]} s ({seqs[clip].length / timing[clip]:.1f} fps)")
    (args.out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                      "seconds": timing}, indent=2))


if __name__ == "__main__":
    main()
