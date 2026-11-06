"""B07 · WASB (BMVC 2023), the published state of the art for sports balls, on our test clips, with its own
basketball weights and no training (ball chapter).

Why
---
Our ball detector (B03) is a YOLO box detector on ONE frame, trained on 1,245 frames. WASB ("Widely Applicable Strong
Baseline for Sports Ball Detection and Tracking", Tarashima et al., BMVC 2023; code MIT,
https://github.com/nttcom/WASB-SBDT) differs in three ways: it predicts a **heatmap** (a point, not a box), it looks at
**3 consecutive frames** at once (motion is in the input: a still look-alike does not move), and it keeps a **high
internal resolution**. Its basketball weights were trained on ~275k NBA frames. Run as is on our clips, it tells how
far our approach is from the state of the art, as Deep-EIoU did for the players (024).

What it does
------------
The 1280 x 720 frames are resized to 512 x 288 (WASB's input), normalised as in WASB (ImageNet mean / std), and every
window of 3 consecutive frames gives 3 heatmaps; each frame's heatmaps (up to 3 windows) are averaged. As in WASB's
post-processing, the averaged heatmap is thresholded at 0.5 and every connected blob is a candidate: position = the
heatmap-weighted centre (back in 1280 x 720 pixels), confidence = the blob's peak. The ball diameter, needed by our
tracker, comes from the players around it (``hooptrack.ball.labels.ball_diameter``, with the 038 player tracks).

Output: ``<out>/<clip>.csv`` (frame, cx, cy, d, conf), the format of ``b05_detect_all.py``, so B04/B05's scoring and
tracker run unchanged on it.

Setup (once): ``git clone https://github.com/nttcom/WASB-SBDT third_party/WASB-SBDT`` (not committed) and the WASB
basketball weights from its MODEL_ZOO.md into ``weights/wasb_basketball_best.pth.tar``.

Usage (repo root; CPU, ~15-25 min for the 4 test clips):
    python -u scripts/ball/b07_wasb_probe.py --out outputs/ball/dets_wasb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_hota import result_file  # noqa: E402

from hooptrack.ball.labels import ball_diameter  # noqa: E402
from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import load_mot_txt  # noqa: E402

TEST_CLIPS = ["v_00HRwkvvjtQ_c001", "v_4r8QL_wglzQ_c001", "v_5ekaksddqrc_c003", "v_BgwzTUxJaeU_c012"]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class AttrDict(dict):
    """WASB's code reads its config both as ``cfg["MODEL"]`` and as ``cfg.MODEL`` (an OmegaConf object in WASB, via
    Hydra). This dict allows both, recursively, without installing Hydra / OmegaConf."""

    def __init__(self, d: dict) -> None:
        super().__init__({k: AttrDict(v) if isinstance(v, dict) else v for k, v in d.items()})

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def load_wasb(src: Path, weights: Path):
    """WASB's HRNet from its own code and config, with its checkpoint, on the CPU (or CUDA if present)."""
    import torch

    sys.path.insert(0, str(src))
    from models.hrnet import HRNet                    # noqa: E402  (WASB's code, third_party/)

    cfg = AttrDict(yaml.safe_load((src / "configs" / "model" / "wasb.yaml").read_text()))
    model = HRNet(cfg)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}     # saved from DataParallel
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.eval().to(device), cfg, device


def blobs(hm: np.ndarray, thresh: float = 0.5) -> list[tuple[float, float, float]]:
    """WASB's post-processing: connected blobs above ``thresh`` -> (x, y, peak) in heatmap pixels."""
    if hm.max() <= thresh:
        return []
    n, labels = cv2.connectedComponents((hm > thresh).astype(np.uint8))
    out = []
    for m in range(1, n):
        ys, xs = np.nonzero(labels == m)
        w = hm[ys, xs]
        out.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.max())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--wasb-src", type=Path, default=Path("third_party/WASB-SBDT/src"))
    ap.add_argument("--weights", type=Path, default=Path("weights/wasb_basketball_best.pth.tar"))
    ap.add_argument("--players", type=Path, default=Path("outputs/runs/038_continuity"), help="for the ball size")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clips", nargs="*", default=TEST_CLIPS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/dets_wasb"))
    args = ap.parse_args()

    import torch

    model, cfg, device = load_wasb(args.wasb_src, args.weights)
    W, H, F = cfg["inp_width"], cfg["inp_height"], cfg["frames_in"]
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            [f"data.split={args.split}", "data.only=null"]))}
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"WASB on {device}, input {W}x{H}, {F} frames")
    for clip in args.clips:
        dst = args.out / f"{clip}.csv"
        if dst.is_file():
            print(f"{clip}: already done")
            continue
        seq = seqs[clip]
        n = seq.length
        first = cv2.imread(str(seq.frame_path(1)))
        h0, w0 = first.shape[:2]
        sx, sy = w0 / W, h0 / H                         # 1280 x 720 -> 512 x 288 is a plain resize (same 16:9)

        def frame(f: int) -> np.ndarray:
            img = cv2.cvtColor(cv2.imread(str(seq.frame_path(f))), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            return ((img - MEAN) / STD).transpose(2, 0, 1)             # 3 x H x W

        acc = np.zeros((n, H, W), np.float32)            # sum of the heatmaps of each frame
        cnt = np.zeros(n, np.float32)
        cache: dict[int, np.ndarray] = {}
        starts = list(range(1, n - F + 2))               # window = frames s .. s+F-1
        for b in range(0, len(starts), args.batch):
            chunk = starts[b:b + args.batch]
            x = []
            for s in chunk:
                for f in range(s, s + F):
                    if f not in cache:
                        cache[f] = frame(f)
                x.append(np.concatenate([cache[f] for f in range(s, s + F)], axis=0))
            for f in [f for f in cache if f < chunk[0]]:
                del cache[f]
            with torch.inference_mode():
                hm = torch.sigmoid(model(torch.from_numpy(np.stack(x)).to(device))[0]).cpu().numpy()
            for i, s in enumerate(chunk):
                for j in range(F):
                    acc[s - 1 + j] += hm[i, j]
                    cnt[s - 1 + j] += 1
            if b // args.batch % 25 == 0:
                print(f"{clip}: {chunk[-1]}/{n}")
        players = seq.load_gt() if str(args.players) == "gt" else load_mot_txt(result_file(args.players, clip))
        lines = ["frame,cx,cy,d,conf"]
        for k in range(n):
            if not cnt[k]:
                continue
            for x, y, peak in blobs(acc[k] / cnt[k]):
                X, Y = x * sx, y * sy
                d = ball_diameter(X, Y, players.get(k + 1, (None, np.zeros((0, 4))))[1])
                lines.append(f"{k + 1},{X:.1f},{Y:.1f},{d:.1f},{peak:.4f}")
        tmp = dst.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(dst)
        print(f"{clip}: {len(lines) - 1} candidates in {n} frames")


if __name__ == "__main__":
    main()
