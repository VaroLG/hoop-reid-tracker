"""SportsMOT reader (MOTChallenge-17 layout), restricted to one sport.

Layout after ``scripts/download_data.py``::

    data/SportsMOT/
      splits_txt/{basketball,football,volleyball,train,val,test}.txt
      dataset/{train,val}/<sequence>/
          img1/000001.jpg ...
          gt/gt.txt            # frame,id,x,y,w,h,conf,class,visibility
          seqinfo.ini

Only players on the court are annotated (no referees, coaches or crowd). A
COCO "person" detector will therefore produce extra boxes that count as false
positives: fine-tuning the detector on the train split fixes most of it
(see ``scripts/prepare_yolo_dataset.py``).
"""

from __future__ import annotations

import configparser
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sequence:
    name: str
    root: Path
    fps: float
    length: int
    width: int
    height: int
    im_dir: str = "img1"
    im_ext: str = ".jpg"

    @property
    def gt_path(self) -> Path:
        return self.root / "gt" / "gt.txt"

    @property
    def has_gt(self) -> bool:
        return self.gt_path.is_file()

    def frame_path(self, frame_idx: int) -> Path:
        """1-based frame index -> image path (MOT convention)."""
        return self.root / self.im_dir / f"{frame_idx:06d}{self.im_ext}"

    def frames(self) -> Iterator[tuple[int, Path]]:
        for i in range(1, self.length + 1):
            yield i, self.frame_path(i)

    def load_gt(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """frame -> (ids (G,), boxes (G, 4) TLBR)."""
        return load_mot_txt(self.gt_path)


def read_seqinfo(seq_dir: Path) -> Sequence:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    s = parser["Sequence"]
    return Sequence(
        name=s.get("name", seq_dir.name),
        root=seq_dir,
        fps=float(s.get("frameRate", 25)),
        length=int(s.get("seqLength")),
        width=int(s.get("imWidth")),
        height=int(s.get("imHeight")),
        im_dir=s.get("imDir", "img1"),
        im_ext=s.get("imExt", ".jpg"),
    )


def read_split_file(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def list_sequences(root: str | Path, split: str, sport: str | None, splits_dir: str | Path) -> list[Sequence]:
    """Sequences of ``split`` that belong to ``sport`` and exist on disk."""
    root, splits_dir = Path(root), Path(splits_dir)
    names = read_split_file(splits_dir / f"{split}.txt")
    if sport:
        allowed = set(read_split_file(splits_dir / f"{sport}.txt"))
        names = [n for n in names if n in allowed]
    seqs = []
    for name in names:
        seq_dir = root / split / name
        if (seq_dir / "seqinfo.ini").is_file():
            seqs.append(read_seqinfo(seq_dir))
    return seqs


# --------------------------------------------------------------------- MOT IO
def load_mot_txt(path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read a MOT file (gt or tracker output) into frame -> (ids, TLBR boxes)."""
    path = Path(path)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not path.is_file() or path.stat().st_size == 0:
        return out
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    # Some gt files flag "ignore" rows with conf == 0; SportsMOT uses 1 everywhere.
    if data.shape[1] >= 7:
        data = data[data[:, 6] != 0]
    for frame in np.unique(data[:, 0]).astype(int):
        rows = data[data[:, 0] == frame]
        xywh = rows[:, 2:6]
        tlbr = np.c_[xywh[:, :2], xywh[:, :2] + xywh[:, 2:4]]
        out[int(frame)] = (rows[:, 1].astype(int), tlbr)
    return out


def write_mot_txt(path: str | Path, rows: list[tuple[int, int, np.ndarray, float]]) -> None:
    """Write tracker output: one line per (frame, id, tlbr, score).

    MOT format: frame,id,x,y,w,h,score,-1,-1,-1
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for frame, tid, tlbr, score in rows:
            x1, y1, x2, y2 = (float(v) for v in tlbr)
            f.write(
                f"{frame},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{score:.4f},-1,-1,-1\n"
            )
