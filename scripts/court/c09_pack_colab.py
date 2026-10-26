"""Pack the C09 court-keypoint dataset into one zip for Colab (court chapter, C09).

``data/court_kp/data.yaml`` lists v22 as ``../court_keypoints/yolo_v22``, so the
zip keeps both folders side by side, as under ``data/``:

    court_kp/{deepsport,sportsmot,synth}/{images,labels}, court_kp/data.yaml
    court_keypoints/yolo_v22/{images,labels}, court_keypoints/yolo_v22/data.yaml

Previews, logs and machine-specific ``data_resolved.yaml`` are left out. JPEGs
are already compressed: they are stored, not deflated (faster, same size).

Usage (repo root, after c09_build_keypoints.py ... yaml):
    python scripts/court/c09_pack_colab.py          # -> data/court_kp.zip, upload to MyDrive/hooptrack/
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

DATA = Path("data")
PARTS = [Path("court_kp") / s for s in ("deepsport", "sportsmot", "synth")] + [Path("court_keypoints/yolo_v22")]
YAMLS = [Path("court_kp/data.yaml"), Path("court_keypoints/yolo_v22/data.yaml")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "court_kp.zip")
    args = ap.parse_args()
    n = 0
    with zipfile.ZipFile(args.out, "w") as z:
        for rel in YAMLS:
            z.write(DATA / rel, rel.as_posix(), zipfile.ZIP_DEFLATED)
        for part in PARTS:
            for sub in ("images", "labels"):
                for p in sorted((DATA / part / sub).rglob("*")):
                    if p.is_file():
                        mode = zipfile.ZIP_STORED if p.suffix.lower() in (".jpg", ".png") else zipfile.ZIP_DEFLATED
                        z.write(p, p.relative_to(DATA).as_posix(), mode)
                        n += 1
    print(f"wrote {args.out}: {n} files, {args.out.stat().st_size / 1e6:.0f} MB -> upload to MyDrive/hooptrack/")


if __name__ == "__main__":
    main()
