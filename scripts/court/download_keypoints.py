"""Download a public basketball-court keypoint dataset from Roboflow Universe.

Why this dataset (court chapter, C01)
-----------------------------------------
To map players onto a top-down court (and to build a court mask that works in
every arena) we need a homography per frame, and for that we need to find
known court points (corners, line intersections, ...) in the image. SportsMOT
has no such labels, so we start from a public dataset:

    roboflow-jvuqo / basketball-court-detection-2   (Roboflow's own, CC BY 4.0,
    ~850 broadcast frames, one "court" object with a fixed list of keypoints)

The licence (CC BY 4.0) allows reuse with attribution — see the README credits.

How the download works (no extra Python package)
------------------------------------------------
The official ``roboflow`` SDK pulls ``opencv-python-headless``, which clashes
with the ``opencv-python`` already pinned in requirements.txt (two packages
installing the same ``cv2`` module). Roboflow also has a plain REST API, so we
use the standard library instead:

    GET https://api.roboflow.com/<workspace>/<project>/<version>/<format>?api_key=KEY
      -> JSON with a link to a zip; the export may take a few seconds to be
         generated the first time, so we poll until the link exists.

Before running:
  1. Create a free account at https://app.roboflow.com and copy your API key
     (Settings -> API Keys). NEVER commit it.
  2. PowerShell:  $env:ROBOFLOW_API_KEY = "your_key"
     bash:        export ROBOFLOW_API_KEY=your_key

Usage:
    python scripts/court/download_keypoints.py                     # default version, YOLO-pose format
    python scripts/court/download_keypoints.py --version 22 --format coco
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.roboflow.com"
WORKSPACE = "roboflow-jvuqo"
PROJECT = "basketball-court-detection-2"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:          # noqa: S310 (fixed https host)
        return json.loads(r.read().decode())


def export_link(version: int, fmt: str, key: str, wait_s: int = 120) -> str:
    """Ask Roboflow for an export of ``version`` in ``fmt``; poll until the zip is ready."""
    url = f"{API}/{WORKSPACE}/{PROJECT}/{version}/{fmt}?api_key={key}"
    t0 = time.time()
    while True:
        data = get_json(url)
        link = (data.get("export") or {}).get("link")
        if link:
            return link
        if time.time() - t0 > wait_s:
            raise SystemExit(f"Export not ready after {wait_s}s. Response: {json.dumps(data)[:300]}")
        print("  export being generated on Roboflow's side, waiting...")
        time.sleep(5)


def safe_unzip(zip_path: Path, out_dir: Path) -> int:
    """Extract without letting a member escape ``out_dir`` (zip-slip guard)."""
    n = 0
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            dest = (out_dir / m.filename).resolve()
            if not str(dest).startswith(str(root)):
                raise SystemExit(f"Unsafe path in archive: {m.filename}")
            if m.is_dir():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, default=19, help="dataset version on Roboflow (19 = used in RF-DETR docs)")
    ap.add_argument("--format", default="yolov8", help="yolov8 (Ultralytics pose) | coco")
    ap.add_argument("--out", type=Path, default=Path("data/court_keypoints"))
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY first (see the docstring at the top of this file).")

    out = args.out / f"v{args.version}_{args.format}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty — delete it to download again.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Requesting {WORKSPACE}/{PROJECT} v{args.version} ({args.format})...")
    try:
        link = export_link(args.version, args.format, key)
    except urllib.error.HTTPError as e:
        msg = f"Roboflow answered HTTP {e.code}: check the key and that version {args.version} exists."
        raise SystemExit(msg) from e

    zip_path = out / "dataset.zip"
    print("Downloading...")
    urllib.request.urlretrieve(link, zip_path)                   # noqa: S310
    n = safe_unzip(zip_path, out)
    zip_path.unlink()
    print(f"{n} files extracted to {out}")
    print("Next: send me the folder listing, I will inspect the keypoint schema.")


if __name__ == "__main__":
    main()
