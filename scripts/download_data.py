"""Download SportsMOT from the Hugging Face Hub and keep only one sport.

The dataset is gated (CC BY-NC 4.0). Before running:
  1. Open https://huggingface.co/datasets/MCG-NJU/SportsMOT and accept the terms.
  2. `huggingface-cli login` with a read token.

The archives are big (train.tar ~6.7 GB, val.tar ~6.6 GB) and contain all three
sports. We stream through each tar and extract ONLY the basketball sequences
(~1/3 of the frames), then optionally delete the archive.

Usage:
    python scripts/download_data.py                    # train + val, basketball
    python scripts/download_data.py --splits val --keep-tar
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath

REPO = "MCG-NJU/SportsMOT"


def safe_extract_sequences(tar_path: Path, out_root: Path, split: str, keep: set[str]) -> int:
    """Extract members belonging to ``keep`` sequences into out_root/split/<seq>/...

    We never call ``tar.extractall``: each destination path is rebuilt from the
    sequence name onwards, so a malicious member name ("../../etc") cannot
    escape ``out_root`` (classic tar path-traversal).
    """
    n = 0
    with tarfile.open(tar_path, "r|*") as tar:          # "r|" = streaming, low memory
        for member in tar:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            # The archives were packed on macOS: skip AppleDouble metadata
            # ("._000001.jpg", 176 bytes, not an image) and __MACOSX folders.
            if parts[-1].startswith("._") or "__MACOSX" in parts:
                continue
            hit = next((i for i, p in enumerate(parts) if p in keep), None)
            if hit is None or ".." in parts:
                continue
            dest = out_root / split / Path(*parts[hit:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(dest, "wb") as f:
                while chunk := src.read(1 << 20):
                    f.write(chunk)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/SportsMOT"))
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--sport", default="basketball", choices=["basketball", "football", "volleyball"])
    ap.add_argument("--keep-tar", action="store_true", help="do not delete the archives afterwards")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    # 1) Split lists (tiny) — tell us which sequences are basketball.
    for name in ("basketball", "football", "volleyball", "train", "val", "test"):
        hf_hub_download(REPO, f"splits_txt/{name}.txt", repo_type="dataset", local_dir=args.out)
    keep = set((args.out / "splits_txt" / f"{args.sport}.txt").read_text().split())
    print(f"{len(keep)} {args.sport} sequences in total")

    # 2) Archives, one per split.
    for split in args.splits:
        split_seqs = set((args.out / "splits_txt" / f"{split}.txt").read_text().split())
        wanted = keep & split_seqs
        print(f"[{split}] downloading archive ({len(wanted)} {args.sport} sequences wanted)...")
        tar_path = Path(hf_hub_download(REPO, f"dataset/{split}.tar", repo_type="dataset",
                                        local_dir=args.out / "_archives"))
        n = safe_extract_sequences(tar_path, args.out / "dataset", split, wanted)
        print(f"[{split}] extracted {n} files")
        if not args.keep_tar:
            tar_path.unlink()

    print("Done. Sequences are in", args.out / "dataset")


if __name__ == "__main__":
    main()
