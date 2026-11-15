"""Transfer manifest for the experiment data that is not in git (models/ and tio-runs/ inputs).

write  (on Nibi):   records size for every file and SHA256 for files up to 64 MB.
check  (on Vulcan): compares sizes and those hashes; weight shards and zips are checked by size, and by hash with --full.
Standard library only, so it runs before any venv exists:

    python3 scripts/exp900/manifest.py check                # login node, about a minute
    python3 scripts/exp900/manifest.py check --full         # also hashes the 21 GB of weights; run it in a CPU job
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES = ("models/Qwen3-VL-2B-Instruct", "models/Qwen3-VL-8B-Instruct", "tio-runs/code900", "tio-runs/data900", "tio-runs/out")
MANIFEST = Path(__file__).with_name("transfer-manifest.tsv")
SMALL = 64 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files() -> list[Path]:
    return sorted(p for tree in TREES for p in (ROOT / tree).rglob("*") if p.is_file())


def write() -> None:
    lines = []
    for path in files():
        size = path.stat().st_size
        lines.append(f"{path.relative_to(ROOT)}\t{size}\t{sha256(path) if size <= SMALL else '-'}")
    MANIFEST.write_text("# path\tbytes\tsha256 (- = size only unless --full)\n" + "\n".join(lines) + "\n")
    print(f"wrote {len(lines)} entries to {MANIFEST.relative_to(ROOT)}")


def check(full: bool) -> int:
    expected = [line.split("\t") for line in MANIFEST.read_text().splitlines() if line and not line.startswith("#")]
    missing, wrong_size, wrong_hash, unhashed = [], [], [], 0
    big_hashes = {}
    if full:     # weights were not hashed on Nibi; the 2B weight hash is the official one recorded by scripts/download_qwen.py
        big_hashes["models/Qwen3-VL-2B-Instruct/model.safetensors"] = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
    for rel, size, digest in expected:
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
        elif path.stat().st_size != int(size):
            wrong_size.append(rel)
        elif digest != "-":
            if sha256(path) != digest:
                wrong_hash.append(rel)
        elif full and rel in big_hashes:
            if sha256(path) != big_hashes[rel]:
                wrong_hash.append(rel)
        else:
            unhashed += 1
    extra = sorted({str(p.relative_to(ROOT)) for p in files()} - {e[0] for e in expected})
    print(f"entries {len(expected)}  missing {len(missing)}  wrong size {len(wrong_size)}  wrong sha256 {len(wrong_hash)}  "
          f"size-only {unhashed}  extra (new outputs are fine) {len(extra)}")
    for label, items in (("MISSING", missing), ("WRONG SIZE", wrong_size), ("WRONG SHA256", wrong_hash)):
        for rel in items[:20]:
            print(f"  {label} {rel}")
    ok = not (missing or wrong_size or wrong_hash)
    print("TRANSFER OK" if ok else "TRANSFER INCOMPLETE: re-run the Globus transfer for the listed paths")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--full", action="store_true", help="also hash large files where a reference hash is known")
    args = parser.parse_args()
    if args.action == "write":
        write()
    else:
        sys.exit(check(args.full))


if __name__ == "__main__":
    main()
