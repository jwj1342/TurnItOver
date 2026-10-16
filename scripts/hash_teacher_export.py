"""Write and verify a complete SHA-256 manifest for a local teacher export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write_hash_manifest(dataset: Path) -> dict:
    entries = [{"path": path.relative_to(dataset).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
               for path in sorted(dataset.rglob("*"))
               if path.is_file() and path.name != "sha256.json"]
    manifest = {"algorithm": "sha256", "files": entries}
    (dataset / "sha256.json").write_text(json.dumps(manifest, indent=2))
    for row in entries:
        path = dataset / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"SHA-256 verification failed: {path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    result = write_hash_manifest(args.dataset)
    print(json.dumps({"dataset": str(args.dataset), "files": len(result["files"]),
                      "status": "verified"}, indent=2))
