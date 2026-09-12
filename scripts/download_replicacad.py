"""Download only ReplicaCAD's public articulated assets at a pinned revision.

No Habitat install or authentication needed. This acquires raw URDF/GLB files,
not AssetSpec conversions. Source: https://aihabitat.org/datasets/replica_cad/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

REPO = "ai-habitat/ReplicaCAD_dataset"
REVISION = "3e8c7fe5759f64bfcbc3882f9cdf6de97f82a06d"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download(output: Path):
    if output.exists():
        raise ValueError("Output must not exist; use a new directory for each acquisition")
    rows = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"))
    selected = [r for r in rows if r["type"] == "file" and (r["path"].startswith("urdf/") or r["path"] == "README.md")]
    if not selected or len(rows) >= 1000:
        raise ValueError("Unexpected or potentially truncated upstream file listing")
    output.mkdir(parents=True)
    provenance = {"dataset": REPO, "revision": REVISION, "license": "CC-BY-4.0",
                  "license_url": "https://creativecommons.org/licenses/by/4.0/",
                  "source": "https://aihabitat.org/datasets/replica_cad/", "status": "incomplete", "files": []}
    try:
        for row in selected:
            rel = PurePosixPath(row["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe upstream path")
            data = fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}")
            sha = hashlib.sha256(data).hexdigest()
            if len(data) != row["size"]:
                raise ValueError(f"Size mismatch: {rel}")
            if "lfs" in row:
                if sha != row["lfs"]["oid"]:
                    raise ValueError(f"SHA256 mismatch: {rel}")
            elif hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != row["oid"]:
                raise ValueError(f"Git blob hash mismatch: {rel}")
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            provenance["files"].append({"path": str(rel), "bytes": len(data), "sha256": sha})
        assets = []
        for path in sorted(output.rglob("*.urdf")):
            root = ET.parse(path).getroot()
            meshes = [m.attrib["filename"] for m in root.findall(".//mesh")]
            missing = [m for m in meshes if not (path.parent / m).is_file()]
            if missing:
                raise ValueError(f"Missing meshes in {path}: {missing}")
            assets.append({"urdf": str(path.relative_to(output)), "links": len(root.findall("link")),
                "split_group": str(path.parent.relative_to(output)),
                "joints": [{"name": j.get("name"), "type": j.get("type"),
                            "axis": j.find("axis").attrib if j.find("axis") is not None else None,
                            "limits": j.find("limit").attrib if j.find("limit") is not None else None}
                           for j in root.findall("joint")],
                "mesh_files": sorted(set(meshes)), "missing_meshes": missing,
                "contains_dummy_inertia": "dummy inertia" in path.read_text().lower()})
        provenance.update(status="complete", assets=assets,
                          limitations=["Raw asset acquisition only; URDF to Three.js conversion pending.",
                                       "Dynamic/static and door variants must remain in the same split group.",
                                       "Inertia and physical validity are not certified by this inventory."])
    finally:
        (output / "acquisition.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"output": str(output), "files": len(selected), "urdfs": len(assets),
                      "bytes": sum(r["bytes"] for r in provenance["files"])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/assets/replicacad-3e8c7fe"))
    download(parser.parse_args().out)
