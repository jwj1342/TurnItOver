"""Convert the six acquired ReplicaCAD asset families into an embedded-mesh catalog."""
import argparse
import json
from pathlib import Path

from turnitover.assets.urdf import UrdfModel
from turnitover.core.serde import to_dict

SELECTION = {
    "cabinet": "cabinet/cabinet.urdf",
    "chest_of_drawers": "chest_of_drawers/chestOfDrawers_01.urdf",
    "door": "doors/door1.urdf",
    "fridge": "fridge/fridge.urdf",
    "kitchen_counter": "kitchen_counter/kitchen_counter.urdf",
    "kitchen_cupboards": "kitchen_cupboards/kitchenCupboard_01.urdf",
}


def convert(root, output):
    acquisition = json.loads((root / "acquisition.json").read_text())
    if acquisition["status"] != "complete" or acquisition["license"] != "CC-BY-4.0":
        raise ValueError("Expected a complete ReplicaCAD acquisition")
    if output.exists():
        raise ValueError("Output must not exist")
    output.mkdir(parents=True)
    entries, reports = [], []
    for category, filename in SELECTION.items():
        model = UrdfModel(root / "urdf" / filename)
        hashes = {r["path"]: r["sha256"] for r in acquisition["files"]}
        for item in model.provenance():
            rel = Path(item["path"]).relative_to(root.resolve()).as_posix()
            if hashes.get(rel) != item["sha256"]:
                raise ValueError(f"Acquired asset changed: {rel}")
        spec = model.spec("replicacad_" + category, category)
        spec_path = category + ".json"
        (output / spec_path).write_text(json.dumps(to_dict(spec), separators=(",", ":")))
        entries.append({"spec": spec_path, "split_group": "replicacad:" + filename.split("/")[0],
                        "origin": f"{acquisition['source']} revision={acquisition['revision']} urdf/{filename}",
                        "license": "CC-BY-4.0"})
        reports.append({"asset_id": spec.asset_id, "urdf": str(model.path), "files": model.provenance(),
                        "parts": len(spec.parts), "joints": len(spec.joints),
                        "triangles": sum(len(p.mesh.faces) for p in spec.parts)})
    (output / "catalog.json").write_text(json.dumps({"version": 1, "source_id": "replicacad-" + acquisition["revision"],
                                                    "assets": entries}, indent=2))
    (output / "conversion.json").write_text(json.dumps({"version": 1, "status": "complete", "assets": reports,
        "limitations": ["Textures sampled to vertex colors; not appearance-exact.",
                        "Collision shapes, mass and inertia not imported or certified.",
                        "Conversion requires independent browser FK audit before benchmark use."]}, indent=2))
    print(output / "catalog.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/assets/replicacad-3e8c7fe"))
    parser.add_argument("--out", type=Path, default=Path("data/assets/replicacad-catalog-v1"))
    args = parser.parse_args()
    convert(args.root, args.out)
