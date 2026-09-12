"""Compare emitted Three.js geometry to FK evaluated directly from upstream URDF."""
import json
from pathlib import Path

import numpy as np

from turnitover.assets.catalog import CatalogSource
from turnitover.assets.urdf import UrdfModel
from turnitover.core.program import ObjectProgram


def configurations(model):
    yield "rest", {}
    active = {name: joint for name, joint in model.joints.items() if joint.get("type") != "fixed"}
    for name, joint in active.items():
        lo, hi = model.limits(joint)
        for label, value in (("lower", lo), ("mid", (lo+hi)/2), ("upper", hi)):
            yield f"{name}:{label}", {name: value}
    rng = np.random.default_rng(712)
    for i in range(3):
        yield f"combined:{i}", {name: float(rng.uniform(*model.limits(joint))) for name, joint in active.items()}


def audit_model(session, model, spec, output=None, tolerance=1e-5):
    session.load(ObjectProgram.from_spec(spec), framing=None)
    records = []
    try:
        for name, values in configurations(model):
            session.reset_joints()
            session.set_joints(values)
            expected = model.world_vertices(values)
            actual = session.export_geometry()
            if set(expected) != set(actual):
                raise ValueError("Imported part set does not match source visual links")
            worst = 0.
            for part, vertices in expected.items():
                observed = actual[part]
                if observed.vertices.shape != vertices.shape:
                    raise ValueError(f"Vertex shape mismatch for {part}")
                source_faces, offset = [], 0
                for mesh in model.meshes[part]:
                    source_faces.append(mesh.faces + offset)
                    offset += len(mesh.vertices)
                faces = np.concatenate(source_faces)
                if not np.array_equal(observed.faces, faces):
                    raise ValueError(f"Triangle topology changed for {part}")
                worst = max(worst, float(np.linalg.norm(observed.vertices-vertices, axis=1).max()))
            records.append({"state": name, "joint_values": values, "max_vertex_error_m": worst, "passed": worst <= tolerance})
            if output is not None and name in ("rest", "combined:0"):
                (output / ("rest.png" if name == "rest" else "motion.png")).write_bytes(session.request_view("front").png)
        return {"asset_id": spec.asset_id, "program_sha": ObjectProgram.from_spec(spec).sha,
                "source_files": model.provenance(), "tolerance_m": tolerance, "passed": all(r["passed"] for r in records),
                "max_vertex_error_m": max(r["max_vertex_error_m"] for r in records), "states": records}
    finally:
        session.dispose()


def audit_catalog(catalog: Path, output: Path, session):
    if output.exists():
        raise ValueError("Audit output must not exist")
    output.mkdir(parents=True)
    conversion = json.loads((catalog.parent / "conversion.json").read_text())
    specs = CatalogSource(str(catalog))
    results = []
    manifest = {"status": "incomplete", "results": results}
    try:
        for entry in conversion["assets"]:
            target = output / entry["asset_id"]
            target.mkdir()
            results.append(audit_model(session, UrdfModel(Path(entry["urdf"])), specs.get(entry["asset_id"]), target))
        manifest.update(status="complete", passed=all(r["passed"] for r in results),
                        source="independent URDF FK vs browser-exported vertices",
                        limitations=["Kinematic/visual mesh fidelity only; no physics or texture fidelity claim."])
    finally:
        (output / "audit.json").write_text(json.dumps(manifest, indent=2))
    if not manifest["passed"]:
        raise ValueError("URDF conversion fidelity audit failed")
    return manifest
