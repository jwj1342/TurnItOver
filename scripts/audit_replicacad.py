"""Audit each imported asset at rest, every joint detent, and combined configurations."""
import argparse
from pathlib import Path

from turnitover.assets.audit_urdf import audit_catalog
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import load_views


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/assets/replicacad-catalog-v1/catalog.json"))
    parser.add_argument("--out", type=Path, default=Path("output/replicacad-fk-v1"))
    args = parser.parse_args()
    with ObservationSession(RenderConfig(Path("web/dist"), Path("web/node_modules/.bin/esbuild"), 256, 256),
                            load_views(Path("configs/views.yaml"))) as session:
        report = audit_catalog(args.catalog, args.out, session)
    print({"passed": report["passed"], "assets": len(report["results"]),
           "states": sum(len(r["states"]) for r in report["results"])})
