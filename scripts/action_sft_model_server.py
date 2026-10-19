"""Private localhost model bridge; transport contains only public prompts and PNGs."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turnitover.models.action_qwen import SharedModel


def serve(model, port):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if set(payload) != {"prompt", "images", "enabled", "final"}:
                    raise ValueError("Unexpected inference fields")
                if type(payload["enabled"]) is not bool or type(payload["final"]) is not bool:
                    raise ValueError("Adapter/final flags must be bool")
                if payload["final"] and payload["enabled"]:
                    raise ValueError("Final readout must disable the action adapter")
                with tempfile.TemporaryDirectory(prefix="action-sft-public-") as folder:
                    images = []
                    for index, encoded in enumerate(payload["images"]):
                        path = Path(folder) / f"{index}.png"
                        path.write_bytes(base64.b64decode(encoded, validate=True))
                        images.append(path)
                    result = model.call(payload["enabled"], payload["final"])(payload["prompt"], images)
                response = json.dumps(asdict(result)).encode()
                self.send_response(200)
            except Exception as exc:
                response = json.dumps(dict(error=type(exc).__name__, detail=str(exc))).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Omit for the frozen base-only baseline server")
    parser.add_argument("--port", type=int, default=8893)
    parser.add_argument("--action-tokens", type=int, default=128)
    parser.add_argument("--final-tokens", type=int, default=1024)
    args = parser.parse_args()
    model = SharedModel(args.model, args.adapter, args.action_tokens, args.final_tokens)
    import torch
    import transformers
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(dict(status="ready", host="127.0.0.1", port=args.port,
        model=str(args.model.resolve()), adapter=str(args.adapter.resolve()) if args.adapter else None,
        adapter_sha256=digest(args.adapter / "adapter_model.safetensors") if args.adapter else None,
        base_config_sha256=digest(args.model / "config.json"),
        action_tokens=args.action_tokens, final_tokens=args.final_tokens,
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
        adapter_loaded=model.has_adapter, final_diagnoser="frozen_base_adapter_disabled")), flush=True)
    serve(model, args.port)
