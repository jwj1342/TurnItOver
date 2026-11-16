"""Zero-shot fixed-observation diagnosis by a model behind an OpenAI-compatible API (a large open VLM served by vLLM),
on the same examples, user prompt and images as her evaluate_qwen_episode_sft.py, scored by her functions (score_prediction,
summarize, summarize_by_difficulty, object_bootstrap). Differences from her evaluator, recorded in metrics.json: the model
thinks before answering (the server's reasoning parser strips the reasoning; only the final answer is scored) and uses the
model's own sampling defaults instead of greedy decoding. Run with her code on PYTHONPATH (cwd = $TIO_RUNS/code900-v11).
Usage: python evaluate_api_zero_shot.py --dataset D --manifest-sha S --split validation --out O --model NAME \
           --endpoints http://node:port/v1 [...] [--workers 24] [--max-tokens 16384] [--bootstrap-samples 1000]
The API key is read from COMPATIBLE_API_KEY.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
from evaluate_qwen_episode_sft import (PROTOCOL_VERSION, load_training_examples, object_bootstrap,  # noqa: E402
                                       public_input_digest, score_prediction, summarize, summarize_by_difficulty)
from turnitover.taxonomy import load_taxonomy  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(endpoint: str, model: str, key: str, record: dict, root: Path, max_tokens: int, timeout: float) -> dict:
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode((root / p).read_bytes()).decode()}}
               for p in record["images"]]
    content.append({"type": "text", "text": record["messages"][0]["content"]})
    body = json.dumps({"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(4):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
            choice = data["choices"][0]
            return {"text": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage", {}), "elapsed_ms": (time.perf_counter() - start) * 1000, "attempts": attempt + 1}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = type(exc).__name__
            time.sleep(30 * (attempt + 1))
    return {"text": "", "finish_reason": f"error:{last}", "usage": {}, "elapsed_ms": 0.0, "attempts": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if sha256(args.dataset / "manifest.json") != args.manifest_sha:
        raise SystemExit("Dataset manifest SHA mismatch")
    args.out.mkdir(parents=True)
    examples = list(load_training_examples(args.dataset, args.split, "diagnosis", "evidence"))
    records = [record for record, _ in examples]
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    defect_ids = sorted(manifest.get("experiment_taxonomy", load_taxonomy().ids))
    key = os.environ["COMPATIBLE_API_KEY"]
    jobs = [(args.endpoints[i % len(args.endpoints)], record) for i, record in enumerate(records)]
    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        replies = list(pool.map(lambda job: call(job[0], args.model, key, job[1], args.dataset, args.max_tokens, args.timeout), jobs))
    rows = []
    with (args.out / "predictions.jsonl").open("w") as stream:
        for record, reply in zip(records, replies):
            row = score_prediction(record, reply["text"], "diagnosis")
            row.update(input_tokens=reply["usage"].get("prompt_tokens"), output_tokens=reply["usage"].get("completion_tokens"),
                       elapsed_ms=reply["elapsed_ms"], finish_reason=reply["finish_reason"], attempts=reply["attempts"])
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
    metrics = summarize(rows, "diagnosis", defect_ids)
    metrics["by_difficulty"] = summarize_by_difficulty(rows, "diagnosis", defect_ids, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    admission = manifest.get("label_revision_counts", {}).get(args.split, {})
    report = {"status": "complete", "version": 3, "scope": "fixed_observation_diagnosis", "model": args.model,
              "adapter": None, "decoding": "API, thinking on, model default sampling; final answer only",
              "max_tokens": args.max_tokens, "finish_reasons": {r: sum(x["finish_reason"] == r for x in replies) for r in {x["finish_reason"] for x in replies}},
              "protocol": {"version": PROTOCOL_VERSION, "task": "diagnosis", "split": args.split, "n": len(records),
                           "raw_cases": admission.get("raw_cases", len(records)),
                           "excluded_pending_adjudication": admission.get("quarantined_cases", 0),
                           "dataset_manifest_sha256": args.manifest_sha, "public_input_sha256": public_input_digest(records),
                           "defect_ids": defect_ids},
              **metrics, "elapsed_seconds": time.perf_counter() - started}
    if args.bootstrap_samples:
        report["bootstrap"] = object_bootstrap(rows, "status_accuracy", samples=args.bootstrap_samples, seed=args.seed,
                                               task="diagnosis", defect_ids=defect_ids)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"n": len(rows), "defect_macro_f1": report["defect_macro_f1"], "valid": sum(r["valid"] for r in rows),
                      "finish_reasons": report["finish_reasons"]}))


if __name__ == "__main__":
    main()
