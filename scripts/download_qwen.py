"""Download only inference files from a pinned official Qwen snapshot (login node)."""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# Official Hugging Face LFS SHA256 for this revision's weight file.
WEIGHTS_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
FILES = ("README.md", "chat_template.json", "config.json", "generation_config.json", "merges.txt",
         "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
         "video_preprocessor_config.json", "vocab.json")


def main():
    output = Path(__file__).resolve().parents[1] / "models/Qwen3-VL-2B-Instruct"
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in FILES:
        target = output / name
        if not target.exists():
            partial = output / (name + ".partial")
            print(f"Downloading {name}", flush=True)
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3",
                            "--continue-at", "-", f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
                            "--output", str(partial)], check=True)
            partial.replace(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if name == "model.safetensors" and digest != WEIGHTS_SHA256:
            raise RuntimeError("Weight checksum mismatch; remove the invalid file before retrying")
        entries.append({"file": name, "bytes": target.stat().st_size, "sha256": digest})
    (output / "download-manifest.json").write_text(json.dumps({"repo": REPO, "revision": REVISION,
                                                               "files": entries}, indent=2))
    print(output, flush=True)


if __name__ == "__main__":
    main()
