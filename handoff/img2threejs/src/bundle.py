from __future__ import annotations

import base64
import hashlib
import io
import shutil
import tempfile
import zipfile
from pathlib import Path


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"不安全的 ZIP 路径：{info.filename}")
    archive.extractall(destination)


def materialize_session_bundle(bundle_dir: str | Path) -> Path | None:
    """Decode the versioned Chatbox bundle into a temporary session directory.

    Binary images are stored in git as split base64 text so the repository can be
    updated through text-oriented connectors. At runtime Streamlit reconstructs
    the ZIP once per bundle digest and reads the extracted session directories
    exactly like ordinary ``data/sessions/<id>/`` folders.
    """
    bundle_dir = Path(bundle_dir)
    parts = sorted(bundle_dir.glob("sessions_0910_webp80.zip.b64.part*"))
    if not parts:
        return None

    encoded = "".join(part.read_text(encoding="ascii").strip() for part in parts)
    digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()[:16]
    target = Path(tempfile.gettempdir()) / f"img2threejs-chatbox-{digest}"
    marker = target / ".ready"
    if marker.exists():
        return target

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    try:
        payload = base64.b64decode(encoded, validate=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            _safe_extract(archive, target)
        marker.write_text(digest, encoding="utf-8")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

    return target
