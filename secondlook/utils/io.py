from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if missing; return resolved Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    import yaml

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {p}, got {type(data).__name__}")
    return data


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Serialize *obj* to JSON at *path*."""
    p = Path(path)
    ensure_dir(p.parent)

    def _default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    p.write_text(
        json.dumps(obj, indent=indent, default=_default, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p.resolve()


def write_text(text: str, path: str | Path) -> Path:
    """Write plain text to *path*, creating parent directories as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p.resolve()


def save_command_snapshot(path: str | Path) -> Path:
    """Save the current process command line and timestamp to *path*."""
    cmd = " ".join(sys.argv)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = f"# Stage 0 audit command snapshot\n# UTC: {ts}\n\n{cmd}\n"
    return write_text(body, path)


def safe_relpath(path: str | Path, root: str | Path) -> str:
    """Return *path* relative to *root* when possible; otherwise absolute string."""
    p = Path(path).resolve()
    r = Path(root).resolve()
    try:
        return str(p.relative_to(r))
    except ValueError:
        return str(p)
