"""Print SHA256 of bundled checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_SCALE = {
    "echo_scale/seed42.pt": "0213112a917859afb075de29238aa55279d7554f321696fd79e15c7c60f2c7a2",
    "echo_scale/seed43.pt": "cabc39641a11a3e61698004f21f9fd62da34ff5a2deb7b917763ae8143d2b506",
    "echo_scale/seed44.pt": "54d82b39ae2b32e43f8a0beba06755c88c1ff3443d82495c97fa8242d830c40d",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    lines = []
    for path in sorted((ROOT / "checkpoints").rglob("*.pt")):
        rel = path.relative_to(ROOT / "checkpoints").as_posix()
        digest = sha256(path)
        mark = ""
        if rel in PAPER_SCALE:
            mark = "  MATCH" if digest == PAPER_SCALE[rel] else "  MISMATCH"
        print(f"{digest}  {rel}{mark}")
        lines.append(f"{rel}  {digest}{mark}")
    out = ROOT / "paper_results" / "checkpoint_sha256.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
