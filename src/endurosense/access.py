"""A record of every time the locked test set is opened.

Plain idea: the test set should be looked at once. If the final evaluation has
to be rerun after something changed (a bug fix, a new setting from the guide),
that must be visible and explained, not silent. So every opening is written to
a log together with a fingerprint of the code, configuration and split:

- unchanged fingerprint: a reproducibility rerun, nothing is added;
- changed fingerprint: refused unless a reason is given, which is logged.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from endurosense.config import ROOT, load_config


class ReasonRequired(PermissionError):
    """The code changed since the test set was last opened and no reason was given."""


def code_fingerprint() -> str:
    """Identifies the code, configuration and split an evaluation ran with."""
    cfg = load_config()
    files = (sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
             + sorted((ROOT / "config").glob("*.yaml")) + [ROOT / "config.yaml", ROOT / cfg["split"]["file"]])
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def register_opening(log_path, reason: str | None = None, fingerprint: str | None = None) -> str:
    """Record that the test set is being opened; returns the fingerprint."""
    log_path = Path(log_path)
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    fp = fingerprint or code_fingerprint()
    if log and log[-1]["fingerprint"] == fp:
        return fp
    if log and not reason:
        raise ReasonRequired(f"The code, configuration or split changed since the test set was last opened "
                             f"({log[-1]['opened_at']}). Rerun with --reason \"what changed and why\"; it is added to {log_path}.")
    log.append({"opened_at": datetime.datetime.now().isoformat(timespec="seconds"), "fingerprint": fp,
                "reason": reason or "first final evaluation"})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=1))
    return fp


def artefact_hashes(paths) -> dict:
    """sha256 (first 16 hex digits) of each file, keyed by its path relative to the project.
    Recorded with the final results so it is clear exactly which saved models and tables were scored."""
    out = {}
    for p in sorted(Path(p) for p in paths):
        out[p.resolve().relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out

