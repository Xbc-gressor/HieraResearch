"""run_metadata.json: A/B provenance for model/runtime/prompt identity.

Record + warn, never refuse: a toolchain upgrade must not orphan an
in-flight run, but any drift from the run's starting configuration is
surfaced loudly on resume.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

PERMISSION_POLICY = "bypassPermissions+pre-tool-use-capability-hook"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_metadata(model: str, cli_path: str | None) -> dict:
    try:
        sdk = pkg_version("claude-agent-sdk")
    except PackageNotFoundError:
        sdk = "unknown"
    if cli_path:
        out = subprocess.run(
            [cli_path, "--version"], capture_output=True, text=True, check=True
        )
        cli_source, cli_version = "system", out.stdout.strip()
    else:
        cli_source = "bundled"
        cli_version = f"bundled-with-claude-agent-sdk-{sdk}"
    prompt_hashes = {}
    if PROMPT_DIR.exists():
        prompt_hashes = {
            str(p.relative_to(PROMPT_DIR)): _sha256(p)
            for p in sorted(PROMPT_DIR.rglob("*.md"))
        }
    return {
        "model": model,
        "sdk_version": sdk,
        "cli_source": cli_source,
        "cli_version": cli_version,
        "permission_policy": PERMISSION_POLICY,
        "prompt_hashes": prompt_hashes,
    }


def write_metadata(run_dir: Path, model: str, cli_path: str | None) -> dict:
    meta = collect_metadata(model, cli_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def warn_on_mismatch(run_dir: Path, model: str, cli_path: str | None) -> list[str]:
    stored_path = run_dir / "run_metadata.json"
    if not stored_path.exists():
        return []
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    current = collect_metadata(model or stored.get("model", ""), cli_path)
    warnings: list[str] = []
    for key in ("sdk_version", "cli_version"):
        if stored.get(key) != current.get(key):
            warnings.append(
                f"run metadata mismatch: {key} stored={stored.get(key)!r} "
                f"current={current.get(key)!r}"
            )
    if model and stored.get("model") != model:
        warnings.append(
            f"run metadata mismatch: model stored={stored.get('model')!r} current={model!r}"
        )
    if stored.get("prompt_hashes") != current.get("prompt_hashes"):
        warnings.append("run metadata mismatch: prompt hashes differ from run start")
    return warnings
