#!/usr/bin/env python3
"""Operator-side grading with a fresh, disposable private-data copy per call.

Run only after search/export has finished. The private source belongs outside
the agent's checkout. Smoke and production calls never delete that source or
share the temporary grading tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--competition", required=True)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="prepared data root containing <competition>/prepared/public")
    p.add_argument("--private-source", type=Path, required=True,
                   help="operator-owned private directory, copied for each invocation")
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--output-log", type=Path, required=True)
    p.add_argument("--mlebench", required=True, help="pinned mlebench executable")
    a = p.parse_args(argv)
    public = a.data_dir.resolve() / a.competition / "prepared" / "public"
    private = a.private_source.resolve()
    if not public.is_dir() or not private.is_dir():
        p.error("prepared public data and private source must both exist")
    with tempfile.TemporaryDirectory(prefix="mlebench-grade-") as root:
        prepared = Path(root) / a.competition / "prepared"
        prepared.mkdir(parents=True)
        (prepared / "public").symlink_to(public, target_is_directory=True)
        shutil.copytree(private, prepared / "private")
        with a.output_log.open("w") as log:
            result = subprocess.run([
                a.mlebench, "grade-sample", str(a.submission.resolve()),
                a.competition, "--data-dir", root,
            ], stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            return result.returncode
        # grade-sample can exit successfully while reporting an invalid CSV.
        return 0 if '"valid_submission": true' in a.output_log.read_text() else 1


if __name__ == "__main__":
    raise SystemExit(main())
