"""Run a command under a per-evaluation wall-clock limit — portable (Linux / macOS / Windows).

    python tools/timed_run.py [<limit_seconds>] <command> [args...]

The limit comes from, in order:
  1. an explicit numeric first argument (`timed_run.py 90 python train.py`), or
  2. `per_runtime_limit` (top-level) in the nearest `framework_cfg.json` found by
     walking up from the command's entrypoint — the last argument that is an
     existing file (`timed_run.py python /abs/.../train.py` → reads
     /abs/.../<run_dir>/framework_cfg.json). This matches how the framework's
     `_common.timed_eval` reads the same key, so both methods share one knob.
If neither yields a positive limit, the command runs with NO timeout. A
`framework_cfg.json` that exists but cannot be parsed is a hard error (exit 2):
silently dropping the limit would let the command run unbounded.

Same subprocess-kill mechanism as `timed_eval`: the command runs in its own
session/process group; on timeout the whole group is killed and this exits 124
(the GNU `timeout` convention). Otherwise it forwards the command's exit code.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

from run_cfg import RunConfigError, find_framework_cfg, read_framework_cfg


def _limit_from_cfg(cmd: list[str]) -> float | None:
    entry = next((Path(a).resolve() for a in reversed(cmd) if os.path.isfile(a)), None)
    if entry is None:
        return None
    cfg = find_framework_cfg(entry)
    if cfg is None:
        return None
    v = read_framework_cfg(cfg).get("per_runtime_limit")
    try:
        v = float(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.stderr.write("usage: timed_run.py [<limit_seconds>] <command> [args...]\n")
        return 2
    # explicit numeric limit, else read from framework_cfg.json near the entrypoint
    limit = None
    try:
        limit = float(args[0])
        cmd = args[1:]
    except ValueError:
        cmd = args
        try:
            limit = _limit_from_cfg(cmd)
        except RunConfigError as exc:
            sys.stderr.write(f"timed_run.py: {exc}\n")
            return 2
    if not cmd:
        sys.stderr.write("timed_run.py: no command\n")
        return 2

    posix = os.name == "posix"
    kwargs = {}
    if posix:
        kwargs["start_new_session"] = True
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, **kwargs)
    if limit is None:
        return proc.wait()                       # no limit configured
    try:
        return proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        try:
            if posix:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return 124                               # timed out


if __name__ == "__main__":
    raise SystemExit(main())
