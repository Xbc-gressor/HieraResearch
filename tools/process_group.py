"""Child process-group termination shared by every spawn layer.

Rule: whoever spawns a process group is responsible for killing it when it
is itself terminated. Evaluation children run in their own session
(``start_new_session=True``), so a ``killpg`` aimed at the parent's group
never reaches them; each wrapper therefore registers its live child and, on
SIGTERM, terminates that child's group and waits for it to exit before
exiting itself. Nested layers (driver → driver job → evaluation) each hold
only their own children, so the outermost watchdog needs to address only
the driver's group.

Termination is never "signal sent": ``terminate_group`` returns only after
the child has exited (SIGTERM, bounded grace, SIGKILL), so ownership of
whatever the child held (a device lease, a report file) can be handed over
safely afterwards.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading

_POSIX = os.name == "posix"
_KILL_WAIT_SECONDS = 60.0

_live_children: list[subprocess.Popen] = []
_guard = threading.Lock()
_armed = False


def _signal_group(proc: subprocess.Popen, sig) -> None:
    try:
        if _POSIX:
            os.killpg(os.getpgid(proc.pid), sig)
        else:  # pragma: no cover - Windows has no process groups to signal
            proc.kill() if sig != signal.SIGTERM else proc.terminate()
    except (ProcessLookupError, OSError):
        pass


def terminate_group(proc: subprocess.Popen, grace: float = 30.0) -> int | None:
    """Terminate ``proc``'s whole process group and wait for it to exit.

    ``grace`` > 0: SIGTERM first, wait that long, then SIGKILL anything still
    running. ``grace`` <= 0: SIGKILL immediately. Returns the exit code, or
    None if the group could not be reaped within the kill wait (a process
    stuck in the kernel; the caller records that rather than assuming exit).
    """
    if proc.poll() is not None:
        return proc.returncode
    if grace and grace > 0:
        _signal_group(proc, signal.SIGTERM)
        try:
            return proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    _signal_group(proc, signal.SIGKILL)
    try:
        return proc.wait(timeout=_KILL_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        return None


def register_child(proc: subprocess.Popen) -> None:
    with _guard:
        _live_children.append(proc)


def unregister_child(proc: subprocess.Popen) -> None:
    with _guard:
        if proc in _live_children:
            _live_children.remove(proc)


def terminate_live_children(grace: float = 30.0) -> None:
    with _guard:
        children = list(_live_children)
    for proc in children:
        terminate_group(proc, grace=grace)


def arm_sigterm_forwarding(grace: float = 30.0) -> bool:
    """Install the SIGTERM → terminate-children → exit handler (main thread
    only; idempotent). Also registers the same cleanup at interpreter exit.
    Returns whether the handler is armed after the call."""
    global _armed
    if _armed:
        return True
    if threading.current_thread() is not threading.main_thread():
        return False
    atexit.register(terminate_live_children, grace)

    def _on_sigterm(signum, _frame) -> None:
        terminate_live_children(grace)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_sigterm)
    _armed = True
    return True
