"""Subprocess isolation with bounded diagnostics and process-group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class ProcessError(RuntimeError):
    pass


class ProcessInterrupted(ProcessError):
    def __init__(self, result: "ProcessResult"):
        super().__init__(f"command interrupted: {' '.join(result.args)}")
        self.result = result


@dataclass(frozen=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    output: str
    elapsed_seconds: float
    timed_out: bool = False
    interrupted: bool = False
    output_truncated: bool = False
    output_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.interrupted


class ProcessRunner:
    def __init__(self, *, capture_limit_bytes: int = 128 * 1024, kill_grace_seconds: float = 2.0):
        if capture_limit_bytes <= 0:
            raise ValueError("capture_limit_bytes must be positive")
        if kill_grace_seconds <= 0:
            raise ValueError("kill_grace_seconds must be positive")
        self.capture_limit_bytes = capture_limit_bytes
        self.kill_grace_seconds = kill_grace_seconds

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        timeout: float,
        output_path: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        command = tuple(str(value) for value in args)
        if not command or any(not value for value in command):
            raise ValueError("command must contain non-empty argument strings")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        cwd = Path(cwd).resolve()
        if not cwd.is_dir():
            raise ProcessError(f"working directory does not exist: {cwd}")

        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(key): str(value) for key, value in env.items()})

        owned_output = output_path is None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w+b")
        else:
            output_handle = tempfile.TemporaryFile(mode="w+b")

        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        timed_out = False
        interrupted = False
        try:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=merged_env,
                    stdin=subprocess.DEVNULL,
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ProcessError(f"failed to start {command[0]!r}: {exc}") from exc

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_group(process)
            except KeyboardInterrupt:
                interrupted = True
                self._terminate_group(process)

            elapsed = time.monotonic() - started
            output_handle.flush()
            size = output_handle.seek(0, os.SEEK_END)
            truncated = size > self.capture_limit_bytes
            output_handle.seek(max(0, size - self.capture_limit_bytes))
            output = output_handle.read().decode("utf-8", errors="replace")
            returncode = process.returncode
            if timed_out:
                returncode = 124
            elif interrupted:
                returncode = 130
            result = ProcessResult(
                args=command,
                returncode=int(returncode if returncode is not None else 1),
                output=output,
                elapsed_seconds=elapsed,
                timed_out=timed_out,
                interrupted=interrupted,
                output_truncated=truncated,
                output_path=output_path,
            )
            if interrupted:
                raise ProcessInterrupted(result)
            return result
        finally:
            output_handle.close()
            if owned_output:
                output_path = None

    def _terminate_group(self, process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=self.kill_grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            # The group can still contain a child after its leader exits.
            try:
                os.killpg(process.pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
