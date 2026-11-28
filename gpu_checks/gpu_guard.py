"""Exclusive GPU session for Inkling checks: an flock in WSL plus a free-memory check (exit code 75 if either fails)."""

import contextlib
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

LOCK_PATH = "/tmp/inkling_gpu.lock"  # noqa: S108  # WSL lock file path
BUSY_EXIT_CODE = 75


def gpu_free_mib() -> int:
    """Free memory of GPU 0 in MiB (nvidia-smi, nounits)."""
    out = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],  # noqa: S607  # fixed nvidia-smi command
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


@contextmanager
def gpu_session(min_free_mib: int = 8192) -> Iterator[None]:
    """Hold the GPU lock and require min_free_mib free; exit 75 when busy. The lock file is never deleted."""
    fd = Path(LOCK_PATH).open("a+")  # noqa: SIM115  # closed in finally
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # pyrefly: ignore[missing-attribute]  # POSIX only
        except BlockingIOError:
            print(f"GPU lock {LOCK_PATH} is held by another process; exiting {BUSY_EXIT_CODE}", file=sys.stderr)
            sys.exit(BUSY_EXIT_CODE)
        try:
            with contextlib.suppress(OSError):
                # Fresh mtime so a tmp cleaner (systemd-tmpfiles, 30 days) does not unlink the lock while it is held.
                os.utime(LOCK_PATH)
            try:
                free = gpu_free_mib()
            except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as exc:
                print(
                    f"cannot read free GPU memory from nvidia-smi ({exc!r}); exiting {BUSY_EXIT_CODE}", file=sys.stderr
                )
                sys.exit(BUSY_EXIT_CODE)
            if free < min_free_mib:
                print(f"only {free} MiB free on GPU 0, need {min_free_mib}; exiting {BUSY_EXIT_CODE}", file=sys.stderr)
                sys.exit(BUSY_EXIT_CODE)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)  # pyrefly: ignore[missing-attribute]  # POSIX only
    finally:
        fd.close()
