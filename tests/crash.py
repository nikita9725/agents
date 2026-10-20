"""Инъекция аварии: выполняется внутри тестового контейнера через tests.stack."""

import os
import signal
from pathlib import Path


def main() -> None:
    # Kill the server beneath Docker's init; docker kill suppresses automatic restart.
    children = [int(pid) for pid in Path("/proc/1/task/1/children").read_text().split()]
    target = min(pid for pid in children if pid != os.getpid())
    os.kill(target, signal.SIGKILL)


if __name__ == "__main__":
    main()
