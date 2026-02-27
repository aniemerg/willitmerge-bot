"""Colored terminal logger — mirrors willitmerge-bot logger.ts."""

import sys

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_GRAY   = "\033[90m"


def _write(color: str, prefix: str, msg: str, file=sys.stdout) -> None:
    print(f"{color}{_BOLD}{prefix}{_RESET} {msg}", file=file, flush=True)


def banner(msg: str) -> None:
    sep = "─" * max(0, 60 - len(msg) - 4)
    print(f"\n{_CYAN}{_BOLD}── {msg} {sep}{_RESET}", flush=True)


def info(msg: str) -> None:
    _write(_GRAY, "[info]", msg)


def step(msg: str) -> None:
    _write(_CYAN, "[step]", msg)


def success(msg: str) -> None:
    _write(_GREEN, "[ok]  ", msg)


def warn(msg: str) -> None:
    _write(_YELLOW, "[warn]", msg, file=sys.stderr)


def error(msg: str) -> None:
    _write(_RED, "[err] ", msg, file=sys.stderr)
