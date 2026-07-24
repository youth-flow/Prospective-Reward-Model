#!/usr/bin/env python3
"""Idempotently publish a terminal evidence bundle with its marker last."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path


def die(message: str) -> None:
    raise SystemExit(message)


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str]) -> None:
    if len(argv) < 4:
        die(
            "usage: publish_phase2_terminal_bundle.py "
            "<staging-dir> <destination-dir> <commit-filename> <filename>..."
        )
    staging = Path(argv[0])
    destination = Path(argv[1])
    commit_name = argv[2]
    names = argv[3:]
    if (
        staging.is_symlink()
        or not staging.is_dir()
        or destination.is_symlink()
        or not destination.is_dir()
        or staging.parent.resolve() not in {destination.resolve(), destination.parent.resolve()}
        or os.stat(staging).st_dev != os.stat(destination).st_dev
    ):
        die("terminal publication directories are unsafe or cross-filesystem")
    if (
        len(names) != len(set(names))
        or commit_name not in names
        or names[-1] != commit_name
        or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None for name in names)
    ):
        die("terminal publication filenames or commit ordering are invalid")
    observed = {path.name for path in staging.iterdir()}
    if observed != set(names):
        die("terminal staging directory does not contain the exact declared bundle")

    injection_raw = os.environ.get("PRORM_PHASE2_TEST_INTERRUPT_AFTER_PUBLICATION")
    interrupt_after: int | None = None
    if injection_raw is not None:
        if re.fullmatch(r"[1-9][0-9]*", injection_raw) is None:
            die("test interruption index is invalid")
        interrupt_after = int(injection_raw)

    for index, name in enumerate(names, 1):
        source = staging / name
        target = destination / name
        if source.is_symlink() or not source.is_file():
            die(f"terminal staging file is missing or unsafe: {name}")
        source_raw = source.read_bytes()
        source_sha = hashlib.sha256(source_raw).hexdigest()
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                die(f"existing terminal evidence is unsafe: {name}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != source_sha:
                die(f"terminal rerun changed existing evidence: {name}")
            source.unlink()
        else:
            try:
                os.link(source, target, follow_symlinks=False)
            except FileExistsError:
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or hashlib.sha256(target.read_bytes()).hexdigest() != source_sha
                ):
                    die(f"terminal publication lost an unsafe race: {name}")
            if os.name != "nt":
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
            source.unlink()
            fsync_directory(destination)
        if interrupt_after == index:
            die(f"injected interruption after terminal publication item {index}")
    staging.rmdir()
    fsync_directory(destination)


if __name__ == "__main__":
    main(sys.argv[1:])
