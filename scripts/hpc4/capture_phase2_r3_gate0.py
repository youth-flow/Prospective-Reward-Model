#!/usr/bin/env python3
"""Capture or verify the immutable R2-failure parent required by R3 Gate 0.

Formal ``capture`` and ``verify`` have no output/root override.  They operate
only on the fixed HPC4 production namespace and return a typed capability.
``inspect`` is intentionally non-authorizing and may be used on a copied
artifact for offline review.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

_SCRIPT_RELATIVE = Path("scripts/hpc4/capture_phase2_r3_gate0.py")
_MODULE_RELATIVE = Path("src/smart_reward/phase2_r3_gate0.py")
_MODULE_NAME = "smart_reward.phase2_r3_gate0"


def _load_gate0_module() -> types.ModuleType:
    """Load the exact colocated stdlib-only module without package startup."""

    script = Path(__file__).absolute()
    if script.is_symlink() or script.resolve(strict=True) != script:
        raise RuntimeError("Gate-0 CLI must use its canonical non-symlink path")
    repository = script.parents[2]
    if script != repository / _SCRIPT_RELATIVE:
        raise RuntimeError("Gate-0 CLI escaped its fixed repository-relative path")
    git_metadata = repository / ".git"
    if not git_metadata.exists() or git_metadata.is_symlink():
        raise RuntimeError("Gate-0 CLI must run from a real Git checkout")
    package_directory = repository / "src" / "smart_reward"
    module_path = repository / _MODULE_RELATIVE
    if (
        package_directory.is_symlink()
        or package_directory.resolve(strict=True) != package_directory
        or not package_directory.is_dir()
        or module_path.is_symlink()
        or module_path.resolve(strict=True) != module_path
        or not module_path.is_file()
    ):
        raise RuntimeError("Gate-0 module path is not canonical fixed repository source")
    if "smart_reward" in sys.modules or _MODULE_NAME in sys.modules:
        raise RuntimeError("refusing a preloaded smart_reward package or Gate-0 module")

    package = types.ModuleType("smart_reward")
    package.__package__ = "smart_reward"
    package.__path__ = [str(package_directory)]
    package.__spec__ = importlib.util.spec_from_loader(
        "smart_reward",
        loader=None,
        is_package=True,
    )
    specification = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not construct the fixed Gate-0 module specification")
    module = importlib.util.module_from_spec(specification)
    sys.modules["smart_reward"] = package
    sys.modules[_MODULE_NAME] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        sys.modules.pop("smart_reward", None)
        raise
    if Path(module.__file__).resolve(strict=True) != module_path:
        raise RuntimeError("loaded Gate-0 module did not retain its fixed source identity")
    return module


_gate0 = _load_gate0_module()
capture_live_r3_gate0_bundle = _gate0.capture_live_r3_gate0_bundle
inspect_r3_gate0_bundle = _gate0.inspect_r3_gate0_bundle
verify_live_r3_gate0_bundle = _gate0.verify_live_r3_gate0_bundle


def _capability_payload(value: object) -> dict[str, object]:
    return {
        "status": "validated_live_hpc4_gate0",
        "schema_version": value.schema_version,
        "role": value.role,
        "artifact_sha256": value.artifact_sha256,
        "file_sha256": value.file_sha256,
        "production_relative": value.production_relative,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture and publish live HPC4 Gate 0")
    capture.add_argument("--container", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="reverify published live HPC4 Gate 0")
    verify.add_argument("--container", type=Path, required=True)

    inspect = subparsers.add_parser(
        "inspect",
        help="strict offline inspection; never emits an authorization capability",
    )
    inspect.add_argument("artifact", type=Path)

    arguments = parser.parse_args(argv)
    if arguments.command == "capture":
        result = _capability_payload(capture_live_r3_gate0_bundle(container=arguments.container))
    elif arguments.command == "verify":
        result = _capability_payload(verify_live_r3_gate0_bundle(container=arguments.container))
    else:
        report = inspect_r3_gate0_bundle(arguments.artifact)
        result = {
            "status": "validated_offline_non_authorizing_inspection",
            "schema_version": report.schema_version,
            "artifact_sha256": report.artifact_sha256,
            "file_sha256": report.file_sha256,
            "formal_authorization": report.formal_authorization,
        }
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
