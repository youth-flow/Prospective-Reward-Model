#!/usr/bin/env python3
"""Capture or verify the clean-source and live-container parts of R3 Gate 1."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

_SCRIPT_RELATIVE = Path("scripts/hpc4/capture_phase2_r3_gate1.py")
_MODULE_RELATIVE = Path("src/smart_reward/phase2_r3_gate1.py")
_MODULE_NAME = "smart_reward.phase2_r3_gate1"


def _load_gate1_module() -> types.ModuleType:
    """Load the exact colocated stdlib-only module without package startup."""

    script = Path(__file__).absolute()
    if script.is_symlink() or script.resolve(strict=True) != script:
        raise RuntimeError("Gate-1 CLI must use its canonical non-symlink path")
    repository = script.parents[2]
    if script != repository / _SCRIPT_RELATIVE:
        raise RuntimeError("Gate-1 CLI escaped its fixed repository-relative path")
    git_metadata = repository / ".git"
    if not git_metadata.exists() or git_metadata.is_symlink():
        raise RuntimeError("Gate-1 CLI must run from a real Git checkout")
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
        raise RuntimeError("Gate-1 module path is not canonical fixed repository source")
    if "smart_reward" in sys.modules or _MODULE_NAME in sys.modules:
        raise RuntimeError("refusing a preloaded smart_reward package or Gate-1 module")

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
        raise RuntimeError("could not construct the fixed Gate-1 module specification")
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
        raise RuntimeError("loaded Gate-1 module did not retain its fixed source identity")
    return module


_gate1 = _load_gate1_module()
capture_live_r3_gate1_evidence = _gate1.capture_live_r3_gate1_evidence
capture_r3_gate1_source_test_receipt = _gate1.capture_r3_gate1_source_test_receipt
inspect_r3_gate1_bundle = _gate1.inspect_r3_gate1_bundle
inspect_r3_source_test_receipt = _gate1.inspect_r3_source_test_receipt
verify_live_r3_gate1_bundle = _gate1.verify_live_r3_gate1_bundle


def _emit(value: dict[str, object]) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _inspection_payload(value: object, *, status: str) -> dict[str, object]:
    payload = {
        "status": status,
        "schema_version": value.schema_version,
        "artifact_sha256": value.artifact_sha256,
        "file_sha256": value.file_sha256,
        "source_artifact_sha256": value.source_artifact_sha256,
        "source_commit": value.source_commit,
        "formal_authorization": value.formal_authorization,
    }
    container = getattr(value, "container_artifact_sha256", None)
    if container is not None:
        payload["container_artifact_sha256"] = container
        payload["formal_path_count"] = value.formal_path_count
    else:
        payload["verification_suite_sha256"] = value.verification_suite_sha256
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "source-test",
        help="test the fixed clean repository and publish to fixed persistence",
    )

    capture = commands.add_parser(
        "capture-live",
        help="capture live HPC4 source/container verification against a source receipt",
    )
    capture.add_argument("--container", type=Path, required=True)
    capture.add_argument("--source-test-receipt-file-sha256", required=True)

    verify = commands.add_parser(
        "verify-live",
        help="reverify the caller-pinned live Gate-1 artifact",
    )
    verify.add_argument("--container", type=Path, required=True)
    verify.add_argument("--gate1-file-sha256", required=True)
    verify.add_argument("--source-test-receipt-file-sha256", required=True)

    inspect_source = commands.add_parser(
        "inspect-source-test",
        help="strict non-authorizing inspection of a copied source-test receipt",
    )
    inspect_source.add_argument("artifact", type=Path)
    inspect_source.add_argument("--expected-file-sha256")

    inspect_gate1 = commands.add_parser(
        "inspect-live",
        help="strict non-authorizing inspection of a copied Gate-1 artifact",
    )
    inspect_gate1.add_argument("artifact", type=Path)
    inspect_gate1.add_argument("--expected-file-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "source-test":
        inspection = capture_r3_gate1_source_test_receipt()
        _emit(
            _inspection_payload(
                inspection,
                status="r3_source_test_receipt_published_non_authorizing",
            )
        )
        return 0
    if arguments.command == "capture-live":
        inspection = capture_live_r3_gate1_evidence(
            container=arguments.container,
            expected_source_test_receipt_file_sha256=(arguments.source_test_receipt_file_sha256),
        )
        _emit(
            _inspection_payload(
                inspection,
                status="r3_gate1_live_evidence_published_pending_reverification",
            )
        )
        return 0
    if arguments.command == "verify-live":
        capabilities = verify_live_r3_gate1_bundle(
            container=arguments.container,
            expected_file_sha256=arguments.gate1_file_sha256,
            expected_source_test_receipt_file_sha256=(arguments.source_test_receipt_file_sha256),
        )
        _emit(
            {
                "status": "r3_gate1_live_capabilities_issued",
                "gate1_artifact_sha256": capabilities.gate1.artifact_sha256,
                "gate1_file_sha256": capabilities.gate1.file_sha256,
                "source_artifact_sha256": capabilities.source.artifact_sha256,
                "container_artifact_sha256": capabilities.container.artifact_sha256,
                "live_reverification_sha256": (capabilities.gate1.live_reverification_sha256),
            }
        )
        return 0
    if arguments.command == "inspect-source-test":
        inspection = inspect_r3_source_test_receipt(
            arguments.artifact,
            expected_file_sha256=arguments.expected_file_sha256,
        )
        _emit(
            _inspection_payload(
                inspection,
                status="r3_source_test_receipt_inspected_non_authorizing",
            )
        )
        return 0
    inspection = inspect_r3_gate1_bundle(
        arguments.artifact,
        expected_file_sha256=arguments.expected_file_sha256,
    )
    _emit(
        _inspection_payload(
            inspection,
            status="r3_gate1_evidence_inspected_non_authorizing",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
