from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/hpc4/phase2_budgeted_end_to_end.sbatch"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _embedded_python(function_name: str) -> str:
    source = _source()
    function_start = source.index(f"{function_name}() {{")
    body_start = source.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    body_end = source.index("\nPY\n}", body_start)
    return source[body_start:body_end]


def test_budgeted_sbatch_is_an_exploratory_five_seed_hpc4_array() -> None:
    source = _source()
    for directive in (
        "#SBATCH --account=sigroup",
        "#SBATCH --partition=gpu-l20",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=96G",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --no-requeue",
    ):
        assert directive in source
    assert "#SBATCH --array=" not in source  # submitter owns the array declaration.
    assert 'readonly BUDGETED_STAGE="budgeted_end_to_end"' in source
    assert 'readonly BUDGETED_EVIDENCE_ROLE="budgeted_end_to_end_exploratory_only"' in source
    assert "seeds=(20261001 20261002 20261003 20261004 20261005)" in source
    for check in (
        'SLURM_ARRAY_TASK_COUNT:-}" = 5',
        'SLURM_ARRAY_TASK_MIN:-}" = 0',
        'SLURM_ARRAY_TASK_MAX:-}" = 4',
        'SLURM_ARRAY_TASK_STEP:-}" = 1',
        'SLURM_ARRAY_TASK_ID}" =~ ^[0-4]$',
    ):
        assert check in source


def test_budgeted_sbatch_fails_closed_on_ledger_commit_and_immutable_inputs() -> None:
    source = _source()
    for required in (
        "PRORM_BUDGETED_DESIGN_SHA256",
        "PRORM_BUDGETED_BASE_CONFIG_HASH",
        "PRORM_RECOVERY_AUTHORIZATION_SHA256",
        "PRORM_OPTIMIZER_SCHEDULE_SHA256",
        "PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256",
        "PRORM_BUDGETED_FROZEN_BETA",
        "runtime_export_spec_sha256",
        "PRORM_GIT_COMMIT",
        "verify_submission_ledger(",
        "git clone --quiet --no-hardlinks --no-checkout",
        "checkout --quiet --detach",
        "committed bytes mismatch",
        "HF_HUB_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
    ):
        assert required in source
    assert "PRORM_BUDGETED_EXPORT_SPEC_SHA256" not in source
    assert (
        'runtime_export_spec="PATH=/usr/local/bin:/usr/bin:/bin,'
        "PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT}"
    ) in source
    assert "printf '%s' \"${runtime_export_spec}\" | sha256sum" in source
    assert "phase2-budgeted-end-to-end" in source
    # Post-recovery evidence is mounted read-only for recursive freeze
    # verification; no post-recovery *control plane* is imported or called.
    assert "phase2_confirmatory" not in source
    assert "phase2_campaign_finalize" not in source
    assert "resolve_phase2_campaign_registry" not in source


def test_budgeted_sbatch_orders_fresh_materialization_freeze_run_verification_and_success() -> None:
    source = _source()
    materialize = source.index("controlled-materialize")
    run = source.index("python -m smart_reward.cli phase2-run")
    verifier = source.index("verify_phase2_budgeted_end_to_end_seed_output.py")
    publish = source.index("artifact_staging=")
    assert materialize < run < verifier < publish
    assert source.index("verification_complete=1", verifier) < publish
    assert '[[ "${verification_complete}" = 1 ]] || sync=1' in source
    assert '--beta-source-aggregate "${PRORM_BUDGETED_FREEZE_EVIDENCE}"' in source
    assert '--horizon-parent-aggregate "${PRORM_BUDGETED_FREEZE_EVIDENCE}"' in source
    assert "optimizer state, or head directory is mounted or reused" in source
    assert '--bind "${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}:ro' in source
    assert "formal=false" in source
    for required_success_evidence in (
        "phase2_result_sha256",
        "rollouts_sha256",
        "verification_sha256",
        "manifest_sha256",
        '[[ -L "${project_run}/artifact" && -d "${project_run}/artifact" ]]',
    ):
        assert required_success_evidence in source


def test_budgeted_sbatch_uses_actual_sidecar_and_explicit_cleanenv_identity() -> None:
    source = _source()
    assert "phase2-result.rollouts.jsonl" in source
    assert "phase2-rollout.json" not in source
    for required in (
        '--env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"',
        '--env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"',
        '--env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"',
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "authorization-check-host.json",
        "authorization-check-container.json",
        "--expected-stage budgeted_end_to_end",
        "accepted-freeze-preflight.json",
        "budgeted-overlay-binding-check.json",
        "verify_beta_source_aggregate",
        "verify_horizon_parent_aggregate",
        "install_file_no_replace",
        "atomic_directory_publish_no_replace",
        "renameat2",
        "rename_noreplace",
        "fsync_directory",
    ):
        assert required in source


def test_ledger_precedes_seed_namespace_and_pre_oracle_gate_precedes_success() -> None:
    source = _source()
    ledger = source.index("verify_submission_ledger(")
    namespace = source.index('mkdir -- "${job_dir}" "${project_run}"')
    assert ledger < namespace
    preflight = source.index("accepted-freeze-preflight.json")
    materialize = source.index("controlled-materialize")
    assert preflight < materialize


def test_budgeted_publication_uses_only_atomic_hard_fail_no_replace_operations() -> None:
    source = _source()
    assert "mv -n" not in source
    assert "--no-clobber" not in source
    assert "mv -T" not in source
    assert "os.link(staged, target, follow_symlinks=False)" in source
    assert "raise FileExistsError" in source
    assert 'getattr(libc, "renameat2", None)' in source
    assert "rename_noreplace = 1" in source
    assert 'atomic_copy "${job_dir}/${file}" "${file}" || sync=1' in source
    assert (
        "atomic_directory_publish_no_replace \\\n"
        '  "${artifact_dir}" "${artifact_staging}" "${project_artifact}"'
    ) in source


def test_success_rereads_published_verification_and_all_four_published_hashes() -> None:
    source = _source()
    finalize = source.index("finalize() {")
    copy = source.index('atomic_copy "${job_dir}/${file}" "${file}" || sync=1', finalize)
    closure = source.index("verify_published_success_closure || sync=1", finalize)
    marker = source.index('write_marker "${marker}"', finalize)
    assert copy < closure < marker
    for required in (
        'run / "phase2-budgeted-output-verification.json"',
        'run / "phase2-result.json"',
        'run / "phase2-result.rollouts.jsonl"',
        'run / "run-manifest.json"',
        'artifact / "metadata.json"',
        'verification.get("result_sha256") != observed_hashes["result"]',
        'verification.get("rollouts_sha256") != observed_hashes["rollouts"]',
        'verification.get("run_manifest_sha256") != observed_hashes["run_manifest"]',
        'verification.get("artifact_metadata_sha256")',
        'input_sha256.get("result") != observed_hashes["result"]',
        'input_sha256.get("rollouts") != observed_hashes["rollouts"]',
        'input_sha256.get("run_manifest") != observed_hashes["run_manifest"]',
        'input_sha256.get("artifact_metadata") != observed_hashes["artifact_metadata"]',
    ):
        assert required in source
    marker_source = source[source.index("write_marker() {") : source.index("finalize() {")]
    assert "${project_run}/phase2-result.json" in marker_source
    assert "${job_dir}/phase2-result.json" not in marker_source


def test_embedded_atomic_and_success_verifiers_are_valid_python() -> None:
    for function_name in (
        "install_file_no_replace",
        "atomic_directory_publish_no_replace",
        "verify_published_success_closure",
    ):
        compile(_embedded_python(function_name), f"<{function_name}>", "exec")


def test_atomic_file_installer_treats_eexist_as_a_hard_failure(tmp_path: Path) -> None:
    if os.name == "nt":
        return
    code = _embedded_python("install_file_no_replace")
    target = tmp_path / "published.json"
    staged = tmp_path / ".published.first"
    staged.write_bytes(b"first\n")
    first = subprocess.run(
        [sys.executable, "-I", "-S", "-", str(staged), str(target)],
        input=code,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert target.read_bytes() == b"first\n"
    assert not staged.exists()

    collision = tmp_path / ".published.collision"
    collision.write_bytes(b"second\n")
    second = subprocess.run(
        [sys.executable, "-I", "-S", "-", str(collision), str(target)],
        input=code,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode != 0
    assert "refusing to replace occupied publication target" in second.stderr
    assert target.read_bytes() == b"first\n"


def test_atomic_file_installer_rolls_back_own_target_after_post_link_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = _embedded_python("install_file_no_replace")
    staged = tmp_path / ".SUCCESS.staged"
    target = tmp_path / "SUCCESS"
    staged.write_bytes(b"success\n")
    calls = 0

    def fail_target_fsync(descriptor: int) -> None:
        nonlocal calls
        assert isinstance(descriptor, int)
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "injected target fsync failure")

    monkeypatch.setattr(os, "fsync", fail_target_fsync)
    monkeypatch.setattr(sys, "argv", ["install_file_no_replace", str(staged), str(target)])

    with pytest.raises(OSError, match="injected target fsync failure"):
        exec(compile(code, "<install_file_no_replace>", "exec"), {"__name__": "__main__"})

    assert calls >= 2
    assert not target.exists()
    assert not target.is_symlink()
    assert not staged.exists()
    assert not staged.is_symlink()


def test_atomic_file_installer_never_unlinks_a_target_reported_as_a_different_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = _embedded_python("install_file_no_replace")
    staged = tmp_path / ".SUCCESS.staged"
    target = tmp_path / "SUCCESS"
    staged.write_bytes(b"success\n")
    real_lstat = Path.lstat
    calls = 0
    post_link_failure = False

    def fail_target_fsync(descriptor: int) -> None:
        nonlocal calls, post_link_failure
        assert isinstance(descriptor, int)
        calls += 1
        if calls == 2:
            post_link_failure = True
            raise OSError(errno.EIO, "injected target fsync failure")

    def report_replaced_target(path: Path):
        metadata = real_lstat(path)
        if post_link_failure and path.absolute() == target.absolute():
            fields = list(metadata)
            fields[1] = metadata.st_ino + 1
            return os.stat_result(fields)
        return metadata

    monkeypatch.setattr(os, "fsync", fail_target_fsync)
    monkeypatch.setattr(Path, "lstat", report_replaced_target)
    monkeypatch.setattr(sys, "argv", ["install_file_no_replace", str(staged), str(target)])

    with pytest.raises(OSError, match="injected target fsync failure"):
        exec(compile(code, "<install_file_no_replace>", "exec"), {"__name__": "__main__"})

    assert target.read_bytes() == b"success\n"
    assert not staged.exists()
    assert not staged.is_symlink()


def test_budgeted_sbatch_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    completed = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
