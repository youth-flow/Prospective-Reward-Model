from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_terminalize.sh"
DRIVER = ROOT / "scripts" / "hpc4" / "phase2_r3_terminalize.sbatch"
STDLIB = ROOT / "scripts" / "hpc4" / "phase2_r3_terminalize_stdlib.py"
VERIFIER = ROOT / "scripts" / "hpc4" / "verify_phase2_r3_terminalized.py"


def test_terminalize_launcher_is_dependency_bound_blocking_l20_srun() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "srun \\" in text
    assert '--dependency="afterany:${dependency_job_id}"' in text
    assert "--partition=gpu-l20" in text
    assert "--gpus-per-node=1" in text
    assert "--time=00:30:00" in text
    assert "sbatch " not in text
    assert "9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c" in text
    assert 'git -C "${repo_root}" status --porcelain --untracked-files=all' in text


def test_compute_driver_captures_before_sif_finalize_and_always_revalidates() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    capture_index = text.index('"${route_helper}" capture')
    container_index = text.index("container=(")
    assert capture_index < container_index
    assert '"${capture_cli}" profile-finalize' in text
    assert '"${PRORM_R3_TERMINALIZE_FINALIZER_COMMAND}"' in text
    assert "verify_phase2_r3_terminalized.py" in text
    assert "--cleanenv" in text
    assert '--bind "${repo_root}:${repo_root}:ro"' in text
    assert '--bind "${project_root}:${project_root}:rw"' not in text
    assert 'terminal_readonly_paths=("${operational_bundle}" "${raw_sacct}")' in text
    assert 'terminal_readonly_paths+=("${allocation_intent}" "${runtime_receipt}")' in text
    assert 'terminal_readonly_paths+=("${runtime_closure}")' in text
    assert 'container+=(--bind "${path}:${path}:ro")' in text
    assert '--bind "${terminal_evidence_parent}:${terminal_evidence_parent}:rw"' in text
    assert "R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY" in text
    assert "R3_TASK_%s_TERMINAL_EVIDENCE_DIRECTORY" in text


def test_stdlib_helper_does_not_import_model_dependencies() -> None:
    text = STDLIB.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "import yaml" not in text
    assert "from smart_reward" not in text
    assert '"terminal-evidence" / f"task-{task_id}-segment-{segment_index}"' in text
    assert '"terminal-evidence" / "profile"' in text


def test_reuse_verifier_uses_only_sealed_core_revalidators() -> None:
    text = VERIFIER.read_text(encoding="utf-8")
    assert "revalidate_successful_profile_terminal" in text
    assert "revalidate_continuable_primary_terminal" in text
    assert "revalidate_completed_primary_terminal" in text
    assert '_EXPECTED_ENTRIES = {"raw-sacct.psv", "parsed-sacct.json",' in text
