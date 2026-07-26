from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_budgeted_end_to_end_finalize.sh"
FINALIZER = ROOT / "scripts" / "hpc4" / "phase2_budgeted_end_to_end_finalize.sbatch"
MATERIALIZER = ROOT / "scripts" / "hpc4" / "phase2_budgeted_end_to_end_materialize.sbatch"


def test_finalizer_is_single_submission_after_any_array_outcome() -> None:
    source = SUBMIT.read_text(encoding="utf-8")

    assert '--dependency="afterany:${array_job_id}"' in source
    assert "afterok:" not in source
    assert 'submission_lock="${campaign}/finalizer-submit.lock"' in source
    assert 'mkdir -- "${submission_lock}"' in source
    assert '"${submission_lock}/job-id"' in source
    assert "GateE finalizer evidence already exists" in source
    assert "--kill-on-invalid-dep=yes" in source
    assert "--partition=gpu-l20" in source
    assert "--time=00:30:00" in source


def test_finalizer_persists_raw_sacct_before_success_interpretation() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    raw = source.index("capture-raw")
    interpreted = source.index(
        'capture "${PRORM_BUDGETED_ARRAY_JOB_ID}" "${terminal}"',
    )
    aggregate = source.index("aggregate_phase2_budgeted_end_to_end.py")

    assert raw < interpreted < aggregate
    assert (
        'raw_terminal="${terminal_parent}/array-${PRORM_BUDGETED_ARRAY_JOB_ID}.all.sacct.psv"'
    ) in source
    assert "seeds=(20261001 20261002 20261003)" in source
    assert "for task in 0 1 2; do" in source
    assert '"${runs[@]}"' in source
    assert (
        'output="${aggregate_root}/phase2-budgeted-end-to-end-${PRORM_BUDGETED_DESIGN_SHA256}.json"'
    ) in source


def test_final_aggregation_is_containerized_with_minimal_campaign_mounts() -> None:
    source = FINALIZER.read_text(encoding="utf-8")

    for required in (
        "apptainer exec",
        "--cleanenv",
        "--no-mount home,cwd,bind-paths",
        '--bind "${repo_root}:${repo_root}:ro"',
        '--bind "${campaign}:${campaign}:ro"',
        '--bind "${artifact_campaign}:${artifact_campaign}:ro"',
        '--bind "${aggregate_root}:${aggregate_root}:rw"',
        '--terminal-evidence "${terminal}"',
        '--terminal-evidence-sha256 "${terminal_sha256}"',
        '--array-job-id "${PRORM_BUDGETED_ARRAY_JOB_ID}"',
    ):
        assert required in source
    assert '--bind "${project_root}:${project_root}' not in source


def test_materializer_and_finalizer_pin_host_helpers_to_verified_312() -> None:
    fixed_python = (
        "/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/"
        "miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
    )
    fixed_sha256 = "9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"
    materializer = MATERIALIZER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")

    for source in (materializer, finalizer):
        assert f'readonly HOST_PYTHON="{fixed_python}"' in source
        assert f'readonly HOST_PYTHON_SHA256="{fixed_sha256}"' in source
        assert "sha256sum --check --status" in source
        assert '"Python 3.12.2"' in source
        assert "python3 -I -S" not in source
    assert ('"${host_python}" -I -S "${repo_root}/${BIND_PLAN_RELATIVE}"') in materializer
    assert (
        '"${host_python}" -I -S \\\n'
        '  "${repo_root}/scripts/hpc4/capture_phase2_budgeted_end_to_end_terminal.py"'
    ) in finalizer
