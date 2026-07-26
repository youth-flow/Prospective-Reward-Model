from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts" / "hpc4" / "phase2_r3_gate1.sbatch"
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_gate1.sh"


def _text(path: Path) -> str:
    raw = path.read_bytes()
    assert raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r" not in raw
    return raw.decode("utf-8")


def test_gate1_sbatch_has_one_complete_shebang_and_bounded_resources() -> None:
    text = _text(SBATCH)
    lines = text.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert lines[1].startswith("# ")
    assert lines[1] != "bash"
    assert sum(line == "#!/usr/bin/env bash" for line in lines) == 1
    assert "#!/usr/bin/env\nbash" not in text
    assert "#SBATCH --account=sigroup" in text
    assert "#SBATCH --partition=gpu-l20" in text
    assert "#SBATCH --gpus-per-node=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=96G" in text
    assert "#SBATCH --time=0-12:00:00" in text
    assert "#SBATCH --signal=B:TERM@300" in text
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --requeue" not in text


def test_gate1_submit_passes_the_tracked_file_and_never_uses_wrap_or_stdin() -> None:
    text = _text(SUBMIT)
    assert "--wrap" not in text
    assert 'export_spec="PATH=/usr/bin:/bin"' in text
    assert "--export=NONE" not in text
    assert 'SBATCH_RELATIVE="scripts/hpc4/phase2_r3_gate1.sbatch"' in text
    assert re.search(r"sbatch \\\n\s+--parsable \\\n\s+--export=", text)
    assert '"${sbatch_script}"' in text
    assert 'first_line="$(sed -n \'1p\' -- "${sbatch_script}")"' in text
    assert "[[ \"${first_line}\" == '#!/usr/bin/env bash' ]]" in text
    assert '[[ "${second_line}" != "bash" ]]' in text
    assert "<<EOF" not in text and "<<'EOF'" not in text
    assert "sbatch <" not in text


def test_gate1_surfaces_bind_clean_commit_container_and_source_test_chain() -> None:
    submit = _text(SUBMIT)
    batch = _text(SBATCH)
    for text in (submit, batch):
        assert 'git -C "${repo_root}" rev-parse HEAD' in text
        assert 'git -C "${repo_root}" status --porcelain --untracked-files=all' in text
        assert 'git -C "${repo_root}" show "${commit}:${relative}"' in text
        assert "d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb" in text
        assert "r3-source-test-receipt.json" in text
        assert "r3-implementation-closure.json" in text
    assert "source-test receipt already exists; Gate-1 is no-overwrite" in submit
    assert "Gate-1 artifact already exists; Gate-1 is no-overwrite" in submit

    source_test = batch.index('"${container_python[@]}" "${capture_cli}" source-test')
    capture_live = batch.index('"${host_python}" "${capture_cli}" capture-live')
    verify_live = batch.index('"${host_python}" "${capture_cli}" verify-live')
    assert source_test < capture_live < verify_live
    assert '--source-test-receipt-file-sha256 "${source_receipt_sha256}"' in batch
    assert '--gate1-file-sha256 "${gate1_file_sha256}"' in batch


def test_gate1_batch_uses_sif_tests_and_host_control_python() -> None:
    text = _text(SBATCH)
    submit = _text(SUBMIT)
    assert "/scratch/yyangjo/r3-gate1-tools-ruff01522-pytest744" in text
    assert "64aae5e444938e33121c3b940dff9b3d8ef8fc2a88c477e7f3a4fae2584a8fe8" in text
    assert "miniconda3/24.3.0-quc3pyu" in text
    host_python = (
        "/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/"
        "miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
    )
    symlink_python = host_python.removesuffix("3.12")
    for surface in (text, submit):
        assert f'readonly HOST_PYTHON="{host_python}"' in surface
        assert f'readonly HOST_PYTHON="{symlink_python}"' not in surface
    assert "9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c" in text
    assert 'readonly APPTAINER="/usr/bin/apptainer"' in text
    assert 'readonly SIF_PYTHON="/opt/conda/bin/python"' in text
    assert '--bind "${scratch_tools}:${scratch_tools}:ro"' in text
    assert '--env "PATH=${container_path}"' in text
    assert '--env "PYTHONPATH=${container_pythonpath}"' in text
    assert '"${container_python[@]}" "${capture_cli}" source-test' in text
    assert '"${host_python}" "${capture_cli}" capture-live' in text
    assert '"${host_python}" "${capture_cli}" verify-live' in text
    assert ".venv/" not in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert "export PYTHONDONTWRITEBYTECODE=1" in text
    assert 'export PYTHONPYCACHEPREFIX="${host_pycache}"' in text
    assert 'container_pycache="/tmp/prorm-r3-gate1-' in text


def test_gate1_submit_prepares_core_compatible_output_modes() -> None:
    text = _text(SUBMIT)
    assert 'mkdir -m 0750 -- "${path}"' in text
    assert 'chmod 0750 -- "${path}"' in text
    assert '"${mode}" != "750" && "${mode}" != "2750"' in text
    assert 'ensure_real_directory "R3 output root" "${r3_root}" "r3"' in text
    assert 'ensure_real_directory "Gate-1 output root" "${gate1_root}" "r3"' in text
    assert 'ensure_real_directory "Gate-1 log directory" "${logs}" "r3"' in text


def test_gate1_shell_surfaces_pass_bash_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed in this Windows test environment")
    subprocess.run(
        (bash, "-n", str(SUBMIT), str(SBATCH)),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
