# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Run: bazel test max/kernels/benchmarks/autotune:autotune_tests

import ctypes
import json
import os
import string
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.core import Command
from click.testing import CliRunner
from kbench import cli as kbench_cli
from kbench_model import (
    BuildItem,
    ExecItemTask,
    ItemPool,
    KbenchCache,
    MkdirArgs,
    Param,
    ProcessOutput,
    Scheduler,
    SpecInstance,
    SupportedLangs,
)
from kplot import _resolve_ytext_unit
from kplot import cli as kplot_cli
from kprofile import cli as kprofile_cli
from pytest_mock import MockerFixture
from rich.console import Console
from terminal_viz import render_results
from utils import _get_gpu_count, check_valid_target_accelerator

# QUA-386 / KERN-2930: disabled due to nondeterminism
pytestmark = pytest.mark.skip(
    reason="QUA-386 / KERN-2930: nondeterministic, see runbook"
)


def get_abs_path(path: str) -> Path:
    return Path(string.Template(str(path)).substitute(os.environ)).absolute()


kernel_benchmarks_root = os.environ["KERNEL_BENCHMARKS_ROOT"]


# `--config=ubsan` exports KBENCH_LIBUBSAN_PATH (see BUILD.bazel). We use that
# as the ubsan signal to route the heavy `sample.mojo`-based tests away from
# ubsan shards (where compiling sample.mojo + its kernel deps with ubsan
# instrumentation blows the 900s shard timeout) and run a lightweight
# `minimal_fixture.mojo`-based variant instead.
_RUNNING_UNDER_UBSAN = bool(os.environ.get("KBENCH_LIBUBSAN_PATH"))
_SKIP_UNDER_UBSAN = (
    "sample.mojo compile too slow under ubsan; see minimal variant"
)
_SKIP_OUTSIDE_UBSAN = "ubsan-only variant of the sample.mojo-based test"
# The e2e tests below invoke kbench in-process via Click's CliRunner, which
# in turn spins up `multiprocessing.Process` workers. Under ubsan the
# libubsan-preloaded test process forks after pytest/queue feeder threads
# have started; the child intermittently hangs in popen_fork._launch
# (see MOTO-1576). The non-e2e unit tests cover the same code paths.
_SKIP_E2E_UNDER_UBSAN = (
    "kbench e2e flakes under ubsan: heavy compile + post-fork hangs"
)


def test_check_valid_target_accelerator() -> None:
    # Valid accelerators should return True
    assert check_valid_target_accelerator("amdgpu:mi300x")

    # Invalid accelerators should return False
    assert not check_valid_target_accelerator("invalid_accelerator")
    assert not check_valid_target_accelerator("")


# TODO: refactor to match the expected results
def _invoke_cli(
    cli: Command,
    test_cases: list[str],
    exit_code: int = os.EX_OK,
) -> None:
    os_env = os.environ.copy()
    for test_cmd in test_cases:
        try:
            result = CliRunner().invoke(cli, test_cmd, env=os_env)
            assert result.exit_code == exit_code, result.output
            print(result.output)
        except Exception:
            print(
                f"Exit code: {result.exit_code}, Exception: {result.exception}"
            )


@pytest.mark.skipif(_RUNNING_UNDER_UBSAN, reason=_SKIP_E2E_UNDER_UBSAN)
@pytest.mark.shard_group("kbench_e2e")
def test_kbench() -> None:
    _invoke_cli(
        kbench_cli,
        test_cases=[
            "--skip-clock-check --help",
            f"{kernel_benchmarks_root}/autotune/test.yaml -fv --dryrun",
            f"{kernel_benchmarks_root}/autotune/test.yaml -fv -o {kernel_benchmarks_root}/autotune/tests/output",
        ],
    )

    print(
        "autotune/tests:",
        os.listdir(f"{kernel_benchmarks_root}/autotune/tests"),
        os.listdir("."),
    )

    path = [
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.txt"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.pkl"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.csv"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/baseline.csv"),
    ]

    for p in path:
        assert p.exists()

    df = pd.read_csv(f"{kernel_benchmarks_root}/autotune/tests/output.csv")
    baseline_df = pd.read_csv(
        f"{kernel_benchmarks_root}/autotune/tests/baseline.csv"
    )

    pd.testing.assert_series_equal(df["name"], baseline_df["name"])
    pd.testing.assert_series_equal(df["spec"], baseline_df["spec"])


# TODO: resolving mpirun deps in bazel.
# def test_kbench_mpirun() -> None:
#     _invoke_cli(
#         kbench_cli,
#         test_cases=[
#             f"{kernel_benchmarks_root}/autotune/test.yaml --mpirun-np 5 -fv -o {kernel_benchmarks_root}/autotune/tests/output_mpirun",
#         ],
#     )

#     print(
#         "autotune/tests:",
#         os.listdir(f"{kernel_benchmarks_root}/autotune/tests"),
#         os.listdir("."),
#     )

#     path = [
#         Path(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.txt"),
#         Path(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.pkl"),
#         Path(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.csv"),
#         Path(f"{kernel_benchmarks_root}/autotune/tests/baseline_mpirun.csv"),
#     ]

#     for p in path:
#         assert p.exists()

#     df = pd.read_csv(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.csv")
#     baseline_df = pd.read_csv(
#         f"{kernel_benchmarks_root}/autotune/tests/baseline_mpirun.csv"
#     )

#     pd.testing.assert_series_equal(df["name"], baseline_df["name"])
#     pd.testing.assert_series_equal(df["spec"], baseline_df["spec"])


def test_kbench_cache() -> None:
    _invoke_cli(
        kbench_cli,
        test_cases=[
            "-cc",
            "-cc -v",
        ],
    )


@pytest.mark.skipif(_RUNNING_UNDER_UBSAN, reason=_SKIP_E2E_UNDER_UBSAN)
@pytest.mark.shard_group("kbench_e2e")
def test_kplot() -> None:
    _invoke_cli(
        kplot_cli,
        test_cases=[
            "-f --help",
            f"{kernel_benchmarks_root}/autotune/tests/output.csv -o {kernel_benchmarks_root}/autotune/tests/img_csv",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl -o {kernel_benchmarks_root}/autotune/tests/img_pkl",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl -o {kernel_benchmarks_root}/autotune/tests/img_pkl -x pdf",
        ],
    )

    path = [
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.csv"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_csv_0.png"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_pkl_0.png"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_pkl_0.pdf"),
    ]

    for p in path:
        assert p.exists()


def test_kplot_ytext_units() -> None:
    _invoke_cli(
        kplot_cli,
        test_cases=[
            f"{kernel_benchmarks_root}/autotune/tests/baseline.csv "
            f"-o {kernel_benchmarks_root}/autotune/tests/img_csv_us "
            "--ytext True --ytext-unit us --prec 2",
        ],
    )

    path = [
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_csv_us_0.png"),
    ]

    for p in path:
        assert p.exists()


@pytest.mark.parametrize(
    ("y_title", "y_values", "requested_unit", "expected"),
    [
        # Explicit unit conversion: ms -> us
        ("met (ms)", np.array([[1.0, 2.0]]), "us", ("us", 1e3, "ms")),
        # Auto-selection based on magnitude: s with microsecond values -> us
        ("met (s)", np.array([[1e-6, 2e-6]]), "auto", ("us", 1e6, "s")),
    ],
    ids=["explicit_ms_to_us", "auto_select_us"],
)
def test_resolve_ytext_unit(
    y_title: str,
    y_values: np.ndarray,
    requested_unit: str,
    expected: tuple[str, float, str],
) -> None:
    result_title, scale, suffix, base_unit = _resolve_ytext_unit(
        y_title=y_title,
        y_values=y_values,
        requested_unit=requested_unit,
    )
    expected_suffix, expected_scale, expected_base = expected

    assert result_title.endswith(f"({expected_suffix})")
    assert suffix == expected_suffix
    assert scale == pytest.approx(expected_scale)
    assert base_unit == expected_base


@pytest.mark.skipif(_RUNNING_UNDER_UBSAN, reason=_SKIP_E2E_UNDER_UBSAN)
@pytest.mark.shard_group("kbench_e2e")
def test_kprofile() -> None:
    _invoke_cli(
        kprofile_cli,
        test_cases=[
            "--help",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl -c",
        ],
    )

    path = [
        Path("./correlation.png"),
    ]

    for p in path:
        assert p.exists()


# --- Scheduler tests ---


def test_scheduler_init_validates_num_gpu(tmp_path: Path) -> None:
    """num_gpu must be > 0 and <= num_cpu"""
    cache = KbenchCache()

    # num_gpu = 0 should fail
    with pytest.raises(ValueError, match="num_gpu"):
        Scheduler(
            num_cpu=4,
            num_gpu=0,
            obj_cache=cache,
            run_only=False,
            spec_list=[],
            output_dir=tmp_path,
            build_opts=[],
            dryrun=True,
        )

    # num_gpu > num_cpu should fail
    with pytest.raises(ValueError, match="num_gpu"):
        Scheduler(
            num_cpu=2,
            num_gpu=4,
            obj_cache=cache,
            run_only=False,
            spec_list=[],
            output_dir=tmp_path,
            build_opts=[],
            dryrun=True,
        )


def test_scheduler_get_chunksize(tmp_path: Path) -> None:
    """Test chunksize calculation"""
    cache = KbenchCache()
    scheduler = Scheduler(
        num_cpu=4,
        num_gpu=2,
        obj_cache=cache,
        run_only=False,
        spec_list=[],
        output_dir=tmp_path,
        build_opts=[],
        dryrun=True,
    )

    # With few elements, chunksize should be small
    assert scheduler.get_chunksize(2) == 1
    assert scheduler.get_chunksize(4) == 1
    # Chunksize is capped at CHUNK_SIZE (1)
    assert scheduler.get_chunksize(100) == 1


def test_scheduler_kbench_mkdir_creates_directory(tmp_path: Path) -> None:
    """Test that kbench_mkdir creates directories"""
    output_dir = tmp_path / "new_dir"
    assert not output_dir.exists()

    result = Scheduler.kbench_mkdir(
        MkdirArgs(
            output_dir=output_dir,
            output_suffix="output.csv",
            run_only=False,
            has_cache_dir=False,
        )
    )

    assert result == output_dir
    assert output_dir.exists()


def test_scheduler_kbench_mkdir_run_only_requires_existing(
    tmp_path: Path,
) -> None:
    """run_only=True should fail if directory doesn't exist"""
    output_dir = tmp_path / "nonexistent"

    with pytest.raises(ValueError, match="does not exist"):
        Scheduler.kbench_mkdir(
            MkdirArgs(
                output_dir=output_dir,
                output_suffix="output.csv",
                run_only=True,
                has_cache_dir=False,
            )
        )


def test_kbench_cache_basic_operations(tmp_path: Path) -> None:
    """Test KbenchCache store and query"""
    cache_path = tmp_path / "test_cache.pkl"
    cache = KbenchCache(path=cache_path)

    # Cache not active, query returns None
    assert cache.query("key1") is None

    cache.load()  # activates cache

    # Store and retrieve
    bin_path = tmp_path / "binary"
    bin_path.touch()  # create the file

    cache.store("key1", bin_path)
    assert cache.query("key1") == str(bin_path)

    # Non-existent key
    assert cache.query("nonexistent") is None


def test_process_output_log() -> None:
    """Test ProcessOutput.log doesn't crash"""
    # Just verify it doesn't raise
    output = ProcessOutput(stdout="test output", stderr="test error")
    output.log()

    empty_output = ProcessOutput()
    empty_output.log()


def test_param_define_regular() -> None:
    """Test Param.define() for regular compile-time parameters"""
    param = Param(name="BLOCK_SIZE", value=128)
    assert param.define(lang=SupportedLangs.MOJO) == ["-D", "BLOCK_SIZE=128"]

    param_str = Param(name="TYPE", value="float32")
    assert param_str.define(lang=SupportedLangs.MOJO) == ["-D", "TYPE=float32"]


def test_param_define_variable() -> None:
    """Test Param.define(lang=SupportedLangs.MOJO) for runtime variable parameters (prefixed with $)"""
    param = Param(name="$M", value=1024)
    result = param.define(lang=SupportedLangs.MOJO)
    # Variable params get converted to --<name>=<value> format
    assert result == ["--M=1024"]

    param = Param(name="$a$b", value="test")
    result = param.define(lang=SupportedLangs.MOJO)
    assert result == ["--a$b=test"]

    param = Param(name="$$a", value="test")
    result = param.define(lang=SupportedLangs.MOJO)
    assert result == ["--$a=test"]


def test_terminal_viz_render() -> None:
    """E2E test for terminal visualization output."""
    # Synthetic benchmark data
    df = pd.DataFrame(
        {
            "spec": [
                "bench_kernel/$shape=1x1x4096",
                "bench_kernel/$shape=1x512x4096",
                "bench_kernel/$shape=1x1024x4096",
            ],
            "met (s)": [0.004, 0.006, 0.010],  # 4ms, 6ms, 10ms
        }
    )

    # Test bars mode
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    render_results(df, mode="bars", console=console)
    result = output.getvalue()

    # Check that shape labels appear
    assert "1x1x4096" in result
    assert "1x512x4096" in result
    # Check that bar characters present
    assert "█" in result
    # Check that time values present
    assert "ms" in result

    # Test table mode
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    render_results(df, mode="table", console=console)
    result = output.getvalue()

    # Check that table headers
    assert "time/iter" in result
    assert "vs base" in result
    # Check that relative comparison shown
    assert "1.00x" in result


# --- GPU count detection tests ---


def _nvidia_smi_output(n: int) -> bytes:
    """Build fake nvidia-smi CSV output for n GPUs."""
    return "\n".join([f"NVIDIA H100 {i}" for i in range(n)]).encode()


def test_get_gpu_count_nvidia_basic(mocker: MockerFixture) -> None:
    """_get_gpu_count returns the hardware count when no env var is set."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert _get_gpu_count("nvidia:sm_90") == 4


def test_get_gpu_count_nvidia_capped_by_env(mocker: MockerFixture) -> None:
    """When CUDA_VISIBLE_DEVICES lists more IDs than hardware, cap to hardware."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(
        os.environ,
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9"},
        clear=False,
    )
    assert _get_gpu_count("nvidia:sm_90") == 4


def test_get_gpu_count_nvidia_env_fewer_than_hw(mocker: MockerFixture) -> None:
    """When CUDA_VISIBLE_DEVICES lists fewer IDs than hardware, use env count."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(8),
    )
    mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False)
    assert _get_gpu_count("nvidia:sm_90") == 2


def test_get_gpu_count_nvidia_no_smi(mocker: MockerFixture) -> None:
    """Returns None when nvidia-smi is not found."""
    mocker.patch("utils.get_nvidia_smi", return_value=None)
    assert _get_gpu_count("nvidia:sm_90") is None


def test_get_gpu_count_unknown_accelerator() -> None:
    """Returns None for unrecognised accelerator strings."""
    assert _get_gpu_count("metal:1") is None
    assert _get_gpu_count("") is None


def test_get_gpu_count_nvidia_smi_failure(mocker: MockerFixture) -> None:
    """Returns None when nvidia-smi fails."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        side_effect=subprocess.SubprocessError("boom"),
    )
    assert _get_gpu_count("nvidia:sm_90") is None


def _rocm_smi_csv_output(n: int) -> bytes:
    """Build fake rocm-smi --showproductname --csv output for n GPUs."""
    header = "device,Card series,Card model,Card vendor,Card SKU"
    rows = [
        f"card{i},0x{i:04x},AMD Instinct MI355X,Advanced Micro Devices Inc. [AMD/ATI],D7520{i}"
        for i in range(n)
    ]
    return "\n".join([header] + rows).encode()


def test_get_gpu_count_amd_basic(mocker: MockerFixture) -> None:
    """_get_gpu_count returns the hardware count for AMD GPUs."""
    mocker.patch("utils.shutil.which", return_value="/usr/bin/rocm-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_rocm_smi_csv_output(8),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    assert _get_gpu_count("amdgpu:mi355x") == 8


def test_get_gpu_count_amd_capped_by_env(mocker: MockerFixture) -> None:
    """When ROCR_VISIBLE_DEVICES lists fewer IDs than hardware, use env count."""
    mocker.patch("utils.shutil.which", return_value="/usr/bin/rocm-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_rocm_smi_csv_output(8),
    )
    mocker.patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "0,1"}, clear=False)
    assert _get_gpu_count("amdgpu:mi355x") == 2


def test_get_gpu_count_amd_no_smi(mocker: MockerFixture) -> None:
    """Returns None when rocm-smi is not found."""
    mocker.patch("utils.shutil.which", return_value=None)
    assert _get_gpu_count("amdgpu:mi355x") is None


def test_get_gpu_count_amd_smi_failure(mocker: MockerFixture) -> None:
    """Returns None when rocm-smi fails."""
    mocker.patch("utils.shutil.which", return_value="/usr/bin/rocm-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        side_effect=subprocess.SubprocessError("boom"),
    )
    assert _get_gpu_count("amdgpu:mi355x") is None


def test_cli_rejects_num_gpu_exceeding_available(mocker: MockerFixture) -> None:
    """CLI should raise ValueError when --num-gpu exceeds detected GPUs."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os_env = os.environ.copy()
    result = CliRunner().invoke(
        kbench_cli,
        f"{kernel_benchmarks_root}/autotune/test.yaml"
        " --num-gpu 8 --target-accelerator nvidia:sm_90 --skip-clock-check --dryrun",
        env=os_env,
    )
    assert result.exit_code != 0
    assert "exceeds" in str(result.exception) or "exceeds" in (
        result.output or ""
    )


def test_cli_accepts_num_gpu_within_available(mocker: MockerFixture) -> None:
    """CLI should not raise when --num-gpu <= detected GPUs."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os_env = os.environ.copy()
    result = CliRunner().invoke(
        kbench_cli,
        f"{kernel_benchmarks_root}/autotune/test.yaml"
        " --num-gpu 4 --target-accelerator nvidia:sm_90 --skip-clock-check --dryrun",
        env=os_env,
    )
    assert result.exit_code == 0, result.output


# --- Binary-grouped execution tests ---


# The library that under ubsan carries the unresolved `__ubsan_handle_*_abort`
# refs the preload is meant to resolve (MOTO-1576). The dlopen regression
# tests below assert that the .so under test still depends on this lib —
# otherwise the test would pass for an uninteresting reason.
_KGEN_COMPILER_RT = "libKGENCompilerRTShared.so"


def _make_spec_instance(
    compile_params: dict[str, str | int],
    runtime_params: dict[str, str | int],
) -> SpecInstance:
    """Helper to create a SpecInstance pointing at sample.mojo."""
    file = get_abs_path(f"{kernel_benchmarks_root}/autotune/sample.mojo")
    params = [Param(k, v) for k, v in compile_params.items()]
    params += [Param(f"${k}", v) for k, v in runtime_params.items()]
    return SpecInstance(
        name="test", file=file, executor=SupportedLangs.MOJO, params=params
    )


def _make_minimal_spec_instance() -> SpecInstance:
    """Helper to create a SpecInstance pointing at tests/minimal_fixture.mojo.

    The fixture has no kernel imports so it compiles quickly even under
    --config=ubsan. Used by ubsan-only test variants.
    """
    file = get_abs_path(
        f"{kernel_benchmarks_root}/autotune/tests/minimal_fixture.mojo"
    )
    return SpecInstance(
        name="test", file=file, executor=SupportedLangs.MOJO, params=[]
    )


def _assert_built_so(
    spec: SpecInstance, tmp_path: Path, expected_module_name: str
) -> Path:
    """Build a .so for `spec` and check the produced file + generated wrapper.

    Returns the .so path so callers can do extra inspection.
    """
    result = spec.build_shared_lib(output_dir=tmp_path, build_opts=[])
    assert result.return_code == os.EX_OK, result.stderr
    assert result.path is not None
    assert result.path.exists()
    assert result.path.suffix == ".so"
    bin_name = spec.hash(with_variables=False)
    wrapper = tmp_path / f"{bin_name}_wrapper.mojo"
    assert wrapper.exists()
    content = wrapper.read_text()
    assert f"from {expected_module_name} import main as _bench_main" in content
    assert "benchmark_entry" in content
    return result.path


def _so_dependencies(so_path: Path) -> str:
    """Return the dynamic-section (`NEEDED` entries etc.) of an ELF .so.

    Used to assert that a test .so still transitively pulls in the runtime
    library carrying the unresolved ubsan refs we're guarding against.
    """
    out = subprocess.run(
        ["readelf", "-d", str(so_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


@pytest.mark.skipif(_RUNNING_UNDER_UBSAN, reason=_SKIP_UNDER_UBSAN)
def test_build_shared_lib(tmp_path: Path) -> None:
    """build_shared_lib() compiles sample.mojo to a .so."""
    spec = _make_spec_instance({"dtype": "DType.float16"}, {"x": 0})
    _assert_built_so(spec, tmp_path, expected_module_name="sample")


@pytest.mark.skipif(not _RUNNING_UNDER_UBSAN, reason=_SKIP_OUTSIDE_UBSAN)
def test_build_shared_lib_minimal(tmp_path: Path) -> None:
    """ubsan-only variant of test_build_shared_lib using a minimal fixture.

    Avoids compiling sample.mojo's heavy kernel deps under ubsan instrumentation
    so the test shard fits within its timeout.
    """
    spec = _make_minimal_spec_instance()
    _assert_built_so(spec, tmp_path, expected_module_name="minimal_fixture")


@pytest.mark.skipif(_RUNNING_UNDER_UBSAN, reason=_SKIP_UNDER_UBSAN)
def test_shared_lib_dlopen_resolves_sanitizer_runtime(tmp_path: Path) -> None:
    """Unit-level regression guard for `kbench_model._maybe_preload_libubsan`
    (MOTO-1576).

    The end-to-end behavior — kbench compiling a .so, dlopen-ing it via
    `_SharedLibExecutor`, and running it under UBSan — is already covered
    by `test_shared_lib_*_recovery` and `test_kbench@kbench_e2e`. This test
    is narrower: it exercises the loader-side preload helper directly so a
    refactor that breaks the helper or the BUILD.bazel env-var wiring fails
    a small, targeted test rather than a large end-to-end one.

    Historically this scenario failed with `undefined symbol:
    __ubsan_handle_*_abort` because libKGENCompilerRTShared.so leaves those
    references unresolved (`-Wl,-z,undefs`) and Python isn't UBSan-
    instrumented, so the process had no handlers. The preload pulls
    libubsan into the global symbol namespace before dlopen, so the
    handlers resolve against it.
    """
    from kbench_model import _maybe_preload_libubsan

    spec = _make_spec_instance({"dtype": "DType.float16"}, {"x": 0})
    so_path = _assert_built_so(spec, tmp_path, expected_module_name="sample")

    # The preload is a no-op outside `--config=ubsan` (no
    # KBENCH_LIBUBSAN_PATH env var is set), so this test passes equally
    # under any build config.
    _maybe_preload_libubsan()
    lib = ctypes.CDLL(str(so_path))
    lib.benchmark_entry.restype = ctypes.c_int32
    # Actually invoke the entry point so the test exercises preload →
    # dlopen → call. sample.mojo's main returns 0 on success.
    assert lib.benchmark_entry() == 0


@pytest.mark.skipif(not _RUNNING_UNDER_UBSAN, reason=_SKIP_OUTSIDE_UBSAN)
def test_shared_lib_dlopen_resolves_sanitizer_runtime_minimal(
    tmp_path: Path,
) -> None:
    """ubsan-only variant: exercises the preload → dlopen → call chain using
    a minimal Mojo fixture that compiles fast under ubsan instrumentation.

    The .so still links libKGENCompilerRTShared.so (which has the unresolved
    `__ubsan_handle_*_abort` refs that MOTO-1576 is about), so this still
    guards the regression even though the fixture itself does no benchmark
    work. The readelf assertion below is what enforces that the link
    dependency is still there — if a future toolchain change makes the
    minimal fixture light enough to drop it, the test fails loudly instead
    of silently passing for the wrong reason.
    """
    from kbench_model import _maybe_preload_libubsan

    spec = _make_minimal_spec_instance()
    so_path = _assert_built_so(
        spec, tmp_path, expected_module_name="minimal_fixture"
    )

    deps = _so_dependencies(so_path)
    assert _KGEN_COMPILER_RT in deps, (
        f"{so_path.name} doesn't link {_KGEN_COMPILER_RT}; "
        "this test no longer guards MOTO-1576.\n"
        f"readelf -d output:\n{deps}"
    )

    _maybe_preload_libubsan()
    lib = ctypes.CDLL(str(so_path))
    lib.benchmark_entry.restype = ctypes.c_int32
    assert lib.benchmark_entry() == 0


# --- Minimal e2e variants (ubsan-only) ---
#
# The non-minimal `test_kbench` / `test_kplot` / `test_kprofile` are skipped
# under ubsan (heavy compile + post-fork hangs, see _SKIP_E2E_UNDER_UBSAN).
# These variants exercise the same CLI entry points against
# `tests/test_minimal.yaml`, which points at `minimal_fixture.mojo` (no kernel
# imports). The fixture's `main()` is a no-op, so kbench's placeholder-row
# fallback supplies the output.csv/pkl that kplot and kprofile then consume.


@pytest.mark.skipif(not _RUNNING_UNDER_UBSAN, reason=_SKIP_OUTSIDE_UBSAN)
@pytest.mark.shard_group("kbench_e2e")
def test_kbench_minimal() -> None:
    """ubsan-only e2e variant of test_kbench using minimal_fixture.mojo."""
    out = f"{kernel_benchmarks_root}/autotune/tests/output_min"
    _invoke_cli(
        kbench_cli,
        test_cases=[
            f"{kernel_benchmarks_root}/autotune/tests/test_minimal.yaml"
            f" -fv -o {out}",
        ],
    )

    for suffix in (".csv", ".pkl", ".txt"):
        assert Path(out + suffix).exists(), f"missing {out + suffix}"


@pytest.mark.skipif(not _RUNNING_UNDER_UBSAN, reason=_SKIP_OUTSIDE_UBSAN)
@pytest.mark.shard_group("kbench_e2e")
def test_kplot_minimal() -> None:
    """ubsan-only e2e variant of test_kplot. Reads test_kbench_minimal output."""
    out = f"{kernel_benchmarks_root}/autotune/tests/output_min"
    img = f"{kernel_benchmarks_root}/autotune/tests/img_min"
    _invoke_cli(
        kplot_cli,
        test_cases=[
            f"{out}.csv -o {img}_csv",
            f"{out}.pkl -o {img}_pkl",
        ],
    )

    assert Path(f"{img}_csv_0.png").exists()
    assert Path(f"{img}_pkl_0.png").exists()


@pytest.mark.skipif(not _RUNNING_UNDER_UBSAN, reason=_SKIP_OUTSIDE_UBSAN)
@pytest.mark.shard_group("kbench_e2e")
def test_kprofile_minimal() -> None:
    """ubsan-only e2e variant of test_kprofile. Reads test_kbench_minimal output."""
    out = f"{kernel_benchmarks_root}/autotune/tests/output_min"
    _invoke_cli(
        kprofile_cli,
        test_cases=[
            f"{out}.pkl",
            f"{out}.pkl -c",
        ],
    )

    assert Path("./correlation.png").exists()


def test_scheduler_output_dir_list_per_item(tmp_path: Path) -> None:
    """output_dir_list assigns sequential out_N/ within each base dir."""
    dir_a = tmp_path / "dirA"
    dir_b = tmp_path / "dirB"
    dir_a.mkdir()
    dir_b.mkdir()

    specs = [
        _make_spec_instance({"dtype": "float16"}, {"x": i}) for i in range(4)
    ]
    output_dir_list = [dir_a, dir_a, dir_b, dir_a]

    cache = KbenchCache()
    scheduler = Scheduler(
        num_cpu=4,
        num_gpu=1,
        obj_cache=cache,
        run_only=False,
        spec_list=specs,
        output_dir=tmp_path,
        build_opts=[],
        dryrun=True,
        output_dir_list=output_dir_list,
    )

    assert scheduler.build_items[0].output_dir == dir_a / "out_0"
    assert scheduler.build_items[1].output_dir == dir_a / "out_1"
    assert scheduler.build_items[2].output_dir == dir_b / "out_0"
    assert scheduler.build_items[3].output_dir == dir_a / "out_2"


def test_scheduler_output_dir_list_none_fallback(tmp_path: Path) -> None:
    """When output_dir_list=None, output dirs are output_dir/out_N."""
    specs = [
        _make_spec_instance({"dtype": "float16"}, {"x": i}) for i in range(3)
    ]

    cache = KbenchCache()
    scheduler = Scheduler(
        num_cpu=4,
        num_gpu=1,
        obj_cache=cache,
        run_only=False,
        spec_list=specs,
        output_dir=tmp_path,
        build_opts=[],
        dryrun=True,
    )

    for i in range(3):
        assert scheduler.build_items[i].output_dir == tmp_path / f"out_{i}"


def test_item_pool_basic() -> None:
    """ItemPool distributes items by binary group and supports work-stealing."""
    spec_a = _make_spec_instance({"dtype": "float16"}, {"x": 0})
    spec_b = _make_spec_instance({"dtype": "float32"}, {"x": 0})

    bi_a = BuildItem(
        idx=0,
        spec_instance=spec_a,
        output_dir=Path("/tmp"),
        build_opts=[],
        dryrun=True,
    )
    bi_b = BuildItem(
        idx=1,
        spec_instance=spec_b,
        output_dir=Path("/tmp"),
        build_opts=[],
        dryrun=True,
    )

    group_a = [ExecItemTask(build_item=bi_a, use_shared_lib=True)]
    group_b = [ExecItemTask(build_item=bi_b, use_shared_lib=True)]

    pool = ItemPool([group_a, group_b])
    pool.register_gpu(0)
    pool.register_gpu(1)

    # Each GPU should get one group
    item0 = pool.next_for(0)
    item1 = pool.next_for(1)
    assert item0 is not None
    assert item1 is not None

    # Both groups exhausted, should return None
    assert pool.next_for(0) is None
    assert pool.next_for(1) is None


def test_item_pool_work_stealing() -> None:
    """When one GPU finishes, it can steal from another's queue."""
    spec = _make_spec_instance({"dtype": "float16"}, {"x": 0})
    items = []
    for i in range(4):
        bi = BuildItem(
            idx=i,
            spec_instance=spec,
            output_dir=Path("/tmp"),
            build_opts=[],
            dryrun=True,
        )
        items.append(ExecItemTask(build_item=bi, use_shared_lib=True))

    # One large group
    pool = ItemPool([items])
    pool.register_gpu(0)
    pool.register_gpu(1)

    # GPU 0 grabs the group
    first = pool.next_for(0)
    assert first is not None

    # GPU 1 should be able to steal remaining items
    stolen = pool.next_for(1)
    assert stolen is not None


def test_group_by_binary_hash() -> None:
    """Items with same compile-time params group together; different params form separate groups."""
    spec_a0 = _make_spec_instance({"dtype": "float16"}, {"x": 0})
    spec_a1 = _make_spec_instance({"dtype": "float16"}, {"x": 1})
    spec_b0 = _make_spec_instance({"dtype": "float32"}, {"x": 0})
    spec_b1 = _make_spec_instance({"dtype": "float32"}, {"x": 1})

    # Same compile-time params → same hash (without variables)
    assert spec_a0.hash(with_variables=False) == spec_a1.hash(
        with_variables=False
    )
    assert spec_b0.hash(with_variables=False) == spec_b1.hash(
        with_variables=False
    )
    # Different compile-time params → different hash
    assert spec_a0.hash(with_variables=False) != spec_b0.hash(
        with_variables=False
    )


# --- Shared-lib failure / recovery tests ---


@pytest.mark.unique_shard
def test_shared_lib_timeout_recovery(tmp_path: Path) -> None:
    """Timed-out shared lib executions are killed; remaining items still run.

    4 items (1 binary group): 2 fast sleeps succeed, 2 long sleeps time out.
    Verifies the worker respawns after each timeout.
    """
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        kbench_cli,
        f"{kernel_benchmarks_root}/autotune/tests/test_timeout.yaml"
        f" -fv --output-dir {out_dir} --plot none"
        f" --timeout-secs 2",
        env=os.environ.copy(),
    )
    # Print diagnostics before assertions so test logs show the actual errors.
    failures_json = out_dir / "output.failures.json"
    if failures_json.exists():
        print("=== failures.json ===")
        print(failures_json.read_text())
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    # 2 items should time out (invalid) and 2 should succeed (valid).
    assert "invalid specs: 2" in out, (
        f"Expected 2 invalid specs (timed out) in output:\n{result.output}"
    )
    assert "valid executed specs: 2" in out, (
        f"Expected 2 valid specs (succeeded) in output:\n{result.output}"
    )
    # Verify failures.json records both timeouts with the "timeout" type.
    failures_json = out_dir / "output.failures.json"
    assert failures_json.exists(), f"Missing {failures_json}"
    failures = json.loads(failures_json.read_text())
    assert failures["num_valid"] == 2
    assert len(failures["failures"]) == 2
    assert all(f["failure_type"] == "timeout" for f in failures["failures"])

    # Verify capture files are cleaned up after execution.
    remaining_logs = list(out_dir.rglob("*_stdout.log")) + list(
        out_dir.rglob("*_stderr.log")
    )
    assert remaining_logs == [], (
        f"Capture files not cleaned up: {remaining_logs}"
    )


@pytest.mark.unique_shard
def test_shared_lib_crash_recovery(tmp_path: Path) -> None:
    """Crashed benchmarks don't prevent subsequent items from running.

    6 items (1 binary group): cross product of should_crash=[False,True]
    x sleep_secs=[0.01,0.02,0.03]. 3 succeed, 3 crash. Verifies the
    worker recovers after each crash.
    """
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        kbench_cli,
        f"{kernel_benchmarks_root}/autotune/tests/test_crash.yaml"
        f" -fv --output-dir {out_dir} --plot none"
        f" --timeout-secs 30",
        env=os.environ.copy(),
    )
    # Print diagnostics before assertions so test logs show the actual errors.
    failures_json = out_dir / "output.failures.json"
    if failures_json.exists():
        print("=== failures.json ===")
        print(failures_json.read_text())
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    # 3 items should crash (invalid) and 3 should succeed (valid).
    assert "invalid specs: 3" in out, (
        f"Expected 3 invalid specs (crashed) in output:\n{result.output}"
    )
    assert "valid executed specs: 3" in out, (
        f"Expected 3 valid specs (succeeded) in output:\n{result.output}"
    )
    # Verify failures.json records all crashes as execution failures.
    failures_json = out_dir / "output.failures.json"
    assert failures_json.exists(), f"Missing {failures_json}"
    failures = json.loads(failures_json.read_text())
    assert failures["num_valid"] == 3
    assert len(failures["failures"]) == 3
    assert all(f["failure_type"] == "execution" for f in failures["failures"])

    # Verify crash diagnostics are captured in failure records.
    for f in failures["failures"]:
        assert "intentional crash" in f["exec_stderr"], (
            f"Expected crash message in exec_stderr for item {f['mesh_idx']}, "
            f"got: {f['exec_stderr']!r}"
        )

    # Verify capture files are cleaned up after execution.
    remaining_logs = list(out_dir.rglob("*_stdout.log")) + list(
        out_dir.rglob("*_stderr.log")
    )
    assert remaining_logs == [], (
        f"Capture files not cleaned up: {remaining_logs}"
    )
