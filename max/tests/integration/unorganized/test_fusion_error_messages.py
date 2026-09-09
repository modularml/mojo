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
"""End-to-end tests for improved fusion error messages on GPU.

These tests build graphs with various fusion patterns, trigger intentional
GPU crashes, and verify the error messages contain Python source tracebacks
and useful context about fused operations.

Each crash scenario runs in a subprocess because the `intentional_gpu_crash`
op poisons the CUDA context for the remainder of the process.

Run with:
    bt-b200 //max/tests/integration/unorganized:test_fusion_error_messages
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fusion_error_scenarios import SCENARIO_NAMES
from max.driver import accelerator_count


def _run_scenario_in_subprocess(scenario_name: str) -> str:
    """Run a crash scenario in a subprocess and return the error message.

    Each scenario poisons the CUDA context, so it must run in isolation.
    """
    env = os.environ.copy()
    env["_FUSION_ERROR_SCENARIO"] = scenario_name

    result = subprocess.run(
        [sys.executable, __file__],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    return result.stdout.strip()


# Cache for pre-warmed scenario results. All crash scenarios are launched
# in parallel the first time any one is requested, so the compilation
# cost is paid once concurrently instead of sequentially.
_scenario_cache: dict[str, str] = {}


def _warm_all_scenarios() -> None:
    """Launch all crash scenarios in parallel and cache results."""
    if _scenario_cache:
        return
    with ThreadPoolExecutor(max_workers=len(SCENARIO_NAMES)) as pool:
        futures = {
            name: pool.submit(_run_scenario_in_subprocess, name)
            for name in SCENARIO_NAMES
        }
        for name, fut in futures.items():
            _scenario_cache[name] = fut.result()


def _get_scenario_result(scenario_name: str) -> str:
    """Get a scenario result, warming the cache if needed."""
    _warm_all_scenarios()
    return _scenario_cache[scenario_name]


# ---------------------------------------------------------------------------
# Pytest tests — each runs a scenario in a subprocess
# ---------------------------------------------------------------------------


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize(
    "scenario_name,description",
    [
        ("single_crash", "Single op crash shows source trace"),
        ("elementwise_chain_crash", "Fused elementwise chain + crash"),
        ("matmul_epilogue_crash", "Matmul + epilogue fusion + crash"),
        ("many_fused_crash", "Many fused ops (8+) + crash"),
        ("big_matmul_fusion", "Matmul + 6 epilogues + crash"),
        ("two_matmuls_crash", "Two sequential matmuls + crash"),
    ],
)
def test_crash_scenario(
    scenario_name: str,
    description: str,
    kernel_verification_ops_path: Path,
) -> None:
    output = _get_scenario_result(scenario_name)
    assert output, f"Subprocess produced no output for {scenario_name}"

    data = json.loads(output)
    error_msg = data.get("error", "")

    print(f"\n{'=' * 72}")
    print(f"  {description}")
    print(f"{'=' * 72}")
    print(error_msg)
    print(f"{'=' * 72}")

    assert error_msg, f"No error captured for {scenario_name}"
    assert (
        "Source Traceback:" in error_msg
        or "source traceback" in error_msg.lower()
    ), (
        f"Expected source traceback in error message for {scenario_name}.\n"
        f"Got: {error_msg}"
    )

    # Multi-op scenarios should include the fusion tree and type signatures.
    if scenario_name != "single_crash":
        assert "Fusion info:" in error_msg, (
            f"Expected 'Fusion info:' in multi-op error for {scenario_name}.\n"
            f"Got: {error_msg}"
        )
        assert "->" in error_msg, (
            f"Expected type signatures (with '->') in error for {scenario_name}.\n"
            f"Got: {error_msg}"
        )


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_sqrt_negative_fused_ops() -> None:
    """sqrt(negative) in a fused kernel — may trigger assertion with MODULAR_MAX_DEBUG_ASSERT_LEVEL=all."""
    output = _get_scenario_result("sqrt_negative")
    assert output, "Subprocess produced no output"

    data = json.loads(output)
    if "no_error" in data:
        if data.get("has_nan"):
            pytest.skip(
                "sqrt(negative) produced NaN, not assertion (set MODULAR_MAX_DEBUG_ASSERT_LEVEL=all)"
            )
        else:
            pytest.skip("No assertion triggered")

    error_msg = data.get("error", "")
    print(f"\n{'=' * 72}")
    print("  sqrt(negative) in fused kernel — assertion error")
    print(f"{'=' * 72}")
    print(error_msg)
    print(f"{'=' * 72}")

    assert error_msg, "Expected an error message for sqrt(negative) scenario"
    assert "Fusion info:" in error_msg, (
        f"Expected 'Fusion info:' in sqrt(negative) error.\nGot: {error_msg}"
    )


# ---------------------------------------------------------------------------
# Subprocess entry point — runs a single scenario when invoked directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenario = os.environ.get("_FUSION_ERROR_SCENARIO")
    if scenario:
        import numpy as np
        from fusion_error_scenarios import build_scenario
        from max.driver import Accelerator, Buffer
        from max.engine import InferenceSession

        kernel_ops = Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])
        graph, numpy_arrays = build_scenario(scenario, kernel_ops)

        session = InferenceSession(devices=[Accelerator()])
        model = session.load(graph)
        bufs = [Buffer.from_numpy(a).to(Accelerator()) for a in numpy_arrays]
        try:
            result = model(*bufs)
            # For scenarios that might not crash (sqrt_negative)
            arr = result[0].to_numpy()
            print(
                json.dumps(
                    {
                        "no_error": True,
                        "has_nan": bool(np.any(np.isnan(arr))),
                    }
                )
            )
        except Exception as e:
            print(json.dumps({"error": str(e)}))
