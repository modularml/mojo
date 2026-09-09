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
"""Compare fusion error messages between main and this branch.

Builds GPU graphs with different patterns, triggers crashes via
intentional_gpu_crash, and prints the resulting error messages.  Run on
both `main` and the feature branch to compare output. This test is not
run on CI. It is intended to run manually to diff error output between
main and a branch that is changing error output so humans can understand
the differences.

Each crash scenario runs in a subprocess because a GPU trap poisons the
CUDA context for the rest of the process.  All subprocesses are launched
in parallel for speed.

Usage:
    bt-b200 //max/tests/integration/unorganized:compare_fusion_errors \
        --test_output=streamed --cache_test_results=no
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fusion_error_scenarios import (
    SCENARIO_DESCRIPTIONS,
    SCENARIO_NAMES,
)

SEPARATOR = "=" * 78

# Exclude sqrt_negative from compare output — it doesn't use the crash op.
_COMPARE_SCENARIOS = [n for n in SCENARIO_NAMES if n != "sqrt_negative"]


def _run_scenario_subprocess(scenario_name: str) -> str:
    env = os.environ.copy()
    env["_COMPARE_SCENARIO"] = scenario_name
    result = subprocess.run(
        [sys.executable, __file__],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if not result.stdout.strip() and result.stderr.strip():
        return json.dumps(
            {"error": f"[STDERR] {result.stderr.strip()[-2000:]}"}
        )
    return result.stdout.strip()


def main() -> None:
    from max.driver import accelerator_count

    if accelerator_count() == 0:
        print("ERROR: No GPU available.")
        sys.exit(1)

    kernel_ops = os.environ.get("MODULAR_KERNEL_VERIFICATION_OPS_PATH")
    if not kernel_ops:
        print("ERROR: MODULAR_KERNEL_VERIFICATION_OPS_PATH not set.")
        sys.exit(1)

    print(f"MODULAR_DEBUG: {os.environ.get('MODULAR_DEBUG', 'not set')}")
    print(
        "MODULAR_DEVICE_CONTEXT_SYNC_MODE:"
        f" {os.environ.get('MODULAR_DEVICE_CONTEXT_SYNC_MODE', 'not set')}"
    )
    print(
        "MODULAR_MAX_DEBUG_ASSERT_LEVEL:"
        f" {os.environ.get('MODULAR_MAX_DEBUG_ASSERT_LEVEL', 'not set')}"
    )

    # Launch all scenarios in parallel.
    with ThreadPoolExecutor(max_workers=len(_COMPARE_SCENARIOS)) as pool:
        futures = {
            name: pool.submit(_run_scenario_subprocess, name)
            for name in _COMPARE_SCENARIOS
        }
        results = {name: fut.result() for name, fut in futures.items()}

    for i, name in enumerate(_COMPARE_SCENARIOS, 1):
        description = f"{i}. {SCENARIO_DESCRIPTIONS[name]}"
        print(f"\n{SEPARATOR}")
        print(f"  {description}")
        print(SEPARATOR)

        output = results[name]
        if not output:
            print("  [EMPTY OUTPUT] Subprocess produced no output.")
            continue

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            print(f"  [RAW OUTPUT]\n{output}")
            continue

        if "error" in data:
            print(data["error"])
        elif "no_error" in data:
            print("  [NO ERROR] Executed successfully.")
        else:
            print(f"  [UNEXPECTED] {data}")

        print(SEPARATOR)

    print(f"\n{SEPARATOR}")
    print("  Done. Compare this output between main and the feature branch.")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Subprocess entry point — runs a single scenario when invoked directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenario = os.environ.get("_COMPARE_SCENARIO")
    if scenario:
        from fusion_error_scenarios import build_scenario
        from max.driver import Accelerator, Buffer
        from max.engine import InferenceSession

        kernel_ops = Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])
        graph, numpy_arrays = build_scenario(scenario, kernel_ops)

        session = InferenceSession(devices=[Accelerator()])
        model = session.load(graph)
        bufs = [Buffer.from_numpy(a).to(Accelerator()) for a in numpy_arrays]
        try:
            model(*bufs)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        main()
