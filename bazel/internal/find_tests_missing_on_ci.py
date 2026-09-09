#!/usr/bin/env python3
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

import shlex
import subprocess
import sys

_IGNORED_TARGETS = {
    "@@//max/kernels/benchmarks:nn/bench_gather_reduce.mojo.test",  # Disabled
    "@@//max/kernels/test/gpu/compile:test_wmma_nvptx.mojo.test",  # A100 only
    "@@//max/kernels/test/gpu/layout:test_tensor_core_mi300x.mojo.test",  # MI300x only
    "@@//max/kernels/test/gpu/layout:test_wgmma_e4m3_e5m2_layouts_f16.mojo.test",  # Disabled
    "@@//max/kernels/test/gpu/layout:test_wgmma_e4m3_e5m2_layouts_f32.mojo.test",  # Disabled
    "@@//max/kernels/test/gpu/linalg:_test_tma.mojo.test",  # Disabled
    "@@//max/kernels/test/gpu/linalg:test_matmul_selection_heuristic.mojo.test",  # A100 only
    "@@//max/kernels/test/gpu/linalg:test_tma_wgmma_with_multicast.mojo.test",  # Disabled
    "@@//max/tests/integration/kv_cache/transfer_engine:test_send_recv_concurrent_gpu",  # Disabled
}


def _should_ignore_target(label: str) -> bool:
    return label in _IGNORED_TARGETS


def _cquery_tests(config: str, tag: str | None) -> set[str]:
    starlark_query = '"" if "IncompatiblePlatformProvider" in providers(target) else target.label'
    command = [
        "bazel",
        "cquery",
        "--color=yes",
        "//...",
        f"--config={config}",
        "--output=starlark",
        f"--starlark:expr={starlark_query}",
    ]

    if tag:
        command.extend(
            [
                f"--build_tag_filters={tag}",
                f"--test_tag_filters={tag}",
            ]
        )

    print(shlex.join(command))
    result = subprocess.check_output(command).decode()
    targets = {x.strip() for x in result.splitlines() if x.strip()}

    if not targets:
        raise SystemExit(
            f"No tests found for config '{config}' with tag '{tag}'"
        )
    if len(targets) < 10:
        raise SystemExit(
            f"Only found {len(targets)} tests for config '{config}' with tag '{tag}'"
        )
    return targets


def _main() -> None:
    all_tests = {
        x.strip() if x.startswith("@") else f"@@{x.strip()}"
        for x in subprocess.check_output(
            ["bazel", "query", "tests(//...) - attr(tags, manual, //...)"]
        )
        .decode()
        .splitlines()
    }

    cpu_tests = _cquery_tests("remote-intel", None)
    macos_tests = _cquery_tests("remote-macos", None)
    b200_tests = _cquery_tests("remote-b200", "gpu")
    mi355_tests = _cquery_tests("remote-mi355", "gpu")

    missing_on_ci = all_tests - (
        cpu_tests | mi355_tests | macos_tests | b200_tests
    )
    missing_on_ci = {
        test for test in missing_on_ci if not _should_ignore_target(test)
    }
    if missing_on_ci:
        print("error: these tests do not run on known CI configurations:")
        for test in sorted(missing_on_ci):
            print(f"  {test}")
        sys.exit(1)
    else:
        print("All tests should be run on CI")


if __name__ == "__main__":
    _main()
