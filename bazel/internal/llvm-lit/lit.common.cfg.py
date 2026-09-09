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

import os
import platform
import shutil
import sys
from pathlib import Path
from sysconfig import get_config_var as var

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst


if "MODULAR_RUNNING_TESTS" not in os.environ:
    raise SystemExit(
        "\033[31merror\033[0m: use 'bazel test' instead of 'bazel run' to run tests"
    )

# Choose between lit's internal shell pipeline runner and a real shell.  If
# LIT_USE_INTERNAL_SHELL is in the environment, we use that as an override.
use_lit_shell = os.environ.get("LIT_USE_INTERNAL_SHELL")
if use_lit_shell:
    # 0 is external, "" is default, and everything else is internal.
    execute_external = use_lit_shell == "0"
else:
    # Otherwise we default to internal on Windows and external elsewhere, as
    # bash on Windows is usually very slow.
    execute_external = sys.platform not in ["win32"]

import modular_test_format

config.test_format = modular_test_format.ModularShTest(
    execute_external, force_execute_external=execute_external
)

if execute_external:
    config.available_features.add("shell")
# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
]

llvm_config.add_tool_substitutions([ToolSubst("python3", sys.executable)])

llvm_config.with_environment(
    "MOJO_PYTHON",
    sys.executable,
)
libpython = str(
    Path(sys.executable).resolve().parent.parent / "lib" / var("INSTSONAME")
)

llvm_config.with_environment(
    "MOJO_PYTHON_LIBRARY",
    libpython,
)

# Ensure the tests find the packages from the venv.
llvm_config.with_environment(
    "PATH",
    str(Path(sys.executable).parent),
    append_path=True,
)

if sys.platform == "darwin":
    # Expose macOS 26+ as a lit feature so tests can be selectively skipped.
    # The ASAN runtime (libclang_rt.asan_osx_dynamic.dylib) comes from the
    # pre-built @clang-macos toolchain (LLVM 20.1.8), which hangs on macOS 26
    # in FindAvailableMemoryRange (fixed in llvm/llvm-project#191039,
    # merged 2026-04-09 to LLVM main, not yet in any release). Remove once
    # @clang-macos is updated to a release that includes the fix. MOTO-1516.
    _mac_ver = platform.mac_ver()[0]
    if _mac_ver and int(_mac_ver.split(".")[0]) >= 26:
        config.available_features.add("macos-26+")

    # Metal AIR disassembly tests require Xcode's air-objdump utility.
    if shutil.which("air-objdump") or (
        shutil.which("xcrun")
        and os.system(
            "xcrun -sdk macosx air-objdump --version >/dev/null 2>&1"
        )
        == 0
    ):
        config.available_features.add("air-objdump")

#---------------------------------------
# Mojo tools
#---------------------------------------

mojo_exe = "mojo"

def mojo_subst(name, args):
    return ToolSubst(name, mojo_exe, extra_args=args)

sanitize_args = []
if config.llvm_use_sanitizer and config.llvm_use_sanitizer != "undefined":
    sanitize_args = ["--sanitize", config.llvm_use_sanitizer.lower()]

# The rest of the mojo commands just inherit the assert options. In this case
# run with assertions enabled unless one explicitly sets
# MOJO_ENABLE_ASSERTIONS_IN_TESTS=0 environment variable.
assert_args = ["-D", "ASSERT=all"] if bool(int(os.environ.get("MOJO_ENABLE_ASSERTIONS_IN_TESTS", 1))) else []

asan_args = ["--sanitize", "address"] if "--sanitize" not in set(sanitize_args) else []
debug_full_args = ["--debug-level", "full"]

llvm_config.add_tool_substitutions([
    mojo_subst("%mojo-no-debug-no-assert",       [         "-Werror"] + sanitize_args                                            ),
    mojo_subst("%mojo-build-no-debug-no-assert", ["build", "-Werror"] + sanitize_args                                            ),
    mojo_subst("%mojo-no-debug",                 [         "-Werror"] + sanitize_args + assert_args                              ),
    mojo_subst("%mojo-build-no-debug",           ["build", "-Werror"] + sanitize_args + assert_args                              ),
    mojo_subst("%mojo",                          [         "-Werror"] + sanitize_args + assert_args + debug_full_args            ),
    mojo_subst("%mojo-build",                    ["build", "-Werror"] + sanitize_args + assert_args + debug_full_args            ),
    mojo_subst("%mojo-build-no-werror",          ["build"           ] + sanitize_args + assert_args + debug_full_args            ),
    mojo_subst("%mojo-build-asan",               ["build", "-Werror"] + sanitize_args + assert_args + debug_full_args + asan_args),
])

#---------------------------------------

# For %mpirun-gpu-per-process, we dynamically detect the number of GPUs at runtime
mpirun_gpu_per_process = (
    "N=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l);"
    "mpirun --allow-run-as-root --bind-to none -n $N"
)

config.substitutions.append(("%mpirun-gpu-per-process", mpirun_gpu_per_process))
config.substitutions.append(("%bare-mojo", mojo_exe))
config.substitutions.append(("%{mojo_version_major}", str(config.mojo_version_major)))
config.substitutions.append(("%{mojo_version_minor}", str(config.mojo_version_minor)))
config.substitutions.append(("%{mojo_version_patch}", str(config.mojo_version_patch)))
