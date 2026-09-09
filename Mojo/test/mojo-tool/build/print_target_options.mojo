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

# Test --print-* target information options.

# Test --print-supported-targets lists registered targets
# RUN: %mojo-build --print-supported-targets 2>&1 | FileCheck %s --check-prefix=CHECK_TARGETS

# Test --print-effective-target shows host target by default, with no
# --target-abi line since it is unset
# RUN: %mojo-build --print-effective-target 2>&1 | FileCheck %s --check-prefixes=CHECK_EFFECTIVE,CHECK_NO_ABI

# Test --print-effective-target with cross-compilation options
# RUN: %mojo-build --print-effective-target --target-triple x86_64-unknown-linux-gnu --mcpu=haswell 2>&1 | FileCheck %s --check-prefix=CHECK_EFFECTIVE_CROSS

# Test --print-effective-target reflects --target-abi
# RUN: %mojo-build --print-effective-target --target-abi lp64d 2>&1 | FileCheck %s --check-prefix=CHECK_EFFECTIVE_ABI

# Test --print-supported-cpus requires --target-triple
# RUN: not %mojo-build --print-supported-cpus 2>&1 | FileCheck %s --check-prefix=CHECK_CPUS_NO_TARGET

# Test --print-supported-cpus lists CPUs for specific target
# RUN: %mojo-build --print-supported-cpus --target-triple x86_64-unknown-linux-gnu 2>&1 | FileCheck %s --check-prefix=CHECK_CPUS_X86

# Test error when multiple print options specified
# RUN: not %mojo-build --print-effective-target --print-supported-targets 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_MULTI

# Test error for unsupported target
# RUN: not %mojo-build --print-supported-cpus --target-triple invalid-unknown-unknown 2>&1 | FileCheck %s --check-prefix=CHECK_INVALID_TARGET

# Every listed target must have a registered code-generation backend, so each
# entry here is expected to survive `--emit` (see mojo_targets.mojo for the
# matching emission checks).
# CHECK_TARGETS: Registered Targets:
# CHECK_TARGETS-DAG: aarch64
# CHECK_TARGETS-DAG: riscv32
# CHECK_TARGETS-DAG: riscv64
# CHECK_TARGETS-DAG: x86-64

# CHECK_EFFECTIVE: Effective target configuration:
# CHECK_EFFECTIVE: --target-triple
# CHECK_EFFECTIVE: --target-cpu
# CHECK_EFFECTIVE: --target-features

# CHECK_EFFECTIVE_CROSS: Effective target configuration:
# CHECK_EFFECTIVE_CROSS: --target-triple x86_64-unknown-linux-gnu
# CHECK_EFFECTIVE_CROSS: --target-cpu haswell
# CHECK_EFFECTIVE_CROSS: --target-features +avx

# CHECK_EFFECTIVE_ABI: Effective target configuration:
# CHECK_EFFECTIVE_ABI: --target-abi lp64d

# CHECK_NO_ABI-NOT: --target-abi

# CHECK_CPUS_NO_TARGET: error: --print-supported-cpus requires --target-triple to be specified

# CHECK_CPUS_X86: Available CPUs for target x86_64-unknown-linux-gnu:
# CHECK_CPUS_X86: haswell
# CHECK_CPUS_X86: skylake

# CHECK_ERROR_MULTI: error: only one --print-* option can be specified at a time

# CHECK_INVALID_TARGET: error: unknown target triple 'invalid-unknown-unknown'
# CHECK_INVALID_TARGET: Use --print-supported-targets to see available architectures.


def main():
    pass
