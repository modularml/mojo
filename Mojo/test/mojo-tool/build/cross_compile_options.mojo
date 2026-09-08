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

# Test cross-compilation target options and validation of option family mixing.
#
# There are two option families that should not be mixed:
# - LLVM-style: --target-cpu, --target-features
# - GCC/Clang-style: --march, --mcpu, --mtune

# Cross-compile to x86_64 Linux using --mcpu (should derive target CPU from mcpu)
# RUN: %mojo-build --target-triple x86_64-unknown-linux-gnu --mcpu=haswell --emit=llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_X86

# Cross-compile to AArch64 Linux using --mcpu
# RUN: %mojo-build --target-triple aarch64-unknown-linux-gnu --mcpu=cortex-a72 --emit=llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_AARCH64

# When both --march and --mcpu are specified, --march takes precedence for the arch name
# RUN: %mojo-build --target-triple x86_64-unknown-linux-gnu --march=x86-64 --mcpu=haswell --emit=llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_MARCH

# Error when --target-cpu is used with --mcpu (mixing option families)
# RUN: not %mojo-build --target-cpu=haswell --mcpu=skylake %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_CPU_MCPU

# Error when --target-cpu is used with --march (mixing option families)
# RUN: not %mojo-build --target-cpu=haswell --march=x86-64 %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_CPU_MARCH

# Error when --target-features is used with --mcpu (mixing option families)
# RUN: not %mojo-build --mcpu=haswell --target-features="+avx512f" %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_FEATURES_MCPU

# Error when --target-features is used with --march (mixing option families)
# RUN: not %mojo-build --march=x86-64 --target-features="+avx2" %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_FEATURES_MARCH

# CHECK_ERROR_CPU_MCPU: error: --target-cpu cannot be used with --march or --mcpu
# CHECK_ERROR_CPU_MARCH: error: --target-cpu cannot be used with --march or --mcpu
# CHECK_ERROR_FEATURES_MCPU: error: --target-features cannot be used with --march or --mcpu
# CHECK_ERROR_FEATURES_MARCH: error: --target-features cannot be used with --march or --mcpu

# CHECK_X86: target triple = "x86_64-unknown-linux-gnu"
# CHECK_X86: "target-cpu"="haswell"

# CHECK_AARCH64: target triple = "aarch64-unknown-linux-gnu"
# CHECK_AARCH64: "target-cpu"="cortex-a72"

# CHECK_MARCH: target triple = "x86_64-unknown-linux-gnu"
# CHECK_MARCH: "target-cpu"="x86-64"

# Test that compile-time CPU feature queries reflect what LLVM will actually
# compile for. znver4 enables avx512f by default; an explicit -avx512f in
# --target-features (simulating an OS that withholds AVX-512 via XCR0) must
# make has_avx512f() return false, while omitting -avx512f must return true.
#
# RUN: %mojo-build --target-triple x86_64-unknown-linux-gnu --target-cpu znver4 \
# RUN:   --target-features "+avx2,-avx512f" --emit=llvm -o - %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK_AVX512F_OFF
#
# RUN: %mojo-build --target-triple x86_64-unknown-linux-gnu --target-cpu znver4 \
# RUN:   --target-features "+avx2" --emit=llvm -o - %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK_AVX512F_ON

# CHECK_AVX512F_OFF-LABEL: @has_avx512f
# CHECK_AVX512F_OFF-NEXT:  ret i1 false

# CHECK_AVX512F_ON-LABEL: @has_avx512f
# CHECK_AVX512F_ON-NEXT:  ret i1 true


from std.sys.info import CompilationTarget


@export
def has_avx512f() abi("C") -> Bool:
    return CompilationTarget.has_avx512f()


def main():
    pass
