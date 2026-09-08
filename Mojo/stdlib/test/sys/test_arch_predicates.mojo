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

# RUN: %bare-mojo build --target-triple=aarch64-unknown-linux-gnu --target-cpu=neoverse-n1 -D EXPECT_ARM --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=arm64-unknown-linux-gnu --target-cpu=neoverse-n1 -D EXPECT_ARM --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=x86_64-unknown-linux-gnu --target-cpu=x86-64 -D EXPECT_X86 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=x86_64-unknown-linux-gnu --target-cpu=znver3 -D EXPECT_X86 -D EXPECT_SSE4 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=i686-unknown-linux-gnu --target-cpu=i686 -D EXPECT_X86 --emit=llvm %s -o /dev/null

# RISC-V. `generic-rv32`/`generic-rv64` are bare `I`, so they pin down that the
# extension query reads the CPU rather than reporting the architecture twice;
# `sifive-e31` (RV32IMAC) and `sifive-u54` (RV64IMAFDC) are the populated
# counterparts. `riscv32-unknown-unknown` is the freestanding triple firmware
# builds use.
# RUN: %bare-mojo build --target-triple=riscv32-unknown-none-elf --target-cpu=generic-rv32 -D EXPECT_RISCV -D EXPECT_RV32 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=riscv32-unknown-unknown --target-cpu=generic-rv32 -D EXPECT_RISCV -D EXPECT_RV32 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=riscv32-unknown-none-elf --target-cpu=sifive-e31 -D EXPECT_RISCV -D EXPECT_RV32 -D EXPECT_RISCV_M -D EXPECT_RISCV_C --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=riscv64-unknown-none-elf --target-cpu=generic-rv64 -D EXPECT_RISCV -D EXPECT_RV64 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=riscv64-unknown-none-elf --target-cpu=sifive-u54 -D EXPECT_RISCV -D EXPECT_RV64 -D EXPECT_RISCV_M -D EXPECT_RISCV_C -D EXPECT_RISCV_D --emit=llvm %s -o /dev/null

# `is_arm()`, `is_x86()`, and `is_riscv()` describe the architecture, so they
# must not vary with the target CPU. The baseline `x86-64` CPU is the
# interesting case: it has no SSE4.1, which `is_x86()` used to be an alias for.
# The `arm64` triple covers a spelling that only matches after canonicalization.

from std.sys import is_defined
from std.sys.info import CompilationTarget


def main():
    comptime expect_arm = is_defined["EXPECT_ARM"]()
    comptime expect_x86 = is_defined["EXPECT_X86"]()
    comptime expect_sse4 = is_defined["EXPECT_SSE4"]()
    comptime expect_riscv = is_defined["EXPECT_RISCV"]()
    comptime expect_rv32 = is_defined["EXPECT_RV32"]()
    comptime expect_rv64 = is_defined["EXPECT_RV64"]()
    comptime expect_riscv_m = is_defined["EXPECT_RISCV_M"]()
    comptime expect_riscv_c = is_defined["EXPECT_RISCV_C"]()
    comptime expect_riscv_d = is_defined["EXPECT_RISCV_D"]()

    comptime assert (
        CompilationTarget.is_arm() == expect_arm
    ), "is_arm() disagrees with the target triple"
    comptime assert (
        CompilationTarget.is_x86() == expect_x86
    ), "is_x86() disagrees with the target triple"
    comptime assert (
        CompilationTarget.has_sse4() == expect_sse4
    ), "has_sse4() must track the target CPU, not the architecture"

    comptime assert (
        CompilationTarget.is_riscv() == expect_riscv
    ), "is_riscv() disagrees with the target triple"
    comptime assert (
        CompilationTarget.is_rv32() == expect_rv32
    ), "is_rv32() disagrees with the target triple"
    comptime assert (
        CompilationTarget.is_rv64() == expect_rv64
    ), "is_rv64() disagrees with the target triple"

    # `I` is the base ISA, so every RISC-V target has it and nothing else does.
    comptime assert (
        CompilationTarget.has_riscv_extension["i"]() == expect_riscv
    ), "has_riscv_extension() must report the base integer ISA on RISC-V"
    comptime assert (
        CompilationTarget.has_riscv_extension["m"]() == expect_riscv_m
    ), "has_riscv_extension() must track the target CPU"
    comptime assert (
        CompilationTarget.has_riscv_extension["c"]() == expect_riscv_c
    ), "has_riscv_extension() must track the target CPU"
    comptime assert (
        CompilationTarget.has_riscv_extension["d"]() == expect_riscv_d
    ), "has_riscv_extension() must track the target CPU"

    # Implied extensions count: `M` implies `Zmmul`, and `D` implies `F`.
    comptime assert (
        CompilationTarget.has_riscv_extension["zmmul"]() == expect_riscv_m
    ), "has_riscv_extension() must see extensions implied by another"
    comptime assert (
        CompilationTarget.has_riscv_extension["f"]() == expect_riscv_d
    ), "has_riscv_extension() must see extensions implied by another"
