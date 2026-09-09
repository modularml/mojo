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

# Validate that we correctly cross compile with a small set of arguments.

# TODO: It should be valid to only pass the --target-triple and get sensible default CPU and features.

# RUN: mojo build --target-triple arm64-apple-macosx11.0 --target-cpu=apple-m1 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_MACOS
# RUN: mojo build --target-triple x86_64-unknown-linux-gnu --target-cpu=x86-64-v3 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_LINUX_X86_64
# RUN: mojo build --target-triple aarch64-unknown-linux-gnu --target-cpu=neoverse-v1 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_LINUX_AARCH64

# Bare-metal RISC-V. Emission requires a registered `TargetBackend`, so these
# also guard against RISC-V silently losing its backend while staying listed in
# `--print-supported-targets`.
# RUN: mojo build --target-triple riscv32-unknown-none-elf --target-cpu=generic-rv32 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_RV32
# RUN: mojo build --target-triple riscv64-unknown-none-elf --target-cpu=generic-rv64 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_RV64
# RUN: mojo build --target-triple riscv32-unknown-none-elf --target-cpu=sifive-e31 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_RV32_CPU
# RUN: mojo build --target-triple riscv32-unknown-none-elf --march=rv32i_zba_zbb --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_RV32_MARCH

# CHECK_MACOS: target triple = "arm64-apple-macosx11.0"
# CHECK_MACOS: "target-cpu"="apple-m1"
# CHECK_MACOS: "target-features"="+aes,+altnzcv,+ccdp,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs"

# CHECK_LINUX_X86_64: target triple = "x86_64-unknown-linux-gnu"
# CHECK_LINUX_X86_64: "target-cpu"="x86-64-v3"
# CHECK_LINUX_X86_64: "target-features"="+avx,+avx2,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"

# CHECK_LINUX_AARCH64: target triple = "aarch64-unknown-linux-gnu"
# CHECK_LINUX_AARCH64: "target-cpu"="neoverse-v1"
# CHECK_LINUX_AARCH64: "target-features"="+aes,+bf16,+ccdp,+ccidx,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+rand,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+spe,+ssbs,+sve"

# CHECK_RV32: target triple = "riscv32-unknown-none-elf"
# CHECK_RV32: "target-cpu"="generic-rv32"
# CHECK_RV32: "target-features"="+32bit,+i"

# CHECK_RV64: target triple = "riscv64-unknown-none-elf"
# CHECK_RV64: "target-cpu"="generic-rv64"
# CHECK_RV64: "target-features"="+64bit,+i"

# A RISC-V CPU model implies extensions that clang only derives from `-march`,
# and an `-march` ISA string has to survive as more than the base ISA. Both
# feed `CompilationTarget.has_riscv_extension()`, so both are pinned here.
# CHECK_RV32_CPU: "target-cpu"="sifive-e31"
# CHECK_RV32_CPU: "target-features"="+32bit,+a,+c,+i,+m,+zaamo,+zalrsc,+zca,+zicsr,+zifencei,+zmmul"

# CHECK_RV32_MARCH: "target-features"="+32bit,+i,+zba,+zbb"


def main():
    pass
