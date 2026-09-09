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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# RISC-V ISA strings are conventionally written uppercase (`RV32IMAC`), but the
# feature names are lowercase. Without this guard the mismatch is a silent
# `False`, which quietly picks the wrong code path forever.

from std.sys.info import CompilationTarget


def main():
    # CHECK: constraint failed: RISC-V extension names are lowercase, got: M
    var _has_m = CompilationTarget.has_riscv_extension["M"]()
