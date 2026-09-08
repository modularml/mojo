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

# Test the --mlir-timing option for `mojo run`. The compiler writes the report
# to stderr once the compilation is over, which is before the program runs.
# The report therefore covers the compilation alone, and a program that calls
# `exit()` cannot suppress it.

# The compiler runs the pass pipeline as a cached transform. Thus a warm cache
# gives a report that shows only the parse, with no passes. Each timing test
# below uses an empty MODULAR_CACHE_DIR. This keeps the report the same for
# each run, and lets the checks below find the passes.

# RUN: rm -rf %t.tree && env MODULAR_CACHE_DIR=%t.tree %mojo --mlir-timing %s 2>&1 | FileCheck %s --check-prefix=CHECK_TIMING
# RUN: rm -rf %t.list && env MODULAR_CACHE_DIR=%t.list %mojo --mlir-timing --mlir-timing-display=list %s 2>&1 | FileCheck %s --check-prefix=CHECK_LIST

# The timing stays off if the option is not present. The timing does not
# change the output of the program.
# RUN: rm -rf %t.off && env MODULAR_CACHE_DIR=%t.off %mojo %s 2>&1 | FileCheck %s --check-prefix=CHECK_OFF

# CHECK_TIMING: MLIR pass timing (--mlir-timing)
# CHECK_TIMING: Execution time report
# CHECK_TIMING-DAG: Import Mojo
# CHECK_TIMING-DAG: LowerLIT
# CHECK_TIMING-DAG: LowerKGENToLLVM
# CHECK_TIMING: hello from the program
# The report appears once, not again when the command ends.
# CHECK_TIMING-NOT: Execution time report

# CHECK_LIST: MLIR pass timing (--mlir-timing)
# CHECK_LIST: Execution time report
# CHECK_LIST: root
# CHECK_LIST-DAG: Import Mojo
# CHECK_LIST-DAG: LowerLIT
# CHECK_LIST-DAG: LowerKGENToLLVM
# CHECK_LIST: hello from the program

# CHECK_OFF: hello from the program
# CHECK_OFF-NOT: Execution time report
# CHECK_OFF-NOT: pass timing


def main():
    print("hello from the program")
