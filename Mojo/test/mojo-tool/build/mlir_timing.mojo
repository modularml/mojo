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

# Test the --mlir-timing option for `mojo build`.

# The compiler runs the pass pipeline as a cached transform. Thus a warm cache
# gives a report that shows only the parse, with no passes. Each timing test
# below uses an empty MODULAR_CACHE_DIR. This keeps the report the same for
# each run, and lets the checks below find the passes.

# RUN: rm -rf %t.tree && env MODULAR_CACHE_DIR=%t.tree %mojo-build --mlir-timing %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_TREE

# The list display collects the times of each timer under one `root` row. The
# tree display does not print this row.
# RUN: rm -rf %t.list && env MODULAR_CACHE_DIR=%t.list %mojo-build --mlir-timing --mlir-timing-display=list %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_LIST

# The timing stays off if the option is not present.
# RUN: rm -rf %t.off && env MODULAR_CACHE_DIR=%t.off %mojo-build %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_OFF --allow-empty

# The compiler examines the display mode, also when it does not use the mode.
# Thus the compiler reports an error for an incorrect mode.
# RUN: not %mojo-build --mlir-timing --mlir-timing-display=bogus %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_BAD_MODE
# RUN: not %mojo-build --mlir-timing-display=bogus %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_BAD_MODE

# A correct display mode without --mlir-timing is permitted, but has no effect.
# RUN: rm -rf %t.unused && env MODULAR_CACHE_DIR=%t.unused %mojo-build --mlir-timing-display=list %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_OFF --allow-empty

# These two passes are from the start and the end of the pipeline. Thus the
# report shows more than the parse.
# CHECK_TREE: MLIR pass timing (--mlir-timing)
# CHECK_TREE: Execution time report
# CHECK_TREE-DAG: Import Mojo
# CHECK_TREE-DAG: LowerLIT
# CHECK_TREE-DAG: LowerKGENToLLVM

# CHECK_LIST: MLIR pass timing (--mlir-timing)
# CHECK_LIST: Execution time report
# CHECK_LIST: root
# CHECK_LIST-DAG: Import Mojo
# CHECK_LIST-DAG: LowerLIT
# CHECK_LIST-DAG: LowerKGENToLLVM

# CHECK_OFF-NOT: Execution time report
# CHECK_OFF-NOT: pass timing

# CHECK_BAD_MODE: error: invalid mlir-timing-display 'bogus', expected one of: `tree` (the default value), or `list`


def main():
    pass
