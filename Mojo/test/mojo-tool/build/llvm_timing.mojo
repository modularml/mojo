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

# Test the --llvm-timing option for `mojo build`. The compiler writes the
# LLVM report to stderr when the compilation is complete.

# LLVM gets a module only if the compilation cache does not have the object
# code. Therefore each run below uses an empty MODULAR_CACHE_DIR. Then the
# passes that the checks look for do run.

# RUN: rm -rf %t.on && env MODULAR_CACHE_DIR=%t.on %mojo-build --llvm-timing %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_ON

# The compiler writes the report to stderr. Therefore the report does not go
# into the output of the tool.
# RUN: rm -rf %t.err && env MODULAR_CACHE_DIR=%t.err %mojo-build --llvm-timing %s -o %t 2>/dev/null | FileCheck %s --check-prefix=CHECK_OFF --allow-empty

# The timing stays off if the command does not have the option.
# RUN: rm -rf %t.off && env MODULAR_CACHE_DIR=%t.off %mojo-build %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_OFF --allow-empty

# The two timing options are independent. The MLIR report comes first.
# RUN: rm -rf %t.both && env MODULAR_CACHE_DIR=%t.both %mojo-build --mlir-timing --llvm-timing %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK_BOTH

# CHECK_ON: LLVM pass timing (--llvm-timing)
# CHECK_ON: Pass execution timing report
# The compiler prints the report one time. The process does not print the
# report again when it stops.
# CHECK_ON-NOT: Pass execution timing report

# CHECK_OFF-NOT: timing report
# CHECK_OFF-NOT: pass timing

# A title names the tool that made each report.
# CHECK_BOTH: MLIR pass timing (--mlir-timing)
# CHECK_BOTH: Execution time report
# CHECK_BOTH: LLVM pass timing (--llvm-timing)
# CHECK_BOTH: Pass execution timing report


def main():
    pass
