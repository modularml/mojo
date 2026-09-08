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

# Test the --llvm-timing option for `mojo run`. The compiler writes the LLVM
# report when the compilation is complete. This is before the program starts,
# as it is for the MLIR report.

# LLVM gets a module only if the compilation cache does not have the object
# code. Therefore each run below uses an empty MODULAR_CACHE_DIR. Then the
# passes that the checks look for do run.

# RUN: rm -rf %t.on && env MODULAR_CACHE_DIR=%t.on %mojo --llvm-timing %s 2>&1 | FileCheck %s --check-prefix=CHECK_ON

# The timing stays off if the command does not have the option. The timing
# does not change the output of the program.
# RUN: rm -rf %t.off && env MODULAR_CACHE_DIR=%t.off %mojo %s 2>&1 | FileCheck %s --check-prefix=CHECK_OFF

# The compiler prints the report one time. The two CHECK_ON-NOT lines cover
# the two places where a second report can appear: before the program starts,
# and after the process stops.
# CHECK_ON: LLVM pass timing (--llvm-timing)
# CHECK_ON: Pass execution timing report
# CHECK_ON-NOT: Pass execution timing report
# CHECK_ON: hello from the program
# CHECK_ON-NOT: Pass execution timing report

# CHECK_OFF: hello from the program
# CHECK_OFF-NOT: timing report
# CHECK_OFF-NOT: pass timing


def main():
    print("hello from the program")
