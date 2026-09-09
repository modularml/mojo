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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test __is_run_in_comptime_interpreter bare keyword.
# Verifies that it emits kgen.is_run_in_comptime_interpreter : scalar<bool>
# directly.
# It is a runtime value and is intended for use in runtime 'if', not
# 'comptime if' (which requires a compile-time-parametric condition).


# CHECK-LABEL: lit.fn @"test_basic
def test_basic():
    # CHECK: kgen.is_run_in_comptime_interpreter : !kgen.scalar<bool>
    var x = __is_run_in_comptime_interpreter
    _ = x


# CHECK-LABEL: lit.fn @"test_in_runtime_if
def test_in_runtime_if():
    # CHECK: kgen.is_run_in_comptime_interpreter : !kgen.scalar<bool>
    if __is_run_in_comptime_interpreter:
        pass
