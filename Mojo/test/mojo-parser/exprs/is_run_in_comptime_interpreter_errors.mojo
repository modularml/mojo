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
# RUN: %parse-mojo-isolated %s --verify-diagnostics

# ===----------------------------------------------------------------------=== #
# __is_run_in_comptime_interpreter
# ===----------------------------------------------------------------------=== #


def test_with_comptime_if():
    #expected-error@+1 {{cannot use a dynamic value in 'comptime if' condition}}
    comptime if __is_run_in_comptime_interpreter:
        var x: Int

def test_as_comptime_expression[b: Bool]():
    #expected-error@+1 {{cannot use a dynamic value in comptime initializer}}
    comptime i = __is_run_in_comptime_interpreter and b:
