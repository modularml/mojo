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
# NOTE: A parallel copy of this file lives at
#   KGEN/test/mojo-debug/Inputs/unsafe_pointer.mojo
# The two files must be kept in sync. Any change here (adding variables,
# reordering lines) requires updating the sibling file AND updating the
# hardcoded line numbers in StdlibTypesTest.cpp / unsafe-pointer-formatter.lldb.


def keep_alive[*Ts: AnyType](*args: *Ts):
    pass


def main():
    var p_int = alloc[Int]({count = 1}).unsafe_leak()
    p_int[0] = 42
    keep_alive(p_int)  # breakpoint

    var p_neg = alloc[Int]({count = 1}).unsafe_leak()
    p_neg[0] = -5
    keep_alive(p_neg)  # breakpoint

    var p_bool = alloc[Bool]({count = 1}).unsafe_leak()
    p_bool[0] = True
    keep_alive(p_bool)  # breakpoint

    var p_float = alloc[Float64]({count = 1}).unsafe_leak()
    p_float[0] = Float64(3.125)
    keep_alive(p_float)  # breakpoint
