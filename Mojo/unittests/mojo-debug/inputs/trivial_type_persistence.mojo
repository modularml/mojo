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

# Test whether trivial types (Int, Float, Bool) persist in the debugger
# past their last use point through the end of the scope.


def use_int(x: Int):
    pass


def use_float(x: Float64):
    pass


def main():
    # Define trivial-type variables and use them early.
    var my_int = 42
    var my_float: Float64 = 3.14
    var my_bool = True
    use_int(my_int)  # last use of my_int
    use_float(my_float)  # last use of my_float

    # This breakpoint is PAST the last use of all three trivial variables.
    # If trivial types persist through the scope, they should still be visible.
    print("after last use")  # breakpoint

    # Use a non-trivial type to contrast behavior.
    var my_string = String("hello")
    print(my_string)  # breakpoint

    # Final breakpoint: my_string's last use was above, and all trivial
    # variables are still in scope but long past their last use.
    print("end")  # breakpoint
