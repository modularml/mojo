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

# Test that extending debug lifetimes of trivial types does not
# transitively keep non-trivial values alive.  A trivial Int derived
# from a non-trivial String (via len()) must not prevent the String
# from being ASAP-destroyed.
#
# Note: origin-carrying types like UnsafePointer and Span are
# TrivialRegisterPassable and not tracked by CheckLifetimes at all,
# so the debug lifetime extension cannot affect them.


def use_int(x: Int):
    pass


def get_string() -> String:
    var s = "hel"
    s += "lo"
    return s


def main():
    # Create a non-trivial value and a trivial value derived from it.
    var my_string = get_string()
    var length = len(my_string.bytes())
    use_int(length)  # last use of length

    # my_string should still be alive because print uses it.
    print(my_string)  # breakpoint

    # After this point, my_string should be ASAP-destroyed (non-trivial).
    # The trivial `length` should NOT keep my_string alive.
    print("after string use")  # breakpoint
