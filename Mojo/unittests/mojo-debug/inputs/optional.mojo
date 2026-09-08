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

from std.collections import Optional


def keep_alive[*Ts: AnyType](*args: *Ts):
    pass


def main():
    var opt_int = Optional(42)

    print("breakpoint1")  # breakpoint

    var opt_none = Optional[Int](None)

    print("breakpoint2")  # breakpoint

    var opt_str = Optional(String("hello, world"))

    print("breakpoint3")  # breakpoint

    # Exercise the small-string (inline) path.
    var opt_str_small = Optional(String("hi"))

    print("breakpoint4")  # breakpoint

    var opt_str_none = Optional[String](None)

    print("breakpoint5")  # breakpoint

    # Exercise the GetSummaryAsCString path in renderActivePayload (a type that
    # has its own registered formatter rather than a raw scalar or String).
    var opt_bool = Optional(True)

    print("breakpoint6")  # breakpoint

    keep_alive(
        opt_int, opt_none, opt_str, opt_str_small, opt_str_none, opt_bool
    )
