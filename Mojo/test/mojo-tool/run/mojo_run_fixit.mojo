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

# RUN: mojo run --experimental-fixit %s | FileCheck %s --check-prefix=AUTO-FIXIT
# RUN: mojo run --experimental-fixit %s | FileCheck %s --check-prefix=NO-FIXIT

# AUTO-FIXIT: Fixits applied.
# NO-FIXIT: No fixits to apply.

# After applying the fixits, the build should succeed.
# RUN: mojo run %s

# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cat %s | grep -v "# CHECK" | FileCheck %s


# CHECK-LABEL: def old_origin_of
def old_origin_of[T: AnyType](a: T):
    # CHECK-NEXT: _ = origin_of(a)
    # CHECK-NOT: _ = __origin_of(a)
    _ = __origin_of(a)


# CHECK-LABEL: def old_origin_of_2
def old_origin_of_2[T: AnyType](b: T):
    # CHECK-NEXT: _ = origin_of(b)
    # CHECK-NOT: _ = __origin_of(b)
    _ = __origin_of(b)


@fieldwise_init
struct Bar:
    @implicit(deprecated=True)
    def __init__(out self, i: Int):
        pass


def take_bar(b: Bar) -> Int:
    return 1


# CHECK-LABEL: def main() raises:
def main() raises:
    old_origin_of(1)
    old_origin_of_2(2)

    # CHECK: comptime b = Bar()
    # CHECK-NEXT: var a = materialize[b]()
    comptime b = Bar()
    var a = b

    # Note: there are two nested fixits we're testing here.
    # CHECK: var _: Bar = Bar(take_bar(Bar(10 + 5)))
    var _: Bar = take_bar(10 + 5)
