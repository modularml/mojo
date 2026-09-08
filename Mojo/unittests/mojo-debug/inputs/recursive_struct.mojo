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

from debug_test_utils import keep_alive


@fieldwise_init
struct Foo:
    var x: Int

    @__allow_legacy_any_origin_fields
    var ptr: Pointer[Foo, MutAnyOrigin]


struct Bar:
    var x: Int

    @__allow_legacy_any_origin_fields
    var ptr: Pointer[Bar, MutAnyOrigin]

    def __init__(out self, x: Int, ptr: Pointer[Bar, MutAnyOrigin]):
        self.x = x
        self.ptr = ptr


def main():
    var f1: Foo = Foo(7, Pointer[Foo, MutAnyOrigin].unsafe_dangling())
    var f2: Foo = Foo(8, Pointer(to=f1).as_unsafe_any_origin())
    print(f2.ptr[].x)

    var b1: Bar = Bar(22, Pointer[Bar, MutAnyOrigin].unsafe_dangling())
    var b2: Bar = Bar(23, Pointer(to=b1).as_unsafe_any_origin())
    print(b2.ptr[].x)  # breakpoint

    keep_alive(f1, f2, b1, b2)
