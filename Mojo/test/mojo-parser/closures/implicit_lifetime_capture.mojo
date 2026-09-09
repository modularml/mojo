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


struct Thing[a: Origin[mut=True]](TrivialRegisterPassable):
    pass


struct Foo(Movable where False):
    pass


def use(y: Thing):
    pass


# Check that the implicit lifetime of `x` is properly captured when
# referenced through a parameter of `y`.


# CHECK-LABEL: lit.fn @"capture_implicit_origin
def capture_implicit_origin(var x: Foo, y: Thing[origin_of(x)]):
    # COM: The closure captures `y`, whose `Thing` type carries `x`'s implicit
    # COM: origin, so the closure storage struct is parametrized by `x`'s origin
    # COM: and its initializer receives `y`.
    # CHECK: lit.call {{.*}}capture_it::__storage"::@"__init__
    # CHECK-SAME: <:origin<true> *"x
    # CHECK-SAME: "y":
    def capture_it() {imm y}:
        use(y)
