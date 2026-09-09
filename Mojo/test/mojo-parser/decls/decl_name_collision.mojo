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


# MOCO-2327
struct Foo(TrivialRegisterPassable):
    # CHECK: lit.fn @"__init__(a:::SIMD[DType.int, 1])"
    def __init__(out self, *, a: Int):
        pass

    # CHECK: lit.fn @"__init__(b:::SIMD[DType.int, 1])"
    def __init__(out self, *, b: Int):
        pass


def main() raises:
    # CHECK: lit.alias.decl *"{{.*}}": !Foo = <apply(:!lit.generator<(*, "b": !Int) -> !Foo> @decl_name_collision::@Foo::@"__init__(b:::SIMD[DType.int, 1])", {:scalar<index> 42})>
    comptime _foo = Foo(b=42)
