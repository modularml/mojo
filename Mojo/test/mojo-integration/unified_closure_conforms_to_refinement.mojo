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
# RUN: %mojo %s | FileCheck %s

# COM: Regression test for MOCO-4240: `where conforms_to(...)` refinement and
# COM: in-body `comptime assert conforms_to(...)` on weakly-bounded unified
# COM: closure type parameters must accept closures whose wrapper conforms to
# COM: the marker traits (ImplicitlyCopyable, RegisterPassable), matching what
# COM: a direct strong bound accepts.


def needs_strong[
    FuncType: ImplicitlyCopyable & RegisterPassable & def() -> Int,
](func: FuncType) -> Int:
    return func()


def where_refines[
    FuncType: def() -> Int,
](func: FuncType) -> Int where conforms_to(
    FuncType, ImplicitlyCopyable & RegisterPassable
):
    return needs_strong(func)


def assert_refines[FuncType: def() -> Int](func: FuncType) -> Int:
    comptime assert conforms_to(FuncType, ImplicitlyCopyable)
    comptime assert conforms_to(FuncType, RegisterPassable)
    return needs_strong(func)


def copy_via_refinement[FuncType: def() -> Int](func: FuncType) -> Int:
    comptime assert conforms_to(FuncType, ImplicitlyCopyable)
    # An implicit copy of the borrowed value: exercises the Copyable witness
    # reached through the refined bound.
    var copy = func
    return copy() + func()


def needs_param_strong[
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int) -> Int,
](func: FuncType) -> Int:
    return func[2](10)


def param_where_refines[
    FuncType: def[width: Int](Int) -> Int,
](func: FuncType) -> Int where conforms_to(
    FuncType, ImplicitlyCopyable & RegisterPassable
):
    return needs_param_strong(func)


def main():
    def plain() {} -> Int:
        return 7

    # CHECK: non-capturing where: 7
    print("non-capturing where:", where_refines(plain))
    # CHECK: non-capturing assert: 7
    print("non-capturing assert:", assert_refines(plain))

    var captured = 30
    var scale = 4

    def capture_closure() {var captured} -> Int:
        return captured + 1

    # CHECK: capturing where: 31
    print("capturing where:", where_refines(capture_closure))
    # CHECK: capturing assert: 31
    print("capturing assert:", assert_refines(capture_closure))
    # CHECK: copy via refinement: 62
    print("copy via refinement:", copy_via_refinement(capture_closure))

    @always_inline
    def parametric[width: Int](x: Int) {var scale} -> Int:
        return width * x * scale

    # CHECK: parametric where: 80
    print("parametric where:", param_where_refines(parametric))
