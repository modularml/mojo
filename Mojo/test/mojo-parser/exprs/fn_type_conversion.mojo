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


def take_func_without_arg_name[f: def(Int) thin -> None]():
    pass


def func_with_arg_name(a: Int):
    pass


# COM: Issue https://github.com/modular/mojo/issues/1307
# COM: Test that functions with defaults can be passed where no defaults are expected
def take_func_without_default[f: def(a: Int) thin -> None]():
    pass


def func_with_default(a: Int = 0):
    pass


# CHECK-LABEL: lit.fn @"test_passing_funcs
def test_passing_funcs():
    # CHECK: lit.call tail @{{.*}}::@"take_func_without_arg_name{{.*}}"<
    # CHECK-SAME: :!lit.generator<(!Int, |) -> !kgen.none> rebind(:!lit.generator<("a": !Int) -> !kgen.none>
    take_func_without_arg_name[func_with_arg_name]()

    # CHECK: lit.call {{.*}}::@"take_func_without_default{{.*}}"<
    # CHECK-SAME: :!lit.generator<("a": !Int) -> !kgen.none> rebind(:!lit.generator<("a": !Int = {:scalar<index> 0}) -> !kgen.none>
    take_func_without_default[func_with_default]()


def fn_doesnt_raise() -> Int:
    pass


def fn_returns_ref(x: String) -> ref[x] String:
    pass


# CHECK-LABEL: lit.fn @"test_more_conversions
def test_more_conversions():
    # CHECK: %test_raises = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
    # CHECK-NEXT: lit.ref.store [[TMP]], %test_raises
    var test_raises: def() thin raises -> Int = fn_doesnt_raise

    # CHECK: %test_result_convert = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
    # CHECK-NEXT: lit.ref.store [[TMP]], %test_result_convert
    var test_result_convert: def() thin raises -> Float32 = fn_doesnt_raise

    # CHECK: %test_error_convert = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
    # CHECK-NEXT: lit.ref.store [[TMP]], %test_error_convert
    var test_error_convert: def() thin raises Float32 -> Float32 = (
        fn_doesnt_raise
    )

    # CHECK: %test_ref_result_convert = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
    # CHECK-NEXT: lit.ref.store [[TMP]], %test_ref_result_convert
    var test_ref_result_convert: def(x: String) thin -> String = fn_returns_ref


# Check that we can take /explicitly copyable/ return values as ref returns.
# This is a hack (see EXPLICIT-COPY-REF-RETURN) to support __next__ promoting
# its result type.  We should remove this when we have more powerful Iterator
# traits and origins that can support that.
trait TraitExpectingValueReturn:
    comptime Element: Deinitable

    def return_value(self) -> Self.Element:
        ...


struct StructProvidingRefReturn[T: Copyable & Deinitable](
    TraitExpectingValueReturn, Movable where False
):
    comptime Element = Self.T

    def return_value(self) -> ref[self] Self.T:
        pass


struct FromType(Movable where False):
    var n: Int

    def __init__(out self, n: Int):
        self.n = n


struct ToTypeImm(Movable where False):
    var n: Int

    @implicit
    def __init__[O: Origin[]](out self, ref[O] f: FromType):
        self.n = f.n


struct ToTypeMut(Movable where False):
    var n: Int

    @implicit
    def __init__[O: Origin[mut=True]](out self, ref[O] f: FromType):
        self.n = f.n


def useToType(s: ToTypeImm):
    pass


def useToType(s: ToTypeMut):
    pass


# Tests that implicit conversions that are ref-dependent are cached correctly.
# If the implicit conversion cache does not take into account the RefType, the
# second call to useToType will emit an error.
def test[
    O: Origin[mut=True], O2: Origin[]
](ref[O] fImm: FromType, ref[O2] fMut: FromType):
    # CHECK: lit.call {{.*}}@ToTypeMut::@"__init__
    useToType(ToTypeMut(fImm))
    # CHECK: lit.call {{.*}}@ToTypeImm::@"__init__
    useToType(fMut)


def main():
    var f1 = FromType(1)
    var f2 = FromType(2)
    test(f1, f2)
