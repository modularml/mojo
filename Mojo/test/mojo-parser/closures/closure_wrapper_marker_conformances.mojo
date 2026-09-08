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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# COM: Regression test for MOCO-4240. The synthesized closure wrapper struct
# COM: must carry a ConformanceOp for every trait claimed by its canonical
# COM: trait, including the method-less marker traits, so that
# COM: `where conforms_to(...)` refinement on a weakly-bounded closure type
# COM: parameter evaluates to True. The call trait conformance pins the
# COM: matches to the wrapper: the trailing @Int struct decl check ensures the
# COM: marker conformances matched inside the wrapper, not a later struct.


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


def use() -> Int:
    def plain() {} -> Int:
        return 7

    return where_refines(plain)


# CHECK: lit.struct.decl @"def() thin -> Int_PtrWrapper"
# CHECK: kgen.conformance @std::@builtin::@stubs::@Deinitable
# CHECK: kgen.conformance @std::@builtin::@stubs::@Movable
# CHECK: kgen.conformance @std::@builtin::@stubs::@Copyable
# CHECK: kgen.conformance @"def() -> Int"
# CHECK: kgen.conformance @std::@builtin::@stubs::@AnyType
# CHECK: kgen.conformance @std::@builtin::@stubs::@ImplicitlyCopyable
# CHECK: kgen.conformance @std::@builtin::@stubs::@TrivialRegisterPassable
# CHECK: kgen.conformance @std::@builtin::@stubs::@RegisterPassable
