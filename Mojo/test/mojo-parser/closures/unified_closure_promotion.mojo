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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -o %t.mlir

# RUN: FileCheck %s --enable-var-scope --check-prefixes=S0 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S1 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S2 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S3 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S4 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S5 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S6 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S7 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S8 < %t.mlir
# COM: Captures are threaded through the closure storage struct's initializer:
# COM: a by mutable reference, b by mut-to-immutable reference, and c by move
# COM: (owned_in_mem). d is omitted because it uses the default convention.
# S0: lit.call {{.*}}::@"captures_with_default_convention()::my_fn::__storage"::@"__init__
# S0-SAME: "a": !lit.ref<!String, mut *"a
# S0-SAME: "b": !lit.ref<!String, muttoimm *"b
# S0-SAME: "c": !lit.ref<!String, mut {{.*}}> owned_in_mem

def captures_with_default_convention():
    var a, b, c, d = ("a", "b", "c", "d")

    def my_fn() {mut a, b, c^, imm}:
        pass

# COM: Verify stateless promoted closures are registered for apply attributes.
# S1-DAG: lit.alias.decl *"dtype{{.*}}": !DType = <apply(:!lit.generator<("n": !Int) -> !DType> @{{.*}}::@"nonsense(::SIMD[DType.int, 1])`{{.*}}", {:scalar<index> 64})>

#


def trigger_dtype():
    comptime k = 64

    def nonsense(n: Int) {} -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    comptime dtype = nonsense(k)
    var x = SIMD[dtype, 1]()
    _ = x

# COM: Verify async closures use an async trait name and async call op.
# S2-DAG: lit.trait.decl @"async def() -> None"
# S2-DAG: sourceName = #debuginfo.source_name<"async def() -> None">
# S2-LABEL: lit.fn @"async_unified_closure()"
# S2-DAG: lit.async.call[!lit.generator<




def async_unified_closure():
    var value = 0

    async def inc() {mut value}:
        value += 1

    _ = inc()

# COM: Verify promoted closures keep captured params before implicit origins.
# S3-LABEL: lit.fn @"trigger_dtype_implicit_origin{{.*}}"<n: !Int>() -> !kgen.none
# S3-DAG: lit.alias.decl *"dtype{{.*}}": !DType = <apply(:!lit.generator<[1]("impl": !lit.ref<!String, imm #lit.comptime.origin> imm_mem) -> !DType> rebind(:!lit.generator<[1]("impl": !lit.ref<!String, imm *[0,0]> imm_mem) -> !DType> @{{.*}}::@"nonsense{{.*}}"<:!Int n, :!AnyType !String>)
# S3-DAG: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:



def trigger_dtype_implicit_origin[n: Int]():
    def nonsense[U: AnyType, //](impl: U) {} -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    comptime dtype = nonsense("here")
    var x = SIMD[dtype, 1]()
    _ = x

# COM: Verify promoted stateless closures can bind captured params
# COM: before satisfying thin function generic constraints.
# S4-LABEL: lit.fn @"trigger_promoted_params{{.*}}"<n: !Int>() -> !kgen.none
# S4-DAG: lit.call tail @{{.*}}::@"takesThin{{.*}}"<:!lit.generator<<"U": !AnyType, +>[1]("impl": !lit.ref<:!AnyType *(0,0), imm *[0,0]> imm_mem) -> !DType> @{{.*}}::@"nonsense{{.*}}"<:!Int n, :!AnyType ?>
# S4-DAG: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:

#


def takesThin[FuncType: def[U: AnyType, //](impl: U) thin -> DType]():
    _ = FuncType("here")


def trigger_promoted_params[n: Int]():
    def nonsense[U: AnyType, //](impl: U) {} -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    takesThin[nonsense]()

# COM: Verify promoted stateless closures create a function wrapper
# COM: when passed as a value to a thin-compatible parameter.
# S5-LABEL: lit.fn @"trigger_promoted_param_wrapper{{.*}}"<n: !Int>() -> !kgen.none
# S5-DAG: %[[S5_WRAP:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct<#PtrWrapper
# S5-DAG: lit.call @{{.*}}::@"def[U: AnyType{{.*}}_PtrWrapper"{{.*}}(%[[S5_WRAP]])
# S5-DAG: %[[S5_WRAP_IMM:.*]] = lit.ref.immut %[[S5_WRAP]]
# S5-DAG: lit.call @{{.*}}::@"takesFatVale{{.*}}"{{.*}}(%[[S5_WRAP_IMM]])
# S5-DAG: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:



def takesFatVale[
    FuncType: def[U: AnyType, //](impl: U) -> DType
](impl: FuncType):
    _ = impl("here")


def trigger_promoted_param_wrapper[n: Int]():
    def nonsense[U: AnyType, //](impl: U) -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    takesFatVale(nonsense)

# COM: Verify comptime conversion of a promoted wrapper value constructs the
# COM: concrete PtrWrapper via apply_result_slot before calling the closure
# COM: parameter.
# S6-LABEL: lit.fn @"s6_trigger
# S6: lit.alias.decl *"X`": !alias_Int1 = <apply(
# S6-SAME: @"take_closure_param[def[n: Int](arg: Int) -> Int & ::AnyType & ::Deinitable & ::Movable]($0)"
# S6-SAME: store_to_mem(apply_result_slot(
# S6-SAME: @"def[n: Int](arg: Int) capturing thin -> Int_PtrWrapper"::@"__init__()"
# S6: lit.call @{{.*}}::@"take_closure_param



def take_closure_param[C: def[n: Int](arg: Int) -> Int](impl: C) -> Int:
    return impl[3](4)


@__parameter
def legacy(arg0: Int) -> Int:
    return arg0 + 3


def s6_trigger[xx: Int, func: def(Int) capturing -> Int]() -> Int:
    def wrapped_ok[n: Int](arg: Int) -> Int:
        return func(arg) + xx

    comptime X = take_closure_param[type_of(wrapped_ok)](wrapped_ok)
    var Y = take_closure_param[type_of(wrapped_ok)](wrapped_ok)

# COM: Verify promoted top-level functions with captured parameters
# COM: build a wrapper whose Impl type is self-contained while preserving the
# COM: promoted function symbol's native parameter ordering.
# S7-DAG: lit.struct.decl @"def[dtype: DType, //, simd_width: Int]() thin -> SIMD[dtype, simd_width]_PtrWrapper"
# S7-DAG: lit.alias.decl dtype: !DType = <__capture_dtype>
# S7-DAG: lit.fn @"__call__[::DType,::SIMD[DType.int, 1]](unified_closure_promotion::def[dtype: DType, //, simd_width: Int]() thin -> SIMD[dtype, simd_width]_PtrWrapper[$0, $1])"
# S7-DAG: {{.*}} = lit.call tail[!lit.generator<() -> !lit.struct<#SIMD {{.*}}: bind_params{{.*}}Impl, :!DType *"Closure_Syn#0", :!Int *"Closure_Syn#1")]()
# S7-DAG: kgen.witness "dtype" : !DType = __capture_dtype
# S7-LABEL: lit.fn @"s7_trigger[::SIMD[DType.int, 1],::DType]()"
# S7: %[[S7_WRAP:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct
# S7-SAME: @{{.*}}::@"compute_init2[::SIMD[DType.int, 1]](){{.*}}"<:!DType *(0,0), :!Int *(0,1)>
# S7: %[[S7_INIT:.*]] = lit.call @{{.*}}::@"def[dtype: DType, //, simd_width: Int]() thin -> SIMD[dtype, simd_width]_PtrWrapper"::@"__init__()"{{.*}}(%[[S7_WRAP]])
# S7: %[[S7_IMM:.*]] = lit.ref.immut %[[S7_WRAP]]
# S7: lit.call @{{.*}}::@"local_higher_order






def local_higher_order[
    rank: Int,
    dtype: DType,
    compute_init: def[simd_width: Int]() -> SIMD[dtype, simd_width],
](compute_init_closure: compute_init):
    pass


def s7_trigger[rank: Int, dtype: DType]():
    def compute_init2[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    local_higher_order[rank, dtype, type_of(compute_init2)](compute_init2)

# COM: Ensure Proper Ordering Of Parameters In Promoted Functions
# S8-DAG: lit.fn @"thinClosure{{.*}}"<T: !AnyType, +, *"list`2x": !lit.struct<#MyList <:!AnyType T>>>() -> !alias_Int1



struct MyList[T: AnyType](Movable where False):
    pass


def callIt[T: AnyType, list: MyList[T]]():
    def thinClosure[list: MyList[T]]() -> Int:
        return 1

    comptime x = thinClosure[list]()
