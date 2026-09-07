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
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S9 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S10 < %t.mlir
# COM: Thin Closures With Concrete Captures Are Properly Lifted
# S0-DAG: lit.fn @"compute_gpu


comptime SIMDSize = Int


struct IndexList[r: Int](Movable where False):
    pass


def target[
    rank: Int,
    dtype: DType,
    ComputeFnType: def[simd_width: SIMDSize](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width],
](compute_func: ComputeFnType,) raises:
    pass




def repro_stencil_indirect_call[
    dtype: DType,
    num_channels: Int,
]() raises:
    comptime rank = 4

    def compute_gpu[
        simd_width: SIMDSize
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        _ = point
        return val + result

    target[rank, dtype, type_of(compute_gpu)](compute_gpu)

# COM: Stateless nested functions whose signature references both a captured
# COM: parameter (dtype) and a free wildcard (alignment) are promoted to a
# COM: top-level fn whose call site binds the captured params directly.
# S1-LABEL: lit.fn @"outer
# S1-DAG: lit.call tail @{{.*}}::@"inner{{.*}}"<:!DType dtype, :!Int *"a.alignment{{.*}}">(%a)



def _current_target() -> __mlir_type.`!kgen.target`:
    return __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`


def _align_of[
    dtype: DType, target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
    return 1


@fieldwise_init
struct LayoutTensor[
    dtype: DType, alignment: Int = _align_of[dtype, _current_target()]()
](TrivialRegisterPassable):
    pass


def outer[dtype: DType, valid: Bool](a: LayoutTensor[dtype, ...]) raises:
    comptime assert valid, "need float"

    def inner(buf: LayoutTensor[dtype, ...]) -> LayoutTensor[dtype]:
        return LayoutTensor[dtype]()

    var x = inner(a)

# COM: Captured type params live on the storage struct; only free param C
# COM: remains on the promoted call method.
# S2-LABEL: lit.struct.decl @"{{.*}}s2_top{{.*}}::closure::__storage"
# S2-DAG: lit.fn @"closure{{.*}}"<C: !AnyType_Copyable_Movable, +>



def s2_bind[
    D: Copyable, E: Copyable, FuncType: def[F: Copyable](a: D, b: E, c: F)
](impl: FuncType):
    pass




def s2_top[A: Copyable, B: Copyable](aa: A, bb: B):
    def closure[C: Copyable, //](a: A, b: B, c: C) {imm}:
        pass

    closure(aa, bb, 3)
    s2_bind[A, B, type_of(closure)](closure)

# COM: Verify Lazy Conformance (adaptor on storage, not parametric wrapper).
# S3-LABEL: lit.struct.decl @"{{.*}}s3_top{{.*}}::closureConcrete::__storage"
# S3: lit.fn @"__call__$def
# S3-NEXT: kgen.rebind %a : !lit.ref<:!AnyType_Copyable_Movable *"Closure_Syn#0", imm *"1_unnamed`"> to !lit.ref<!String, imm *"1_unnamed`">
# S3-NEXT: kgen.rebind %b : !lit.ref<:!AnyType_Copyable_Movable *"Closure_Syn#1", imm *"2_unnamed`"> to !lit.ref<!String, imm *"2_unnamed`">



def s3_bind[
    D: Copyable, E: Copyable, FuncType: def[F: Copyable](a: D, b: E, c: F)
](impl: FuncType):
    pass




def s3_top():
    def closureConcrete[C: Copyable, //](a: String, b: String, c: C) {imm}:
        pass

    s3_bind[String, String, type_of(closureConcrete)](closureConcrete)

# COM: Origins are properly captured and lifted into the storage struct
# S4: lit.struct.decl @"s4_demo{{.*}}::write::__storage"
# S4-SAME: <{{.*}}*"o._mlir_origin`": origin<true>, o: !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *"o._mlir_origin`">>



def can_mutate[FuncType: def() -> None](impl: FuncType):
    impl()


def s4_demo[
    o: Origin[mut=True]
](ptr: UnsafePointer[Int, o, address_space=.GENERIC],):
    def write() {imm ptr}:
        ptr.store(0, 3)

    can_mutate(write)

# COM: If a mutable origin is captured but only in the context of a cast to immutable, do not lift and bind a mutable origin to the closure struct
# S5: lit.struct.decl @"s5_demo{{.*}}::imm::__storage"
# S5-SAME: <{{.*}}*"o._mlir_origin`": origin<false>, {{.*}}*"immut_ptr{{.*}}": origin<false>



def must_be_imm_only[
    Mut: Bool, //, o: Origin[mut=Mut], FuncType: def() -> None
](
    impl: FuncType,
    ptr: UnsafePointer[Int, o, address_space=.GENERIC],
):
    impl()


def s5_demo[
    o: Origin[mut=True]
](ptr: UnsafePointer[Int, o, address_space=.GENERIC],):
    var immut_ptr = ptr.as_imm()


    def imm() {imm immut_ptr}:
        _ = immut_ptr[0]

    must_be_imm_only(imm, immut_ptr)

# COM: MOCO-4128
# S6-LABEL: lit.fn @"apply_closure
# COM: Make sure the field type is paramtric over `T`, not a plain `AnyType`.
# S6: lit.call {{.*}}::f::__storage"::@"__init__
# S6-SAME: "x": !lit.ref<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable T,



def apply_closure[T: TrivialRegisterPassable](x: T):
    def f() {var x}:
        _ = x

    f()


def main():
    apply_closure(Int(42))

# COM: Use where clauses to infer parameters that depend on aliases
# S7-LABEL: lit.fn @"thing
# S7-DAG: lit.call @{{.*}}::@"typed_raises{{.*}}"[{{.*}}]<:!AnyType !Int



def typed_raises[
    R: AnyType,
    F: def() raises R,
    //,
](f: F):
    pass


def thing():
    def closure() raises Int:
        pass

    typed_raises(closure)

# COM: Verify a promoted method reference bridges a captured `Int` parameter to a
# COM: differently-typed (`MyInt`) generator parameter. The captured `Self.width`
# COM: is an `Int`, but `MySIMD.__add__` needs a `MyInt`, so a generator attr
# COM: bridges the gap inside the `_PtrWrapper` Impl type,
# COM: i.e. gen<:!Int x> __add__[MyInt(x)]().
# S8-LABEL: lit.fn @"add()"()
# S8: %__call_result_tmp__ = lit.var.decl "__call_result_tmp__" synth :
# S8-SAME: !lit.ref<!lit.struct<#PtrWrapper <:!Int width, :!lit.generator<<"width": !Int, +>
# S8-SAME: #kgen.gen<#kgen.func.symbol<@{{.*}}::@MySIMD::@"__add__({{.*}}MySIMD[$0],{{.*}}MySIMD[$0])"<:!MyInt



struct MyInt(Movable where False):
    @implicit
    def __init__(out self, value: Int):
        pass


struct MySIMD[width: MyInt](Movable where False):
    def __add__(self, other: MySIMD[Self.width]) -> MySIMD[Self.width]:
        pass


struct Foo[width: Int](Movable where False):
    @staticmethod
    def helper(
        func: Some[
            def(MySIMD[Self.width], MySIMD[Self.width]) -> MySIMD[Self.width]
        ],
    ):
        pass

    @staticmethod
    def add():
        Self.helper(MySIMD[Self.width].__add__)

# S9: lit.trait.decl @"def{{.*}}mut Builder[origin]{{.*}}definesClosure
# S9: lit.alias.decl {{.*}}origin.mut`{{.*}}: !Bool
# S9: lit.alias.decl {{.*}}origin._mlir_origin`1{{.*}}: origin<
# S9: lit.alias.decl origin: !lit.struct<
# S9-SAME: get_witness<{{.*}}"origin.mut`">
# S9-SAME: get_witness<{{.*}}"origin._mlir_origin`1">
# S9-LABEL: lit.fn @"region[
# S9: lit.call
# S9-SAME: bind_params
# S9-SAME: get_witness<{{.*}}work.T{{.*}}__call__
# COM: The three captured origin parameters are bound, in order, to `self`'s
# COM: Builder origin params (`origin.mut`, `origin._mlir_origin`, `origin`).
# S9-SAME: , :!Bool *"origin.mut`", :origin<{{.*}}> *"origin._mlir_origin`1", :!lit.struct<{{.*}}> origin)
# S9-SAME: (%work, %self,

#
# Capture dependencies are interned using get_witness attr
#


struct Builder[origin: Origin](Movable):
    var x: Int

    def test(mut self) raises:
        pass

    def region(mut self, work: Some[def(mut Self) raises]) raises:
        work(self)
        pass

# COM: A captured local's origin is promoted into a nested closure's storage
# COM: struct and interned as a witness.
# S10-DAG: lit.struct.decl @"{{.*}}capture_nested()::outer2::__storage"<["z`"]*"z`": origin<false>
# S10-DAG: kgen.witness "z`" : origin<false> = *"z`"


def capture_nested() -> Int:
    var z = 1

    def inner(x: Int) {imm z} -> Int:
        return z

    def outer2(y: Int) {imm z, imm inner} -> type_of(inner):
        return inner

    return outer2(1)(2)
