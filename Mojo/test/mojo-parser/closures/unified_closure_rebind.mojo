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
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S2 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S3 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S4 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S5 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S6 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S7 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S8 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S9 < %t.mlir
# COM: "U" cannot be called "T" until MOCO-4028 is fixed
# COM: The captured parameter becomes an alias on the trait
# S0: lit.trait.decl @"def{{.*}} -> U{1}"
# S0-NEXT: lit.alias.decl U: !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable
# COM: The captured parameter becomes a parameter of the storage struct
# S0: lit.struct.decl @"makeIt{{.*}}::parametric::__storage"<U: !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable, {{.*}}>
# S0: kgen.witness "U" : !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable = U
# COM: No parametric wrapper around storage.
# S0-NOT: lit.struct.decl @"def{{.*}} -> U{1}_{{.*}}"<impl:


def makeIt[U: TrivialRegisterPassable](a: U):
    def parametric() {var a} -> U:
        return a


def conditionallyDevicePassable(x: Int):
    def device_passable() {var} -> Int:
        return x


# COM: Ensure external parameter references are pulled into alias decls
# S2-DAG: lit.trait.decl @"def{{.*}} -> None"
# S2-DAG: lit.alias.decl T: !AnyType_DoIt
# S2-DAG: lit.alias.decl TT: !AnyType_DoIt


trait DoIt:
    def thing(self):
        ...


struct House[T: DoIt](Movable where False):
    def aMethod[C: def(x: Self.T)](self, impl: C):
        pass


def useIt[TT: DoIt, C: def(x: TT)](impl: C):
    pass


# S3-DAG: kgen.conformance {{.*}}@RegisterPassable {


def takesRegisterPassable[T: RegisterPassable](impl: T):
    pass


def addTrivialRegisterPassable(x: Int):
    def closure() {var} -> Int:
        return x

    takesRegisterPassable(closure)


# COM: Verify top-level function symbols get conformance for count's closure
# COM: trait.
# S4-DAG: lit.struct.decl @"def[w: Int](vec: s4_ToySIMD[Int(1), w]) thin -> s4_ToyMask[Int(1), w]_{{.*}}"
# S4-DAG: kgen.conformance @"def[{{.*}}w: Int](vec: s4_ToySIMD[dtype_tag, w]) -> s4_ToyMask[dtype_tag, w]{1}" {
# S4-DAG: kgen.witness "__call__{{.*}}" : !lit.generator
# S4-DAG: kgen.witness "dtype_tag" : !Int = {:scalar<index> 1}


@fieldwise_init
struct s4_ToyBool(Movable where False):
    var value: Int


@fieldwise_init
struct s4_ToyMask[dtype_tag: Int, w: Int](Movable where False):
    var value: Int


struct s4_ToySIMD[dtype_tag: Int, w: Int](Movable where False):
    pass


struct s4_ToyScalar[dtype_tag: Int](Movable where False):
    pass


@fieldwise_init
struct s4_MiniSpan[dtype_tag: Int](Movable where False):
    var value: Int

    def count[
        F: def[w: Int](vec: s4_ToySIMD[Self.dtype_tag, w]) -> s4_ToyMask[
            Self.dtype_tag, w
        ]
    ](self, func: F) -> Int:
        return 0


def is_vec_a[w: Int](vec: s4_ToySIMD[1, w]) -> s4_ToyMask[1, w]:
    _ = vec
    return s4_ToyMask[1, w](0)


def repro_top_level():
    var s = s4_MiniSpan[1](0)
    _ = s.count(is_vec_a)


# COM: Verify nested captured closures get conformance for count's
# COM: closure trait on the storage struct (no parametric wrapper).
# S5-DAG: lit.struct.decl @"{{.*}}is_vec_a_capturing::__storage"
# S5-DAG: kgen.conformance @"def[{{.*}}u: Int](vec: s5_ToySIMD[dtype_tag, u]) -> s5_ToyMask[dtype_tag, u]{1}" {
# S5-DAG: kgen.witness "__call__{{.*}}" : !lit.generator
# S5-DAG: kgen.witness "dtype_tag" : !Int = {:scalar<index> 1}


@fieldwise_init
struct s5_ToyBool(Movable where False):
    var value: Int


@fieldwise_init
struct s5_ToyMask[dtype_tag: Int, u: Int](Movable where False):
    var value: Int


struct s5_ToySIMD[dtype_tag: Int, u: Int](Movable where False):
    pass


struct s5_ToyScalar[dtype_tag: Int](Movable where False):
    pass


@fieldwise_init
struct s5_MiniSpan[dtype_tag: Int](Movable where False):
    var value: Int

    def count[
        F: def[u: Int](vec: s5_ToySIMD[Self.dtype_tag, u]) -> s5_ToyMask[
            Self.dtype_tag, u
        ]
    ](self, func: F) -> Int:
        return 0


def repro_capturing(mem: String):
    var capture = 0

    def is_vec_a_capturing[
        u: Int
    ](vec: s5_ToySIMD[1, u]) {var capture, var mem} -> s5_ToyMask[1, u]:
        _ = vec
        _ = capture
        return s5_ToyMask[1, u](0)

    var s = s5_MiniSpan[1](0)
    _ = s.count(is_vec_a_capturing)


# COM: Verify nested type parameters constrained by a trait (not just Int
# COM: parameters) get conformance resolved from nested struct type arguments.
# S6-DAG: lit.struct.decl @"{{.*}}apply_concrete::__storage"
# S6-DAG: kgen.conformance @"def[{{.*}}n: Int](item: Box[E, n]) -> Box[E, n]{1}" {
# S6-DAG: kgen.witness "__call__{{.*}}" : !lit.generator
# S6-DAG: kgen.witness "E" : !AnyType_ElemLike = !ConcreteElem


trait ElemLike:
    pass


struct ConcreteElem(ElemLike, Movable where False):
    pass


@fieldwise_init
struct Box[E: ElemLike, n: Int](Movable where False):
    var value: Int


@fieldwise_init
struct Store[E: ElemLike](Movable where False):
    var value: Int

    def apply[
        F: def[n: Int](item: Box[Self.E, n]) -> Box[Self.E, n]
    ](self, func: F) -> Int:
        return 0


def repro_nested_type_param(mem: String):
    var capture = 0

    def apply_concrete[
        n: Int
    ](item: Box[ConcreteElem, n]) {var capture, var mem} -> Box[
        ConcreteElem, n
    ]:
        _ = item
        _ = capture
        return Box[ConcreteElem, n](0)

    var s = Store[ConcreteElem](0)
    _ = s.apply(apply_concrete)


# COM: Verify that custom types (the result type !kgen.none in this case) are compared using equality
# S7-DAG: lit.struct.decl @"{{.*}}my_func::__storage"
# S7-DAG: kgen.conformance @"def[width: Int, rank: Int, alignment: Int = Int(1)]() -> None" {
# S7-DAG:   kgen.witness "__call__{{.*}}" : !lit.generator


def print(x: Int):
    pass


def s7_callee[
    func: def[width: Int, rank: Int, alignment: Int = 1]() -> None,
    //,
    simd_width: Int,
](shape: Int, ctx: Int, closure: func):
    closure[simd_width, 2]()


def main() raises:
    var x = 42
    var mem: String = "hello"

    @always_inline
    def my_func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ]() {imm x, var mem}:
        print(x)

    s7_callee[simd_width=4](10, 11, my_func)


# COM: Verify the result is properly rebound in the struct wrapper when a closure
# COM: lazily conforms to a trait whose return type contains an alias parameter.
# S8: lit.struct.decl @"def[width: Int]() thin -> V[Int(42), width]_PtrWrapper"
# S8: lit.fn @"__call__$def{{.*}} -> V{{.*}}"
# S8: kgen.rebind %{{.*}} : {{.*}}{:scalar<index> 42}{{.*}} to {{.*}}*"Closure_Syn#0"{{.*}}
# S8-NEXT: lit.return
# S8: kgen.conformance @"def[dtype: Int, //, width: Int]() -> V[dtype, width]{1}" {
# S8-NEXT: kgen.witness "__call__{{.*}}" : !lit.generator
# S8-NEXT: kgen.witness "dtype" :{{.*}} = {:scalar<index> 42}


@fieldwise_init
struct V[dtype: Int, width: Int](RegisterPassable):
    var _v: Int


def s8_callee[
    dtype: Int,
    F: RegisterPassable & def[width: Int]() -> V[dtype, width],
](closure: F):
    var result = closure[4]()


def rebindResult():
    def my_closure[width: Int]() {} -> V[42, width]:
        return V[42, width](0)

    s8_callee[42](my_closure)


# COM: Verify ParamListAttr matching: closure returning Tuple with parameterized
# COM: elements requires recursive matching through #kgen.param_list param values.
# S9-DAG: lit.struct.decl @"{{.*}}my_map_fn::__storage"
# S9-DAG: @"def[rank: Int, //](ToyIndex[rank]) -> Tuple[ToyIndex[rank], ToyIndex[rank]]{1}" {
# S9-DAG:   kgen.witness "__call__{{.*}}" : !lit.generator
# S9-DAG:   kgen.witness "rank" : !Int = {:scalar<index> 2}


struct ToyIndex[size: Int](RegisterPassable):
    var _v: Int

    def __init__(out self):
        self._v = 0


def variadic_callee[
    rank: Int,
    map_fn: def(ToyIndex[rank]) -> Tuple[
        ToyIndex[rank],
        ToyIndex[rank],
    ],
](closure: map_fn):
    var point = ToyIndex[rank]()
    var result = closure(point)


def repro_variadic_attr():
    var x = 10
    var mem: String = "hello"

    def my_map_fn(
        point: ToyIndex[2],
    ) {imm x, var mem} -> Tuple[ToyIndex[2], ToyIndex[2]]:
        return ToyIndex[2](), ToyIndex[2]()

    variadic_callee[2, type_of(my_map_fn)](my_map_fn)
