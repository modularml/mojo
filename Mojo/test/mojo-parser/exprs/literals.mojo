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


struct SimpleIntRange(TrivialRegisterPassable):
    def __init__(out self):
        pass

    def __len__(self) -> Int:
        pass

    def __next__(mut self) raises StopIteration -> Int:
        pass

    def __iter__(self) -> Self:
        pass


def var_let_decls():
    # CHECK: %xx = lit.var.decl "xx" var
    # CHECK: %[[V1:.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
    # CHECK: lit.ref.store %[[V1]], %xx
    var xx = 42

    # CHECK: lit.alias.decl {{.*}}il{{.*}}#IntLiteral <:!pop.int_literal 43>
    comptime il = 43

    # CHECK: %yy = lit.var.decl "yy" var
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !FloatDyn = <{{.*}}{:scalar<f64> "1"}
    # CHECK: lit.ref.store [[TMP]], %yy
    var yy = 1.0

    # CHECK: lit.alias.decl {{.*}}fl1{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<2|1>>> = <{}>
    comptime fl1 = 2.0
    # CHECK: lit.alias.decl {{.*}}fl2{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<63|10>>> = <{}>
    comptime fl2 = 6.3
    # CHECK: lit.alias.decl {{.*}}fl3{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<41|2>>> = <{}>
    comptime fl3 = 20.5
    # CHECK: lit.alias.decl {{.*}}fl4{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<-41|2>>> = <{}>
    comptime fl4 = -20.5
    # CHECK: lit.alias.decl {{.*}}fl5{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<neg_zero>>> = <{}>
    comptime fl5 = -0.0

    # Smallest positive float (moco-1796)
    # CHECK: lit.alias.decl {{.*}}fl6{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<1|2{{(0)+}}>>> = <{}>
    comptime fl6 = 5e-324

    # TODO - Python raises an error when dividing by zero.  We need support for
    # parameter-time evaluation of `raise` to support that semantics, in which
    # case these will be static errors instead.
    # CHECK: lit.alias.decl {{.*}}flDivZero{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<nan>>> = <{}>
    comptime flDivZero = 5.0 / 0.0
    # CHECK: lit.alias.decl {{.*}}flDivNegZero{{.*}}#FloatLiteral <:!pop.float_literal #pop.float_literal<nan>>> = <{}>
    comptime flDivNegZero = 5.0 / -0.0

    # CHECK: %str = lit.var.decl {{.*}} : !lit.ref<!String,
    # CHECK: [[CONST:%.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "hello">> = <{}>
    # CHECK: lit.call {{.*}}@String::@"__init__{{.*}}([[CONST]], %str)
    var str = "hello"


# ===----------------------------------------------------------------------=== #
# List Literals
# ===----------------------------------------------------------------------=== #


struct IntList(TrivialRegisterPassable):
    def __init__(
        out self, *list_elements: Int, __list_literal__: NoneType = None
    ):
        pass

    def append(mut self, value: Int):
        pass


def inspect(list: List[_]):
    pass


# CHECK-LABEL: lit.fn @"test_list_literal
def test_list_literal():
    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create
    # CHECK: [[NONE_MARKER:%.*]] = kgen.param.constant: !NoneType
    # CHECK: lit.call {{.*}}@Array::@"__init__{{.*}}({{.*}}, [[NONE_MARKER]], %a)
    var a = [1, 2, 3]

    # CHECK-DAG: [[TMP1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-DAG: [[TMP2:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-DAG: [[TMP3:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 3}>
    # CHECK-DAG: {{%.*}} = pop.array.create [{{.*}}]
    # CHECK: [[NONE_MARKER:%.*]] = kgen.param.constant: !NoneType
    # CHECK: lit.call {{.*}}@IntList::@"__init__{{.*}}({{.*}}, [[NONE_MARKER]])
    var b: IntList = [1, 2, 3]

    # CHECK: [[VARIADIC:%.*]] = kgen.param.constant: !lit.ref<array<0, !lit.ref<!Int, imm {}>>, imm {}> = <#interp.pointer<0>>
    # CHECK: [[NONE_MARKER:%.*]] = kgen.param.constant: !NoneType
    # CHECK: lit.call {{.*}}@IntList::@"__init__{{.*}}({{.*}}, [[NONE_MARKER]])
    var c: IntList = []

    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}<:!AnyType_Copyable_Movable !FloatDyn>
    inspect([1.0, 2])

    # MOCO-2085: List comprehensive fails without explicit use of var
    var impl_definition = [i for i in SimpleIntRange()]


struct ParametricList[T: Movable](Movable where False):
    def __init__(out self, var *elements: Self.T, __list_literal__: NoneType):
        pass


# CHECK-LABEL: lit.fn @"test_parametric_list_literal()"
def test_parametric_list_literal():
    # The struct parameter `T` is inferred as `Int` from the list literal, and
    # the `__init__` call writes its `out self` result directly into x's decl.

    # CHECK: [[VAR_DECL_X:%.*]] = lit.var.decl "x" var : !lit.ref<!lit.struct<#ParametricList
    # CHECK: lit.call @{{.*}}::@ParametricList::@"__init__[{{.*}}]]($0$*,__list_literal__:::NoneType)"{{.*}}<:!AnyType_Movable !Int{{.*}}({{.*}}, [[VAR_DECL_X]]) :
    var x: ParametricList = [1, 2, 3]


# CHECK-LABEL: lit.fn @"test_list_comprehension
def test_list_comprehension():
    # CHECK-NEXT: %a_collection = lit.var.decl{{.*}}#List <:!AnyType_Copyable_Movable !Int>
    # CHECK: lit.loop {
    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous*"
    # CHECK:      lit.call {{.*}}SimpleIntRange::@"__next__{{.*}}(%$ITER, %__call_error_tmp__, [[ANON]])
    # CHECK: %i1 = lit.var.decl "i1" ref
    # CHECK: [[TMPREF:%.*]] = lit.ref.load %i1
    # CHECK-NEXT: [[REB:%.*]] = kgen.rebind [[TMPREF]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load [[REB]]
    # CHECK-NEXT: [[TMP2:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}@SIMD::@"__mul__{{.*}}([[TMP]], [[TMP2]]
    # CHECK:      lit.call {{.*}}@List::@"append
    # CHECK-NEXT: lit.loop.continue
    # CHECK: }
    var a_collection = [i1 * 2 for i1 in SimpleIntRange()]

    # CHECK: %b_collection = lit.var.decl{{.*}}#List <:!AnyType_Copyable_Movable !Int>
    # CHECK: lit.loop {
    # CHECK-NEXT: [[ANONI2:%.*]] = lit.var.decl "anonymous*"

    # CHECK:   lit.loop {
    # CHECK-NEXT: [[ANONI3:%.*]] = lit.var.decl "anonymous*"
    # CHECK:  [[TMP:%.*]] = lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK: [[TMP2REF:%.*]] = lit.ref.load %i2
    # CHECK-NEXT: [[TMP3REF:%.*]] = lit.ref.load %i3
    # CHECK-NEXT: [[REB2:%.*]] = kgen.rebind [[TMP2REF]]
    # CHECK-NEXT: [[REB3:%.*]] = kgen.rebind [[TMP3REF]]
    # CHECK-NEXT: [[TMP2:%.*]] = lit.ref.load [[REB2]]
    # CHECK-NEXT: [[TMP3:%.*]] = lit.ref.load [[REB3]]
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}@SIMD::@"__mul__{{.*}}([[TMP2]], [[TMP3]]
    # CHECK:      lit.call {{.*}}@List::@"append
    var b_collection = [
        i2 * i3 for i2 in SimpleIntRange() for i3 in SimpleIntRange()
    ]

    # Inferred to type IntList and using an "if" clause.
    # CHECK: %c_collection = lit.var.decl{{.*}}!lit.ref<!IntList,
    # CHECK: lit.loop {
    # CHECK-NEXT: [[ANONI4:%.*]] = lit.var.decl "anonymous*"
    # CHECK:     lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK:     [[TMP:%.*]] = lit.ref.load %i4
    # CHECK-NEXT:    [[REB:%.*]] = kgen.rebind [[TMP]]
    # CHECK-NEXT:    [[TMPREF:%.*]] = lit.ref.load [[REB]]
    # CHECK-NEXT:    @SIMD::@"__bool__
    # CHECK-NEXT:    @Bool::@"__mlir_bool__
    # CHECK-NEXT:    hlcf.elif %{{.*}} {
    # CHECK-NEXT: [[RES:%.*]] = lit.ref.load %i4
    # CHECK-NEXT  lit.call {{.*}}@IntList::@"append{{.*}}(%c_collection, [[RES]])
    var c_collection: IntList = [i4 for i4 in SimpleIntRange() if i4]

    var xs = [1, 2, 3]

    # CHECK:   [[LEN:%.*]] = {{.*}}@List::@"__len__
    # CHECK:   [[LEN0:%.*]] = kgen.rebind [[LEN]]
    # CHECK:   [[ZERO:%.*]] = kgen.param.constant:{{.*}}{:scalar<index> 0}
    # CHECK:   [[NE:%.*]] = {{.*}}@SIMD::@"__ne__{{.*}}([[LEN0]], [[ZERO]])
    # CHECK:   [[RET:%.*]] = {{.*}}@Bool::@"__mlir_bool__{{.*}}([[NE]])
    # CHECK: hlcf.elif [[RET]] {
    # CHECK:   [[NONE:%.*]] = kgen.param.constant: none = <#kgen.none>
    # CHECK:   lit.return [[NONE]]
    # CHECK:   hlcf.yield
    # CHECK: } else {
    # CHECK:   hlcf.yield
    # CHECK: }
    if [x > 1 for x in xs].__len__() != 0:
        return

    # CHECK: [[LIST:%.*]] = lit.var.decl{{.*}} : !lit.ref<!lit.struct<#List
    # CHECK: lit.loop {
    # CHECK:   [[GT:%.*]] = lit.call {{.*}}@SIMD::@"__gt__
    # CHECK:   [[ANON:%.*]] = lit.var.decl
    # CHECK:   lit.ref.store [[GT]], [[ANON]]
    # CHECK:   lit.call {{.*}}@List::@"append{{.*}}([[LIST]],
    # CHECK:   lit.loop.continue
    # CHECK: } else {
    # CHECK:   lit.loop.yield
    # CHECK-NEXT: }

    # CHECK: lit.loop {
    # CHECK: %y = lit.var.decl "y" ref :
    # CHECK: lit.loop.continue
    # CHECK: } else {
    # CHECK: lit.loop.yield
    # CHECK-NEXT: }
    for y in [x > 1 for x in xs]:
        pass


# ===----------------------------------------------------------------------=== #
# Dictionary Literals
# ===----------------------------------------------------------------------=== #


struct MyDict[
    K: Copyable & Deinitable, V: Copyable & Deinitable
](Movable where False):
    def __init__(
        out self,
        var keys: List[Self.K],
        var values: List[Self.V],
        __dict_literal__: NoneType,
    ):
        pass


struct IntDict(Movable where False):
    def __init__(
        out self,
        keys: IntList,
        values: IntList,
        __dict_literal__: NoneType = None,
    ):
        pass


# CHECK-LABEL: lit.fn @"test_dict_literal
def test_dict_literal(aBool: Bool):
    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[KEYS_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[VALUES_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@Dict::@"__init__{{.*}}([[KEYS_LIST]], [[VALUES_LIST]], {{.*}}, %a) :
    var a = {1: aBool, 2: aBool}

    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[KEYS_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[VALUES_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@MyDict::@"__init__{{.*}}([[KEYS_LIST]], [[VALUES_LIST]], {{.*}}, %b) :
    var b: MyDict[Int, Bool] = {1: aBool, 2: aBool}

    # CHECK: [[KEYS_LIST:%.*]] = lit.call {{.*}}@IntList::@"__init__
    # CHECK: [[VALUES_LIST:%.*]] = lit.call {{.*}}@IntList::@"__init__
    # CHECK: lit.call {{.*}}@IntDict::@"__init__{{.*}}([[KEYS_LIST]], [[VALUES_LIST]], {{.*}}, %c) :
    var c: IntDict = {1: 7, 2: 8}


# CHECK-LABEL: lit.fn @"test_dict_comprehension
def test_dict_comprehension():
    # CHECK-NEXT: %a_collection = lit.var.decl{{.*}}#Dict <:!AnyType_Copyable_Deinitable_Movable !Int, :!AnyType_Copyable_Deinitable_Movable !String>
    # CHECK: lit.loop {
    # CHECK-NEXT: [[ANONI:%.*]] = lit.var.decl "anonymous*"
    # CHECK:     lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK:      %i = lit.var.decl "i"
    # CHECK:      %__call_result_tmp__ = lit.var.decl "__call_result_tmp__"
    # CHECK:      lit.call {{.*}}String::@"__init__(){{.*}}(%__call_result_tmp__)
    # CHECK:      [[TMP:%.*]] = lit.ref.immut %__call_result_tmp__
    # CHECK:      lit.call {{.*}}@Dict::@"__setitem__{{.*}}(%a_collection, {{.*}}, [[TMP]])
    # CHECK-NEXT: lit.loop.continue
    # CHECK: }
    var a_collection = {i: String() for i in SimpleIntRange()}


# ===----------------------------------------------------------------------=== #
# Set Literals
# ===----------------------------------------------------------------------=== #


struct MySet[T: AnyType](Movable where False):
    def __init__(out self, var *values: Self.T, __set_literal__: NoneType):
        pass


def param_infer_equal[T: AnyType](a: T, b: T):
    pass


# CHECK-LABEL: lit.fn @"test_set_literal
def test_set_literal():
    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create
    # CHECK: [[NONE_MARKER:%.*]] = kgen.param.constant: !NoneType
    # CHECK: lit.call {{.*}}@Set::@"__init__{{.*}}({{.*}}, [[NONE_MARKER]], %a)
    var a = {1, 2, 3}

    # MOCO-1974 - Param inference isn't substituting full type
    # CHECK: lit.call {{.*}}@"param_infer_equal
    param_infer_equal(a, {})

    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create
    # CHECK: [[NONE_MARKER:%.*]] = kgen.param.constant: !NoneType
    # CHECK: %b = lit.var.decl
    # CHECK: lit.call {{.*}}@MySet::@"__init__{{.*}}({{.*}}, [[NONE_MARKER]], %b)
    var b: MySet[Int] = {1, 2}


# CHECK-LABEL: lit.fn @"test_set_comprehension
def test_set_comprehension():
    # CHECK-NEXT: %a_collection = lit.var.decl{{.*}}#Set <:!AnyType !Int>
    # CHECK: lit.loop {
    # CHECK-NEXT: [[ANONI1:%.*]] = lit.var.decl "anonymous*"
    # CHECK:   lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK: %i1 = lit.var.decl "i1"
    # CHECK: [[TMP:%.*]] = lit.ref.load %i1
    # CHECK-NEXT: [[REB:%.*]] = kgen.rebind [[TMP]]
    # CHECK-NEXT: [[TMPREF:%.*]] = lit.ref.load [[REB]]
    # CHECK-NEXT: [[TMP2:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}@SIMD::@"__mul__{{.*}}([[TMPREF]], [[TMP2]]
    # CHECK:      lit.call {{.*}}@Set::@"add
    # CHECK-NEXT: lit.loop.continue
    # CHECK: }
    var a_collection = {i1 * 2 for i1 in SimpleIntRange()}


# ===----------------------------------------------------------------------=== #
# Initializer Lists
# ===----------------------------------------------------------------------=== #


struct InitType[T: AnyType](Movable where False):
    def __init__(out self, value: Self.T):
        pass

    def __init__(out self, value: Self.T, value2: Int):
        pass


# CHECK-LABEL: lit.fn @"test_initializer_list
def test_initializer_list():
    # CHECK: [[TMP:%.*]] = lit.ref.immut
    # CHECK: lit.call {{.*}}@InitType::@"__init__{{.*}}([[TMP]], %a)
    var a: InitType[Int] = {1}
    # CHECK: [[TMP:%.*]] = lit.ref.immut
    # CHECK: [[TWO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK: lit.call {{.*}}@InitType::@"__init__{{.*}}([[TMP]], [[TWO]], %b)
    var b: InitType[Int] = {1, 2}
    # CHECK: [[TMP:%.*]] = lit.ref.immut
    # CHECK: [[INT:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 42}>
    # CHECK: lit.call {{.*}}@InitType::@"__init__{{.*}}([[TMP]], [[INT]], %c)
    var c: InitType[String] = {"foo", 42}


# ===----------------------------------------------------------------------=== #
# Ambiguity for e.g. PythonObject
# ===----------------------------------------------------------------------=== #


# This can be formed with any collection and has its own initializer list too.
struct AnyCollection(Movable where False):
    def __init__(out self):
        pass

    def __init__(out self, value: AnyType):
        pass

    def __init__(out self, var *values: Int, __list_literal__: NoneType):
        pass

    def __init__(out self, var *values: Int, __set_literal__: NoneType):
        pass

    def __init__(
        out self, keys: IntList, values: IntList, __dict_literal__: NoneType
    ):
        pass


# CHECK-LABEL: lit.fn @"test_any_collection
def test_any_collection():
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %a){{.*}}__dict_literal__
    var a: AnyCollection = {}
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %b){{.*}}__set_literal__
    var b: AnyCollection = {1}
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %c){{.*}}__set_literal__
    var c: AnyCollection = {1, 2}
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %d){{.*}}__dict_literal__
    var d: AnyCollection = {1: 2}


# ===----------------------------------------------------------------------=== #
# Interesting unpack operations
# ===----------------------------------------------------------------------=== #


struct IntPairRange(TrivialRegisterPassable):
    def __init__(out self):
        pass

    def __len__(self) -> Int:
        pass

    def __next__(mut self) raises StopIteration -> Tuple[Int, Int]:
        pass

    def __iter__(self) -> Self:
        pass


# CHECK-LABEL: lit.fn @"test_unpack
def test_unpack():
    var elts = [a * b for (a, b) in IntPairRange()]


# COM: Check that a list comprehension can be used inside a for loop.


@fieldwise_init
struct IterRange(ImplicitlyCopyable, Iterator):
    comptime Element = Int

    var value: Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        if self.value < 0:
            raise StopIteration()
        return self.value


def useIt(expr: List[Int]):
    pass


# CHECK-LABEL: lit.fn @"comprehensionInForLoop
def comprehensionInForLoop(data: List[List[Int]], rows: Int, cols: Int) raises:
    # CHECK: lit.loop {
    # CHECK: lit.call {{.*}}::@List::@"__init__
    # CHECK: lit.loop {
    # CHECK: lit.call {{.*}}::@List::@"append
    for row in IterRange(rows):
        var my_list = [
            1 if data[row][col] == 1 else 0 for col in IterRange(cols)
        ]
        useIt(my_list)


# ===----------------------------------------------------------------------=== #
# MOCO-3801: Literal arguments to a trait-bound variadic pack
# ===----------------------------------------------------------------------=== #

# Pack inference previously bailed out with "could not infer type of parameter
# pack ... given value with unresolved type" whenever a list/dict/set literal
# was passed to a `*Ts`-bound variadic pack parameter, because initializer
# UValues don't have a resolvable RValue type until bound to a target type.
# After the fix, pack inference falls back to the literal's default type
# (Array/Dict/Set) the same way the single-argument trait-bound path does.


def _moco_3801_pack[*Ts: AnyType](var *values: *Ts):
    pass


# CHECK-LABEL: lit.fn @"test_moco_3801_pack_of_literals
def test_moco_3801_pack_of_literals():
    # COM: Each list literal defaults to Array[Int, 3] and binds its own pack
    # slot.
    # CHECK: lit.call {{.*}}@"_moco_3801_pack{{.*}}<:param_list<!AnyType> [{{.*}}@Array<:!AnyType !Int, :!Int {:scalar<index> 3}>, {{.*}}@Array<:!AnyType !Int, :!Int {:scalar<index> 3}>]
    _moco_3801_pack([1, 2, 3], [4, 5, 6])

    # COM: Dict literal defaults to Dict[Int, Int].
    # CHECK: lit.call {{.*}}@"_moco_3801_pack{{.*}}<:param_list<!AnyType> [{{.*}}@Dict<:!AnyType_Copyable_Deinitable_Movable !Int, :!AnyType_Copyable_Deinitable_Movable !Int>]
    _moco_3801_pack({1: 2})

    # COM: Set literal defaults to Set[Int].
    # CHECK: lit.call {{.*}}@"_moco_3801_pack{{.*}}<:param_list<!AnyType> [{{.*}}@Set<:!AnyType !Int>]
    _moco_3801_pack({1, 2, 3})

    # COM: Mixed-literal packs work because each pack element infers
    # independently.
    # CHECK: lit.call {{.*}}@"_moco_3801_pack{{.*}}<:param_list<!AnyType> [{{.*}}@Array<:!AnyType !Int, :!Int {:scalar<index> 2}>, {{.*}}@Set<:!AnyType !Int>]
    _moco_3801_pack([1, 2], {3, 4})


# ===----------------------------------------------------------------------=== #
# Support optional __literal_size__ parameter
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"foo()"
def foo():
    # CHECK: lit.var.decl "v" var : !lit.ref<!lit.struct<#Array <:!AnyType !Int, :!Int {:scalar<index> 3}>>
    var v: Array[_, _] = [1, 2, 3]
