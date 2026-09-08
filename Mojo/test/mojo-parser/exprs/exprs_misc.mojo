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

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s


def marker():
    pass

struct Unmovable(Movable where False):
    def __init__(out self):
        pass


def throwing_fn() raises -> Int:
    return 0


def literal_promotion[cond: Bool]():
    # This needs to coerce to the materialization type of float literal
    comptime a = 2.0 if cond else Int(3)


##===----------------------------------------------------------------------===##
# Assignment operator
##===----------------------------------------------------------------------===##


struct ListInitializable[T: AnyType](ImplicitlyCopyable):
    def __init__(
        out self, *elements: Self.T, __list_literal__: NoneType = None
    ):
        pass


struct RHSInferenceStruct(Movable where False):
    var field: ListInitializable[Int]

    def __getitem__(self) -> ListInitializable[Int]:
        pass

    def __setitem__(self, value: ListInitializable[Int]):
        pass


# None of these should be ambiguous.
def test_rhs_inference():
    var a: ListInitializable[Int]
    a = []  # DeclRedefode
    (a) = []  # ParenNode

    var lf: RHSInferenceStruct
    (lf).field = []  # AttributeRedefode

    lf[] = []  # SubscriptNode

    a, lf.field = [], []  # TupleNode


# CHECK-LABEL: lit.fn @"test_var_decl_patterns
def test_var_decl_patterns(c: Bool) raises:
    # CHECK-NEXT: lit.call {{.*}}marker
    marker()

    # Var patterns in a def are emitted inline, not at top of function.

    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: [[VAL:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
    # CHECK-NEXT: lit.ref.store [[VAL]], %x
    (var x) = 42

    # This var inside the cond is scoped correctly even though we're in a def.
    # CHECK: lit.call {{.*}}marker
    marker()

    # CHECK: hlcf.elif %{{.*}} {
    # CHECK-NEXT: [[X2:%.*]] = lit.var.decl "x"
    # CHECK-NEXT: [[VAL:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
    # CHECK-NEXT: lit.ref.store [[VAL]], [[X2]]
    if c:
        (var x) = 42

    # This must load X to add to it.
    # CHECK: kgen.rebind %x
    # CHECK: lit.call {{.*}}SIMD::@"__add__{{.*}}
    (var _) = x + 1

    # This should not load X.
    # CHECK-NOT: lit.ref.load %x
    (var _) = x

    var lf: RHSInferenceStruct
    # expected-warning @+1 {{'var' pattern didn't declare a new variable, it can be removed}}
    (var lf.field) = []

    # CHECK: lit.call {{.*}}marker
    marker()

    # CHECK that the var pattern covers both tup1 and tup2
    # CHECK: lit.var.decl "tup1" var
    # CHECK-NEXT: lit.var.decl "tup2" var
    (var tup1, var tup2) = 1, 2

    # Check that trailing commas work.
    # https://github.com/modular/modular/issues/1649
    x, x = 1, 2  # worked
    x, x, = 1, 2  # failed

    # Verify that we stop parsing at the end of the statement - we shouldn't parse
    # the "fn" as part of the list (a function expression).
    _ = (1,)

    def test() capturing:
        pass


# CHECK-LABEL: lit.fn @"test_ref_decl_patterns
def test_ref_decl_patterns(a: List[Int], mut b: List[Int]):
    # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK-NEXT: [[ASUB:%.*]] = lit.call {{.*}}List::@"__getitem__{{.*}}(%a, [[ZERO]])
    # CHECK-NEXT: %r = lit.var.decl "r" ref
    # CHECK-NEXT: lit.ref.store [[ASUB]], %r
    ref r = a[0]

    # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK-NEXT: [[BSUB:%.*]] = lit.call {{.*}}List::@"__getitem__{{.*}}(%b, [[ZERO]])
    # CHECK-NEXT: %r2 = lit.var.decl "r2" ref
    # CHECK-NEXT: lit.ref.store [[BSUB]], %r2
    ref r2 = b[0]

    # CHECK-NEXT: [[RREF:%.*]] = lit.ref.load %r
    # CHECK-NEXT: %r3 = lit.var.decl "r3" ref
    # CHECK-NEXT: lit.ref.store [[RREF]], %r3
    ref r3 = r

    # CHECK-NEXT: [[R2REF:%.*]] = lit.ref.load %r2
    # CHECK-NEXT: [[RREF:%.*]] = lit.ref.load %r
    # CHECK-NEXT: [[RVAL:%.*]] = lit.ref.load [[RREF]]
    # CHECK-NEXT: lit.call {{.*}}SIMD::@"__iadd__{{.*}}([[R2REF]], [[RVAL]])
    r2 += r

    # CHECK-NEXT: [[R2REF:%.*]] = lit.ref.load %r2
    # CHECK-NEXT: [[R3REF:%.*]] = lit.ref.load %r3
    # CHECK-NEXT: [[R3VAL:%.*]] = lit.ref.load [[R3REF]]
    # CHECK-NEXT: lit.call {{.*}}SIMD::@"__iadd__{{.*}}([[R2REF]], [[R3VAL]])
    r2 += r3

    # CHECK-NEXT: [[R2VAL:%.*]] = lit.ref.load %r2
    # CHECK-NEXT: %r4 = lit.var.decl "r4" ref
    # CHECK-NEXT: lit.ref.store [[R2VAL]], %r4
    ref r4 = r2

    # CHECK-NEXT: [[R4REF:%.*]] = lit.ref.load %r4
    # CHECK-NEXT: [[ONE:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: lit.call {{.*}}SIMD::@"__iadd__{{.*}}([[R4REF]], [[ONE]])
    r4 += 1

    # Not useful, but this should work.
    # CHECK-NEXT: %r5 = lit.var.decl "r5" ref
    ref r5: Int


# CHECK-LABEL: lit.fn @"test_type_patterns
def test_type_patterns():
    # CHECK-NEXT: lit.call {{.*}}marker
    marker()

    # CHECK: %a = lit.var.decl "a" var : !lit.ref<:meta<!Int> #alias_Int,
    (var a): Int

    # CHECK: %b = lit.var.decl "b" var : !lit.ref<!UInt8,
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.constant: {{.*}}int_literal 4>
    # CHECK-NEXT: [[TMP2:%.*]] = lit.call {{.*}}UInt8::@"__init__{{.*}}([[TMP]])
    # CHECK-NEXT: lit.ref.store [[TMP2]], %b
    var b: UInt8 = 4

    # Show that the type annotation allows us to use the type in the pattern to
    # infer the RHS type of the collection.
    # CHECK: %c = lit.var.decl "c" var : !lit.ref<!lit.struct<#List <:!AnyType_Copyable_Movable !Int>>,
    # CHECK: lit.call {{.*}}List::@"__init__
    var c: List[Int] = []

    # declare multiple variables at once.
    # CHECK: %d = lit.var.decl "d" var : !lit.ref<!Int,
    # CHECK: %e = lit.var.decl "e" var : !lit.ref<!Int,
    var (d, e): Tuple[Int, Int]


##===----------------------------------------------------------------------===##
# Test return slot optimization
##===----------------------------------------------------------------------===##


# NOTE: Don't remove this argument, this was defeating return slot opzn.
def getUnmovable(a: Unmovable) -> Unmovable:
    return Unmovable()


# This can only be codegen'd directly into x.
# CHECK-LABEL: lit.fn @"testUnmovable
def testUnmovable(a: Unmovable):
    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: lit.call {{.*}}(%a, %x)
    var x: Unmovable = getUnmovable(a)


##===----------------------------------------------------------------------===##
# type_of
##===----------------------------------------------------------------------===##

comptime _index = __mlir_type.index


# CHECK-LABEL: lit.fn @"simple_typeof_return(index)"(%x: index) -> index
def simple_typeof_return(x: _index) -> type_of(x):
    return x


# CHECK-LABEL: lit.fn @"typeof_arg(index,index)"(%x: index, %y: index) -> index
def typeof_arg(x: __mlir_type.index, y: type_of(x)) -> _index:
    var z: type_of(x) = y
    return z


# CHECK-LABEL: lit.fn @"typeof_dynval_in_param(
def typeof_dynval_in_param(x: _index):
    # CHECK-NEXT:  %y = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}String::@"__init__
    var y = String()

    # CHECK-NEXT: lit.alias.decl *"a`1": non_struct_type = <index>
    comptime a = type_of(x)
    # CHECK-NEXT: lit.alias.decl *"b`2": meta<!Int> = <#alias_Int>
    comptime b = type_of(y.__len__())

    # CHECK-NEXT: lit.alias.decl *"c`3": meta<!Int> = <#alias_Int>
    comptime c = type_of(throwing_fn())


##===----------------------------------------------------------------------===##
# origin_of
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"lifetime_of
def lifetime_of(x: Unmovable, y: Unmovable, mut z: Unmovable):
    # CHECK-NEXT: lit.alias.decl *"lt0{{.*}}:origin<false> {}
    comptime lt0 = origin_of()
    # CHECK-NEXT: lit.alias.decl *"lt1{{.*}}:origin<false> *"x`">>
    comptime lt1 = origin_of(x)
    # CHECK-NEXT: lit.alias.decl *"lt2{{.*}}:origin<false> {*"x`", *"y`1"}>>
    comptime lt2 = origin_of(x, y)
    # CHECK-NEXT: lit.alias.decl *"lt3{{.*}}:origin<true> *"z`2">>
    comptime lt3 = origin_of(z)
    # CHECK-NEXT: lit.alias.decl *"lt4{{.*}}:origin<false> {*"x`", (mutcast mut *"z`2")}>>
    comptime lt4 = origin_of(x, z)


def take_string_var(var x: String, y: String) raises:
    # Check mutable to immutable origin conversions + inference.
    imm_ref_to[origin_of(x)](x)
    imm_ref_to(x)
    imm_ref_to[origin_of(y)](y)
    imm_ref_to(y)


def imm_ref_to[origin: Origin[]](ref[origin] to: String):
    pass


##===----------------------------------------------------------------------===##
# in / not in
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"test_in
def test_in(a: String, b: String):
    # CHECK-NEXT: [[SLICE:%.*]] = lit.call {{.*}}StringSpan::@"__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call {{.*}}__contains__{{.*}}(%b, [[SLICE]])
    _ = a in b
    # CHECK: [[SLICE:%.*]] = lit.call {{.*}}StringSpan::@"__init__{{.*}}(%a)
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}__contains__{{.*}}(%b, [[SLICE]])
    # CHECK-NEXT: [[RESB:%.*]] = lit.call {{.*}}__bool__{{.*}}([[RES]])
    # CHECK-NEXT: = lit.call {{.*}}__invert__{{.*}}([[RESB]])
    _ = a not in b


##===----------------------------------------------------------------------===##
# String literals
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"test_string_literal1
def test_string_literal1(cond: Bool):
    _ = 4

    # String literals should be fine at start of expression.
    # expected-warning @+1 {{'Bool' value is unused; assign to '_' to discard the result}}
    "a" == "abc"

    # String literals should merge.
    var _ss: StaticString = "T" if cond else "F"


# Issue #1850: Mojo assumes string literal at start of a function is a doc comment
def test_expr_not_doc_string():
    # expected-warning @+1 {{'Bool' value is unused; assign to '_' to discard the result}}
    "a".__eq__("b")


##===----------------------------------------------------------------------===##
# MergeWith
##===----------------------------------------------------------------------===##


struct TypeA(TrivialRegisterPassable):
    def __merge_with__[other_type: type_of(TypeB)](self) -> TypeB:
        pass

    def __merge_with__[other_type: type_of(TypeC)](self) -> Int:
        pass


struct TypeB(TrivialRegisterPassable):
    def __merge_with__[other_type: type_of(Int)](self) -> Int:
        pass


struct TypeC(TrivialRegisterPassable):
    def __merge_with__[other_type: type_of(TypeA)](self) -> Int:
        pass

    def __merge_with__[other_type: type_of(TypeD)](self) -> TypeE:
        pass


struct TypeD(TrivialRegisterPassable):
    def __merge_with__[other_type: type_of(TypeA)](self) -> Int:
        pass


struct TypeE(TrivialRegisterPassable):
    @implicit
    def __init__(out self, other: TypeD):
        pass


# CHECK-LABEL: lit.fn @"test_mergewith
def test_mergewith(
    cond: __mlir_type.`!kgen.scalar<bool>`,
    a: TypeA,
    b: TypeB,
    c: TypeC,
    d: TypeD,
):
    # One merges to the other.
    _ = a if cond else b
    # CHECK: hlcf.elif %cond
    # CHECK-NEXT:   [[ARES:%.*]] = lit.call {{.*}}TypeA::@"__merge_with__
    # CHECK-NEXT:   hlcf.yield [[ARES]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %b
    # CHECK-NEXT: }

    # This merge with two merge_with
    _ = a if cond else c
    # CHECK: hlcf.elif %cond
    # CHECK:   [[ARES:%.*]] = lit.call {{.*}}TypeA::@"__merge_with__
    # CHECK:   hlcf.yield [[ARES]]
    # CHECK: } else {
    # CHECK:   [[CRES:%.*]] = lit.call {{.*}}TypeC::@"__merge_with__
    # CHECK:   hlcf.yield [[CRES]]
    # CHECK: }

    # One merge and one implicit conversion.
    _ = c if cond else d
    # CHECK: hlcf.elif %cond
    # CHECK:   [[CRES:%.*]] = lit.call {{.*}}TypeC::@"__merge_with__
    # CHECK:   hlcf.yield [[CRES]]
    # CHECK: } else {
    # CHECK:   [[ARES:%.*]] = lit.call {{.*}}TypeE::@"__init__
    # CHECK:   hlcf.yield [[ARES]]
    # CHECK: }

    # Infer UValues from CValues.
    # https://github.com/modular/modular/issues/5239
    _ = Int() if cond else {}
    _ = {} if cond else Int()

    # https://github.com/modular/modular/issues/5380
    # CHECK: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: [[COND:%.*]] = hlcf.elif [[FALSE]]
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   = kgen.param.constant: !Bool = <{:scalar<bool> false}>
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: }
    # CHECK: hlcf.elif %{{.*}} {
    # expected-warning @+1 {{unreachable code on right side of 'False and ...'}}
    if False and cond:
        pass


##===----------------------------------------------------------------------===##
# Chained comparisons.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"chained_cmp
def chained_cmp(a: Int, b: Int, c: Int, d: Int, e: Int):
    # CHECK:      [[CMP_A_B:%.*]] = lit.call tail @{{.*}}__lt__{{.*}}(%a, %b)
    # CHECK-NEXT: %[[CMP_A_B_SB:.*]] = lit.call tail @{{.*}}__mlir_bool__{{.*}}([[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.elif %[[CMP_A_B_SB]]
    # CHECK-NEXT:   %[[CMP_B_C:.*]] = lit.call tail @{{.*}}__lt__{{.*}}(%b, %c)
    # CHECK:        %[[IF_B_C:.*]] = hlcf.elif
    # CHECK-NEXT:     %[[CMP_C_D:.*]] = lit.call tail @{{.*}}__lt__{{.*}}(%c, %d)
    # CHECK-NEXT:     hlcf.yield %[[CMP_C_D]]
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT:   }
    # CHECK-NEXT:   hlcf.yield %[[IF_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield [[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: %res = lit.var.decl "res"
    # CHECK-NEXT: lit.ref.store %[[IF_A_B]], %res
    var res = a < b < c < d

    # COM: This checks the parsing precedence between `<` and `and`.
    # CHECK:      %[[CMP_A_B:.*]] = lit.call {{.*}}__lt__{{.*}}(%a, %b)
    # CHECK:       %[[CMP_A_B_SB:.*]] = lit.call {{.*}}__mlir_bool__{{.*}}(%[[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.elif %[[CMP_A_B_SB]]
    # CHECK:   %[[CMP_B_C:.*]] = lit.call {{.*}}__lt__{{.*}}(
    # CHECK-NEXT:   hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: %[[CMP_SB:.*]] = lit.call {{.*}}__mlir_bool__{{.*}}(%[[IF_A_B]])
    # CHECK-NEXT: %[[IF:.*]] = hlcf.elif %[[CMP_SB]]
    # CHECK-NEXT:   %[[CMP_D_E:.*]] = lit.call {{.*}}__lt__{{.*}}(%d, %e)
    # CHECK-NEXT:   hlcf.yield %[[CMP_D_E]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[IF_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ref.store %[[IF]], %res
    res = a < b < c and d < e


# Test chained comparison op in parameter domain for issue
# https://github.com/modularml/modular/issues/22050
# CHECK: lit.alias.decl *"chainedCmpAlias1{{.*}}": !Bool ={{.*}}{:scalar<bool> false}
comptime chainedCmpAlias1 = 1 == 2 == 3 == 4 == 5
# CHECK: lit.alias.decl *"chainedCmpAlias2{{.*}}": !Bool ={{.*}}{:scalar<bool> true}
comptime chainedCmpAlias2 = 1 <= 2 <= 3 <= 4 <= 5
# CHECK: lit.alias.decl *"chainedCmpAlias3{{.*}}": !Bool ={{.*}}{:scalar<bool> false}
comptime chainedCmpAlias3 = 1 <= 2 <= 9 <= 4 <= 5


# CHECK-LABEL: lit.fn @"chainedCmpSemiDyn
def chainedCmpSemiDyn(x: Int, a: Int, b: Int, c: Int):
    # CHECK-NEXT: [[IFCOND:%.*]] = kgen.param.constant: scalar<bool> = <true>
    # CHECK-NEXT: [[FINALRESULT:%.*]] = hlcf.elif [[IFCOND]] -> !Bool {
    # CHECK-NEXT:   [[PV:%.*]] = {{.*}}constant{{.*}}77
    # CHECK-NEXT:   [[CMPRESULT1:%.*]] = {{.*}}__lt__{{.*}}([[PV]], %x)
    # CHECK-NEXT:   [[IFCONDSB:%.*]] = {{.*}}__mlir_bool__{{.*}}([[CMPRESULT1]])
    # CHECK-NEXT:   [[INNERRESULT:%.*]] = hlcf.elif [[IFCONDSB]] -> !Bool {
    # CHECK-NEXT:     [[PV:%.*]] = {{.*}}constant{{.*}}105
    # CHECK-NEXT:     [[CMPRESULT2:%.*]] = {{.*}}__lt__{{.*}}(%x, [[PV]])
    # CHECK-NEXT:     [[IFCONDSB:%.*]] = {{.*}}__mlir_bool__{{.*}}([[CMPRESULT2]])
    # CHECK-NEXT:     [[MOSTINNERRESULT:%.*]] = hlcf.elif [[IFCONDSB]] -> !Bool {
    # CHECK-NEXT:       [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}{:scalar<bool> true}
    # CHECK-NEXT:       hlcf.yield [[TRUEPARAM]]
    # CHECK-NEXT:     } else {
    # CHECK-NEXT:       hlcf.yield [[CMPRESULT2]]
    # CHECK-NEXT:     }
    # CHECK-NEXT:     hlcf.yield [[MOSTINNERRESULT]]
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     hlcf.yield [[CMPRESULT1]]
    # CHECK-NEXT:   }
    # CHECK-NEXT:   hlcf.yield [[INNERRESULT]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}{:scalar<bool> true}
    # CHECK-NEXT:   hlcf.yield [[TRUEPARAM]]
    # CHECK-NEXT: }
    # CHECK: [[XCMP:%.*]] = lit.var.decl "xCmp"
    # CHECK-NEXT: lit.ref.store [[FINALRESULT]], [[XCMP]]
    var xCmp = 5 < 77 < x < 105 < 177
    # A fully deep check of this would be a lot of work, but this at least
    # shows that its not choking during parsing on a mix of dynamic and
    # parameter comparisons.  It required some care with the interaction
    # between recursive calls of emitNextCmp calls to get this to work.
    var mixedChain = 0 < 1 < a < 10 < 11 < b < 20 < 21 < c < 30 < 31


# MOCO-3608: chained comparison with `in` should not require the RHS to be
# ImplicitlyCopyable.
struct NonCopyableContainer(Movable where False):
    def __contains__(self, x: Int) -> Bool:
        return False


# CHECK-LABEL: lit.fn @"chained_cmp_in_non_copyable
def chained_cmp_in_non_copyable(a: Int, b: Int, c: NonCopyableContainer):
    # CHECK:      [[CMP_A_B:%.*]] = lit.call {{.*}}__lt__{{.*}}(%a, %b)
    # CHECK:      [[CMP_A_B_SB:%.*]] = lit.call {{.*}}__mlir_bool__{{.*}}([[CMP_A_B]])
    # CHECK-NEXT: {{%.*}} = hlcf.elif [[CMP_A_B_SB]]
    # CHECK-NEXT:   {{%.*}} = lit.call {{.*}}__contains__{{.*}}(%c,
    # CHECK:        hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield [[CMP_A_B]]
    # CHECK-NEXT: }
    _ = a < b in c


##===----------------------------------------------------------------------===##
# or/and
##===----------------------------------------------------------------------===##


# MOCO-1987: Parser error when temporary PythonObject appears in or expression
struct RPType(ImplicitlyCopyable, RegisterPassable):
    def __init__(out self):
        pass

    def __bool__(self) -> Bool:
        return Bool()


# CHECK-LABEL: lit.fn @"test_rp_and_or
def test_rp_and_or():
    # Evaluate the LHS, but materialize the rvalue into a memory slot.

    # CHECK-NEXT: [[LHS:%.*]] = lit.call {{.*}}RPType::@"__init__()
    # CHECK-NEXT: [[TMPMEM:%.*]] = lit.var.decl "anonymous
    # CHECK-NEXT: lit.ref.store [[LHS]], [[TMPMEM]]

    # CHECK-NEXT: [[IMMTMP:%.*]] = lit.ref.immut [[TMPMEM]]
    # CHECK-NEXT: lit.call {{.*}}RPType::@"__bool__{{.*}}([[IMMTMP]])
    # CHECK:      hlcf.elif
    # CHECK-NEXT:     [[LHS:%.*]] = lit.load.consume [[TMPMEM]]
    # CHECK-NEXT:     hlcf.yield [[LHS]] : !RPType

    _ = RPType() or RPType()


##===----------------------------------------------------------------------===##
# Keywords as identifiers
##===----------------------------------------------------------------------===##


struct MatchExample(Movable where False):
    def match(self):
        pass


def test_match(a: MatchExample):
    a.match()


##===----------------------------------------------------------------------===##
# if/else expression
##===----------------------------------------------------------------------===##


struct MoveOnly(Movable):
    pass


# CHECK-LABEL: lit.fn @"test_if_else_move
def test_if_else_move(r: Bool, var a: MoveOnly, var b: MoveOnly):
    # This should move a/b into t.
    var t = b^ if r else a^

    # CHECK: hlcf.elif
    # CHECK-NEXT: lit.ownership.use %b
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%b, %t){{.*}}*, "move"
    # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%a, %t){{.*}}*, "move"
    # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: }


def test_contextual_if[cond: Bool]():
    comptime some_type: Movable = Int if cond else String


##===----------------------------------------------------------------------===##
# comptime expression
##===----------------------------------------------------------------------===##


struct NotRuntimeMaterializable(Movable where False):
    def method(self) -> Int:
        pass


def test_comptime_expression[nrm: NotRuntimeMaterializable]():
    # Ok to materialize an int.
    var b = comptime (nrm.method())


##===----------------------------------------------------------------------===##
# Ellipsis.
##===----------------------------------------------------------------------===##


def test_ellipsis_overloading(a: Int):
    pass


def test_ellipsis_overloading(a: EllipsisType):
    pass


def test_ellipsis():
    var x = ...
    test_ellipsis_overloading(4)
    test_ellipsis_overloading(...)
