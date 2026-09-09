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

from std.builtin.stubs import _get_kgen_string

##===----------------------------------------------------------------------===##
# def/def
##===----------------------------------------------------------------------===##


# Method overloading.
# CHECK-LABEL: lit.fn @"testThing(::SIMD[DType.int, 1])"
def testThing(a: Int) -> FloatDyn:
    return 1.0


# CHECK-LABEL: lit.fn @"testThing(::SIMD[DType.int, 1],::SIMD[DType.int, 1])"
def testThing(a: Int, b: Int) -> Int:
    return 1


comptime IntToFloat32Type = def (:Int) thin -> FloatDyn


def takeIntToFloat32Param[f: IntToFloat32Type]():
    pass


def varargOverload(a: Int):
    pass


def varargOverload(*a: Int):
    pass


def varargOverload():
    pass


def packOverload(a: Int):
    pass


def packOverload[*Ts: AnyType](*a: *Ts):
    pass


def packOverload():
    pass


# Varargs + traits are a thing.
# https://github.com/modular/mojo/issues/1443
def variadic_trait_elt[T: ImplicitlyCopyable](*xs: T):
    pass


# CHECK-LABEL: lit.fn @"trait_pack
# CHECK-SAME: <{{.*}}, Ts:

# CHECK-SAME: %rest: !lit.ref<!lit.struct<#VariadicPack <:!Bool {:scalar<bool> false}, :origin<false> *"rest
# CHECK-SAME: :meta<!AnyType> !AnyType_Copyable_ImplicitlyCopyable_Movable, :param_list<!AnyType_Copyable_ImplicitlyCopyable_Movable> *"Ts.values`"
def trait_pack[T: ImplicitlyCopyable, *Ts: ImplicitlyCopyable](first: T, *rest: *Ts):
    pass


# CHECK-LABEL: lit.fn @"callOverload
def callOverload(a: Int):
    # CHECK: lit.call {{.*}}@"testThing({{.*}}SIMD[DType.int, 1])"(%a)
    _ = testThing(a)
    # CHECK: lit.call {{.*}}@"testThing({{.*}}SIMD[DType.int, 1],{{.*}}SIMD[DType.int, 1])"(%a, %a)
    _ = testThing(a, a)

    # CHECK: = kgen.param.constant: !alias_IntToFloat32Type1 = <rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing(::SIMD[DType.int, 1])")>
    var float1: IntToFloat32Type = testThing

    # CHECK: %3 = kgen.param.constant: !alias_IntToFloat32Type1 = <rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing(::SIMD[DType.int, 1])")>
    # CHECK-NEXT: lit.ref.store %3, %float1
    float1 = testThing

    # CHECK: %4 = kgen.param.constant: !alias_IntToFloat32Type1 = <rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing(::SIMD[DType.int, 1])")>
    var float2: IntToFloat32Type = testThing

    # CHECK: lit.call {{.*}}@"takeIntToFloat32Param[def({{.*}}SIMD[DType.int, 1]) thin -> {{.*}}FloatDyn]()"<:
    # CHECK-SAME: !alias_IntToFloat32Type1 rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing(::SIMD[DType.int, 1])")>()
    takeIntToFloat32Param[testThing]()

    # Issue #10036.  This should call the exact match, consider the varargs match
    # less specific.
    # CHECK: lit.call {{.*}}@"varargOverload({{.*}}SIMD[DType.int, 1])"(%{{.*}})
    varargOverload(2)

    # CHECK:  lit.call {{.*}}@"varargOverload()"()
    varargOverload()

    # Expect packs to behave similarly to varargs.
    # CHECK: %[[IDX3:.*]] = {{.*}}constant{{.*}}3
    # CHECK: lit.call {{.*}}@"packOverload({{.*}}SIMD[DType.int, 1])"(%[[IDX3]])
    packOverload(3)
    # CHECK:  lit.call {{.*}}@"packOverload()"()
    packOverload()

    # CHECK: call {{.*}}trait_pack
    # CHECK-SAME: [!Int, !Int]
    trait_pack(1, 2, 3)


struct MyInt(TrivialRegisterPassable):
    var value: Int

    @implicit
    @always_inline("nodebug")
    @implicit
    def __init__(out self, _a: Int):
        self.value = _a

def paramOverload[*x: Int]():
    pass


def paramOverload(y: Int):
    pass


def paramOverload[x: Int, T: TrivialRegisterPassable](y: T):
    pass


def paramOverload[*x: Int](y: Int):
    pass


# CHECK-LABEL: lit.fn @"callParametricOverload
def callParametricOverload[a: Int, b: Int, c: Int](x: Int):
    # CHECK-NEXT: lit.call {{.*}}@"paramOverload[{{.*}}SIMD[DType.int, 1]]()"
    paramOverload[a]()

    # CHECK-NEXT: lit.call {{.*}}@"paramOverload{{.*}}<:param_list<!Int> [a, b, c]
    paramOverload[a, b, c]()

    # CHECK-NEXT: lit.call {{.*}}@"paramOverload({{.*}}SIMD[DType.int, 1])"
    paramOverload(x)

    # CHECK-NEXT: lit.call {{.*}}@"paramOverload{{.*}}<:param_list<!Int> [a, b], {{.*}}>(%x)
    paramOverload[a, b](x)

struct VariadicStruct[*Ts: TrivialRegisterPassable](Movable where False):
    def __init__(out self):
        pass

    @staticmethod
    def param_func[i: Int]():
        pass


def take_variadic_struct[*Ts: TrivialRegisterPassable](a: VariadicStruct[*Ts]):
    pass


# CHECK-LABEL: lit.fn @"variadic_params()"
def variadic_params():
    # CHECK-NEXT: call {{.*}}param_func[{{.*}}SIMD[DType.int, 1]]()"<:param_list<!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable> [!Int, !FloatDyn], {{.*}}, :!Int {:scalar<index> 4}>
    VariadicStruct[Int, FloatDyn].param_func[4]()
    # CHECK: call {{.*}}take_variadic_struct{{.*}}<:param_list<!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable> [!Int, !FloatDyn],
    take_variadic_struct(VariadicStruct[Int, FloatDyn]())



@always_inline("nodebug")
def returnParameter[a: __mlir_type.index]() -> __mlir_type.index:
    return a


# CHECK-LABEL: lit.fn @"callReturnParam
def callReturnParam() -> __mlir_type.index:
    # CHECK-NEXT: %0 = lit.call {{.*}}@"returnParameter[index]()"<3>()
    # CHECK-NEXT: return %0
    return returnParameter[Int(3).__mlir_index__()]()


def paramRefFunc[T: TrivialRegisterPassable](x: T):
    pass


# CHECK-LABEL: lit.fn @"orvalueInferType()"
def orvalueInferType():
    def func(x: __mlir_type.index) -> __mlir_type.index:
        return x

    # CHECK: call {{.*}}paramRefFunc{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable{{.*}}!lit.generator<("x": index) -> index>>
    paramRefFunc(func)




# https://github.com/modular/mojo/issues/1152
# Allow mutable self argument when overloading operators using dunder methods
struct MutatingAdd(Movable where False):
    def __add__(mut self, x: MutatingAdd):
        pass


# CHECK-LABEL: lit.fn @"testMutatingAdd
def testMutatingAdd(var a: MutatingAdd, b: MutatingAdd):
    # CHECK-NEXT: lit.call {{.*}}__add__{{.*}}(%a, %b)
    a + b


# CHECK-LABEL: lit.fn @"testContextSensitiveKeyword
# CHECK-SAME: (%out2: !Int) -> !alias_Int1
def testContextSensitiveKeyword(out x: Int, out2: Int):
    # Check that we handle the result slot correctly.

    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: [[RB:%.*]] = kgen.rebind %out2
    # CHECK-NEXT: lit.ref.store [[RB]], %x
    # CHECK-NEXT: [[LD:%.*]] = lit.load.consume %x
    # CHECK-NEXT: lit.return [[LD]]

    # out is an argument specifier, but that's a context sensitive keyword.
    # The identifier can be used like normal as well.
    x = out2


##===----------------------------------------------------------------------===##
# Conventions
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"ownedConventionMem
# CHECK-SAME: (%a: !lit.ref<!StructWithInit, mut {{.*}}> owned_in_mem,
# CHECK-SAME:  %b: !lit.ref<!StructWithInit, imm {{.*}}> imm_mem)
def ownedConventionMem(var a: StructWithInit, b: StructWithInit):
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK: [[AXR:%.*]] = kgen.rebind [[AX]]
    # CHECK: = lit.ref.load [[AXR]]
    _ = a.x+1
    # CHECK: [[BY:%.*]] = lit.ref.struct.ger %b[y]
    # CHECK: [[BYR:%.*]] = kgen.rebind [[BY]]
    # CHECK: = lit.ref.load [[BYR]]
    _ = b.y+1

    # It is ok to mutate owned values.
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}4
    # CHECK-NEXT: lit.ref.store [[FOUR]], [[AX]]
    a.x = 4


struct RPStructWithInit(RegisterPassable):
    var x: Int
    var y: Int


struct RPStructWithInitTrivial(TrivialRegisterPassable):
    var x: __mlir_type.index


# CHECK-LABEL: lit.fn @"ownedConventionReg
# CHECK-SAME: (%a: !lit.ref<!RPStructWithInit, mut *"a`"> owned_in_mem,
# CHECK-SAME:  %b: !lit.ref<!RPStructWithInit, imm *"b`1"> imm_mem,
# CHECK-SAME:  %triv: !RPStructWithInitTrivial)
def ownedConventionReg(
    var a: RPStructWithInit,
    b: RPStructWithInit,
    triv: RPStructWithInitTrivial,
):
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK: [[AXR:%.*]] = kgen.rebind [[AX]]
    # CHECK:  = lit.ref.load [[AXR]]
    _ = a.x+1
    # CHECK: [[BY:%.*]] = lit.ref.struct.ger %b[y]
    # CHECK: [[BYR:%.*]] = kgen.rebind [[BY]]
    # CHECK:  = lit.ref.load [[BYR]]
    _ = b.y+1

    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK: [[ONE:%.*]]  = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
    # CHECK: lit.ref.store [[ONE]], [[AX]]
    a.x = 1


struct BorrowStruct(Movable where False):
    def testMethod(self):
        pass

    def borrowedVarArgs(self, *x: BorrowStruct):
        pass


# CHECK-LABEL: callerFn
# CHECK-SAME: (%arg0: !lit.ref<{{.*}}> imm_mem)
def callerFn(arg0: BorrowStruct):
    # CHECK-NEXT: lit.call {{.*}}testMethod{{.*}}(%arg0)
    arg0.testMethod()

    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [%arg0, %arg0]
    # CHECK: lit.call {{.*}}borrowedVarArgs{{.*}}(%arg0,
    arg0.borrowedVarArgs(arg0, arg0)


##===----------------------------------------------------------------------===##
# Named Results
##===----------------------------------------------------------------------===##


struct SomeResultType(Movable where False):
    def __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"named_result
# CHECK-SAME: %out: !lit.ref<!SomeResultType, {{.*}}> byref_result
# CHECK-SAME: namedResult = "out"
def named_result(out out: SomeResultType):
    # CHECK-NEXT: call {{.*}}SomeResultType::@"__init__{{.*}}(%out)
    out = SomeResultType()
    # CHECK: lit.return %none
    return
    # CHECK-NEXT: lit.end_fn


# CHECK-LABEL: lit.fn @"named_result_return_expr
def named_result_return_expr(out out: SomeResultType):
    # CHECK-NEXT: call {{.*}}SomeResultType::@"__init__{{.*}}(%out)
    return SomeResultType()


##===----------------------------------------------------------------------===##
# Default arguments and variadics.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"defaultArgument
# CHECK-SAME: %c: !Int = {:scalar<index> 5})
def defaultArgument(a: Int, b: Int = 3, c: Int = 5) -> Int:
    return a + b


# CHECK-LABEL: lit.fn @"callDefaultArgument
def callDefaultArgument(x: Int) -> Int:
    # CHECK: [[ARG1:%.*]] = kgen.param.constant{{.*}}3
    # CHECK-NEXT: [[ARG2:%.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: lit.call {{.*}}defaultArgument{{.*}}(%x, [[ARG1]], [[ARG2]])
    # CHECK-NEXT: %a = lit.var.decl "a"
    # CHECK-NEXT: lit.ref.store {{.*}}, %a
    var a = defaultArgument(x)

    # CHECK-NEXT: %[[ARG2:.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: lit.call {{.*}}defaultArgument{{.*}}(%x, %x, %[[ARG2]])
    var b = defaultArgument(x, x)
    return a + b


# CHECK-LABEL: lit.fn @"defaultArgumentReferencesParameter
# CHECK: scalar<index> = add(#lit.struct.extract<:!Int p, "_mlir_value">, 87)
def defaultArgumentReferencesParameter[p: Int](a: Int = p + 87) -> Int:
    return a


struct MemoryType(Movable where False):
    var value: Int

    @implicit
    def __init__(out self, value: Int):
        self.value = value

    # MOCO-1445: throwing implicit conversions.
    @implicit
    def __init__(out self, value: String) raises:
        self.value = 4

    # Default arguments and variadics.
    @implicit
    def __init__(
        out self, value: SomeResultType, stuff: Int = 4, *other: String
    ):
        self.value = 4


# CHECK-LABEL: lit.fn @"defaultArgumentNonRegisterType
# CHECK-SAME: imm_mem = apply_result_slot({{.*}}__init__
def defaultArgumentNonRegisterType(a: MemoryType = 1):
    pass


# CHECK-LABEL: lit.fn @"callNonRegisterDefaultArg
def callNonRegisterDefaultArg():
    # CHECK: %[[ANON:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!MemoryType, mut *"anonymous*`">
    # CHECK: %[[VALUE:.*]] = kgen.param.materialize: !MemoryType = <apply_result_slot({{.*}}1}
    # CHECK: lit.ref.store %[[VALUE]], %[[ANON]]
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %anonymous2A
    # CHECK: call {{.*}}defaultArgumentNonRegisterType{{.*}}([[IMMREF]])
    defaultArgumentNonRegisterType()
    # CHECK: lit.alias.decl *"none{{.*}}": none = <apply({{.*}}defaultArgumentNonRegisterType
    # CHECK-SAME: store_to_mem(apply_result_slot({{.*}}MemoryType::@"__init__{{.*}}1}
    comptime none = defaultArgumentNonRegisterType()


# CHECK: lit.fn @"referencesDefaultArgumentFunction
def referencesDefaultArgumentFunction():
    # CHECK: %f = lit.var.decl "f"
    # CHECK: lit.ref.store %0, %f
    var f = defaultArgument


# CHECK-LABEL: lit.struct.decl @Outer<X:
struct Outer[X: Int](Movable where False):
    # CHECK: lit.fn @"nested
    # CHECK-SAME: %x: !Int = X)
    def nested(self, x: Int = Self.X):
        pass


# CHECK-LABEL: lit.fn @"variadics{{.*}}SIMD[DType.int, 1]*)"{{.*}}(%a: !lit.ref<!lit.struct<#VariadicList{{.*}}> imm_mem|pos_vararg)
def variadics(*a: Int):
    # CHECK-NEXT: %none = kgen.param.constant
    pass


def parameterizedVariadic[T: TrivialRegisterPassable](*args: T):
    pass


struct ParameterizedStruct[T: TrivialRegisterPassable](Movable where False):
    @implicit
    def __init__(out self, *args: Self.T):
        pass


struct VarArgsParameterizedStruct[*Is: Int](Movable where False):
    def __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"callVariadic{{.*}}"<p: !Int>
def callVariadic[p: Int](x: Int):
    # CHECK: [[VARIADIC:%.*]] = kgen.param.constant: !lit.ref<array<0, !lit.ref<!Int, imm {}>>, imm {}> = <#interp.pointer<0>>
    # CHECK-NEXT: [[T1:%.*]] = lit.call {{.*}}VariadicList::@"__init__{{.*}}([[VARIADIC]])
    # CHECK-NEXT: [[TMPVD:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[T1]], [[TMPVD]]
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.immut [[TMPVD]]
    # CHECK: lit.call {{.*}}@"variadics{{.*}}SIMD[DType.int, 1]*)"{{.*}}([[T2]])
    variadics()
    # CHECK: [[C7:%.*]] = kgen.param.constant{{.*}}7
    # CHECK: [[C11:%.*]] = kgen.param.constant{{.*}}11
    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [{{.*}}]
    # CHECK: [[T1:%.*]] = lit.call {{.*}}VariadicList::@"__init__
    # CHECK-NEXT: [[TMPVD:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[T1]], [[TMPVD]]
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.immut [[TMPVD]]
    # CHECK: lit.call {{.*}}@"variadics{{.*}}SIMD[DType.int, 1]*)"{{.*}}([[T2]])
    variadics(7, 11)
    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [%{{.*}}]
    # CHECK: [[T1:%.*]] = lit.call {{.*}}VariadicList::@"__init__
    # CHECK-NEXT: [[TMPVD:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[T1]], [[TMPVD]]
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.immut [[TMPVD]]
    # CHECK: lit.call {{.*}}@"variadics{{.*}}SIMD[DType.int, 1]*)"{{.*}}([[T2]])
    variadics(x)
    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [{{.*}}]
    # CHECK: [[T1:%.*]] = lit.call {{.*}}VariadicList::@"__init__
    # CHECK-NEXT: [[TMPVD:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[T1]], [[TMPVD]]
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.immut [[TMPVD]]
    # CHECK-NEXT: lit.call {{.*}}@"variadics{{.*}}SIMD[DType.int, 1]*)"{{.*}}([[T2]])
    variadics(x, 1)

    # CHECK: lit.alias.decl *"EmptyVariadic
    # CHECK-SAME: "a": !lit.ref<!lit.struct<#VariadicList{{.*}}, #interp.pointer<0>)))
    comptime EmptyVariadic = variadics()
    # CHECK: lit.alias.decl *"NonEmptyVariadic
    # CHECK-SAME: @"variadics{{.*}}SIMD[DType.int, 1]*)"{{.*}}[store_to_mem(p), store_to_mem({:scalar<index> 1})]
    comptime NonEmptyVariadic = variadics(p, 1)

    # CHECK: lit.call {{.*}}parameterizedVariadic{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int
    parameterizedVariadic(1, 2)
    # CHECK: lit.call {{.*}}@ParameterizedStruct::@"__init__{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int
    _ = ParameterizedStruct(3)
    # CHECK: lit.call {{.*}}@VarArgsParameterizedStruct::@"__init__{{.*}}<:param_list<!Int> [{:scalar<index> 4}, {:scalar<index> 5}]
    _ = VarArgsParameterizedStruct[4, 5]()
    # CHECK: lit.call {{.*}}@VarArgsParameterizedStruct::@"__init__{{.*}}<:param_list<!Int> []
    _ = VarArgsParameterizedStruct()


# COM: Test variadic arguments in a parameter context.
@fieldwise_init
struct MemStruct(Movable where False):
    comptime t = 5

def variadic_mem_only(*values: MemStruct) -> Int:
    return 0

# CHECK-LABEL: lit.fn @"test_variadic_mem_only{{.*}}"<x: !MemStruct, y: !MemStruct>
def test_variadic_mem_only[x: MemStruct, y: MemStruct]():
    # CHECK: lit.alias.decl {{.*}}: !alias_Int1 = <apply(
    # CHECK-SAME: :!lit.generator<[1]("values": {{.*}}#VariadicList{{.*}}:!AnyType !MemStruct, :!Bool {:scalar<bool> false}>>, imm #lit.comptime.origin> imm_mem|pos_vararg) -> !alias_Int1> {{.*}}::@"variadic_mem_only{{.*}}::MemStruct*)"
    # CHECK-SAME: [store_to_mem(x), store_to_mem(y)])
    comptime b = variadic_mem_only(x, y)


##===----------------------------------------------------------------------===##
# raises specifier.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"defAlwaysRaises()"[{{.*}}](?, %__error__: {{.*}}, %__result__: {{.*}}) throws -> !kgen.scalar<bool> attributes {sourceName = "defAlwaysRaises"
def defAlwaysRaises() raises -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}{:scalar<index> 0}
    # CHECK: lit.ref.store [[RESULT]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return [[FALSE]]
    return 0

# CHECK-LABEL: lit.fn @"defCanAlsoRaise()"{{.*}} throws -> !kgen.scalar<bool>
def defCanAlsoRaise() raises Int -> Int:
    pass




# CHECK-LABEL: lit.fn @"fnThatRaises()"{{.*}} throws -> !kgen.scalar<bool>
def fnThatRaises() raises -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}{:scalar<index> 0}
    # CHECK-NEXT: lit.ref.store [[RESULT]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return [[FALSE]]
    return 0


# CHECK-LABEL: lit.fn @"raisesReturnsNone()"{{.*}} throws -> !kgen.scalar<bool>
def raisesReturnsNone() raises:
    # CHECK-NEXT: %none = kgen.param.constant: none
    # CHECK-NEXT: lit.ref.store %none, %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return [[FALSE]]
    # CHECK-NEXT: lit.end_fn
    pass


# COM: We can return an variant of error and index in a non-throwing function.
# CHECK-LABEL: lit.fn @"raisesReturnsVariant()"() -> !kgen.variant<!Error, index>
def raisesReturnsVariant() -> __mlir_type[`!kgen.variant<`, Error, `, index>`]:
    return __mlir_op.`kgen.variant.create`[
        _type = __mlir_type[`!kgen.variant<`, Error, `, index>`],
        index = Int(1).__mlir_index__(),
    ](Int(1).__mlir_index__())


# CHECK-LABEL: lit.fn @"raise_and_return{{.*}} throws -> !kgen.scalar<bool>
def raise_and_return(a: Error) raises -> Error:
    # COM: True result indicates an error.
    # CHECK: [[ERR:%.*]] = lit.call {{.*}}Error::@"__init__{{.*}}(%__result__)
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return [[FALSE]]
    return Error()


@fieldwise_init
struct RaisingGetterSetter(TrivialRegisterPassable):
    def __getitem__(self, i: Int) raises -> FloatDyn:
        return 1.0

    def __setitem__(mut self, i: Int, v: FloatDyn) raises:
        pass


def test_raising_computed_getter() raises:
    var a = RaisingGetterSetter()[2]

# CHECK-LABEL: lit.fn @"test_typed_raises_fn1
# CHECK-SAME: %__error__: !lit.ref<:meta<!Int> #alias_Int, mut *"__error__`"> byref_error
# CHECK-SAME: ) throws -> !kgen.scalar<bool>
def test_typed_raises_fn1() raises Int -> String:
    pass

def test_typed_raises_fn2() raises (Int) -> String:
    pass

# CHECK-LABEL: lit.fn @"test_typed_raises_fn3
def test_typed_raises_fn3() raises Float32:
    # This tests calling a function that raises Int in context that requires
    # raising Float32.  We need a temporary for the implicit conversion.

    # CHECK-NEXT: %anonymous2A = lit.var.decl
    # CHECK-NEXT: %__call_error_tmp__ = lit.var.decl{{.*}}!lit.ref<:meta<!Int>
    # CHECK-NEXT: lit.try %__call_error_tmp__
    # CHECK-NEXT: lit.call {{.*}}test_typed_raises_fn2{{.*}}(%__call_error_tmp__, %anonymous2A)
    # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT:    [[ERB:%.*]] = kgen.rebind %__call_error_tmp__
    # CHECK-NEXT:    = lit.ref.load [[ERB]]
    # CHECK-NEXT:    lit.call {{.*}}SIMD::@"__init__
    # CHECK-NEXT:    = kgen.rebind
    # CHECK-NEXT:    lit.ref.store {{.*}}, %__error__
    # CHECK-NEXT:    lit.raise
    _ = test_typed_raises_fn2()

# CHECK-LABEL: lit.fn @"test_typed_raises_fn4
def test_typed_raises_fn4() raises Float32:
    # CHECK: %anonymous2A = lit.var.decl
    # CHECK-NEXT: %__call_error_tmp__ = lit.var.decl
    # CHECK-NEXT: lit.try %__call_error_tmp__
    # CHECK-NEXT:   lit.call {{.*}}test_typed_raises_fn2(){{.*}}(%__call_error_tmp__, %anonymous2A)

    # Make sure to emit ExprDest outside the try block.
    # CHECK: %str = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}String::@"__init__{{.*}}"{{.*}}(%anonymous2A, %str){{.*}}*, "move"
    var str = test_typed_raises_fn2()
    _ = str.__len__()

# CHECK-LABEL: lit.fn @"test_typed_raises_fn5
def test_typed_raises_fn5() raises Float32:

    # Make sure to emit ExprDest outside the try block.
    # CHECK: %anonymous2A = lit.var.decl
    # CHECK-NEXT: %__call_error_tmp__
    # CHECK-NEXT: lit.try
    _ = test_typed_raises_fn2().__len__()

def test_typed_raises_fn3(x: String) raises Int -> ref [x] String:
    return x


# CHECK-LABEL: lit.fn @"test_typed_raises_fn6
def test_typed_raises_fn6(x: String) raises Float32:

    # Make sure to emit ExprDest outside the try block.
    # CHECK: %__ref_result_tmp__ = lit.var.decl
    # CHECK-NEXT: %__call_error_tmp__
    # CHECK-NEXT: lit.try
    _ = test_typed_raises_fn3(x).__len__()

# https://github.com/modular/modular/issues/5845
def callee_7(x: String) raises Int -> Pointer[String, origin_of(x)]:
    raise 4

def test_typed_raises_fn7() raises Float32:
    var str: String
    var tmp = callee_7(str)

# CHECK-LABEL: lit.fn @"call_test_typed_raises_fn
def call_test_typed_raises_fn() raises Int:
    # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}@"test_typed_raises_fn1{{.*}}(%__error__, %__call_result_tmp__)
    _ = test_typed_raises_fn1()

    # CHECK: %test_type_of = lit.var.decl {{.*}} : !lit.ref<!String,
    var test_type_of : type_of(test_typed_raises_fn1())


# CHECK-LABEL: lit.fn @"test_typed_raises_fn8
# CHECK-SAME: %__error__: !lit.ref<:meta<!Int> #alias_Int, mut *"__error__`"> byref_error
# CHECK-SAME: ) throws -> !kgen.scalar<bool>
def test_typed_raises_fn8() raises Int -> String:
    pass

def parametric_raise_example[ErrorType: AnyType](fp: def () thin raises ErrorType) raises ErrorType:
    fp()

# Test that parametric types raise the correct concrete error type.
# CHECK-LABEL: lit.fn @"call_parametric_raise_example
def call_parametric_raise_example[GenTy: AnyType](func_ptr: def () thin raises GenTy):
    # CHECK:  lit.var.decl "err_int" var : !lit.ref<!Int,
    def raise_int() raises Int: pass
    try:
        parametric_raise_example(raise_int)
    except err_int:
        ref x: Int = err_int # Test no error.

    # CHECK: lit.var.decl "err_string" var : !lit.ref<!String,
    def raise_string() raises String: pass
    try:
      parametric_raise_example(raise_string)
    except err_string:
        ref s: String = err_string # Test no error.

    # CHECK: lit.var.decl "err_gen" var : !lit.ref<:!AnyType GenTy,
    try:
      parametric_raise_example(func_ptr)
    except err_gen:
        ref s: GenTy = err_gen # Test no error.

    # These are not in a try block.

    def raise_never() raises Never: pass
    parametric_raise_example(raise_never)

    def doesnt_raise(): pass
    parametric_raise_example(doesnt_raise)


# A parametric-raises trait method may name its thrown type via
# `Self.<AssociatedType>`.
trait HasAssociatedErrorType:
    comptime E: AnyType

    def do_it(self) raises Self.E:
        ...


struct CutDownDict[V: Copyable](Movable where False):
    # Throws an error that converts to Error. We don't care what it is, so long
    # as it isn't Error specifically.
    def __getitem__(ref self) raises StringLiteral["foo".value] -> ref [self] Self.V:
        pass

def test_cut_down_dict() raises:
    var dict: CutDownDict[Int]
    var ptr = Pointer(to=dict[])
    ptr[] = 17 # should be mutable.

# MOCO-2997: Remapping of implicit origins into typed errors.
def callee(a: String) raises Pointer[String, origin_of(a)]:
    raise Pointer(to=a)

def caller(a: String) raises Pointer[String, origin_of(a)]:
    callee(a)

##===----------------------------------------------------------------------===##
# Constraint Overloading
##===----------------------------------------------------------------------===##

# CHECK: lit.fn @"int_abs[[INT_ABS_NONNEG:[^"]+]]"
def int_abs[x: Int]() -> Int
    where x > -1:
    return x

# CHECK: lit.fn @"int_abs[[INT_ABS_NEG:[^"]+]]"
def int_abs[x: Int]() -> Int
    where x < 0:
    return 0 - x


# CHECK: lit.fn @"constraint_overloading
def constraint_overloading():
    # CHECK: lit.call {{.*}}@"int_abs[[INT_ABS_NONNEG]]"
    _ = int_abs[1]()
    # CHECK: lit.call {{.*}}@"int_abs[[INT_ABS_NEG]]"
    _ = int_abs[-1]()

##===----------------------------------------------------------------------===##
# Structs
##===----------------------------------------------------------------------===##


def forward_ref(x: EmptyStruct):
    pass


# CHECK-LABEL: lit.struct.decl @EmptyStruct({{.*}})
struct EmptyStruct(RegisterPassable):
    pass


# CHECK-LABEL: lit.struct.decl @OneLineStruct<size: !Int>
struct OneLineStruct[size: Int](Movable where False):
    pass
    pass


# CHECK-LABEL: lit.struct.decl @StructWithInit
struct StructWithInit(Movable where False):
    var x: Int
    var y: Int

    # CHECK: lit.fn @"__init__({{.*}}SIMD[DType.int, 1])"
    # CHECK-SAME: %self: !lit.ref<!StructWithInit, mut {{.*}}> byref_result)
    @implicit
    def __init__(out self, a: Int):
        # CHECK: %0 = lit.ref.struct.ger %self[x]
        # CHECK: [[AR:%.*]] = kgen.rebind %a
        # CHECK: lit.ref.store [[AR]], %0
        self.x = a
        # CHECK: [[YP:%.*]] = lit.ref.struct.ger %self[y]
        # CHECK: [[XP:%.*]] = lit.ref.struct.ger %self[x]
        # CHECK: [[XT:%.*]] = lit.ref.load [[XP]]
        # CHECK: lit.ref.store [[XT]], [[YP]]
        self.y = self.x
        # CHECK-NEXT: kgen.param.constant: none
        # CHECK-NEXT: lit.return
        return

    # Not very useful, but this form also works, so test it.
    # CHECK: lit.fn @"__init__
    # CHECK-SAME: %self: !lit.ref<!StructWithInit, mut {{.*}}> byref_result)
    def __init__(out self, a: Int, b: Int):
        # CHECK: hlcf.elif
        if a == b:
            # CHECK:  lit.call {{.*}}__init__{{.*}}(%a, %self)
            self = StructWithInit(a)
        else:
            # CHECK: [[XP:%.*]] = lit.ref.struct.ger %self[x]
            # CHECK: [[AR:%.*]] = kgen.rebind %a
            # CHECK: lit.ref.store [[AR]], [[XP]]
            # CHECK: [[YP:%.*]] = lit.ref.struct.ger %self[y]
            # CHECK: [[BR:%.*]] = kgen.rebind %b
            # CHECK: lit.ref.store [[BR]], [[YP]]
            self.x = a
            self.y = b

    def __init__(out self):
        self = Self(0)


# CHECK-LABEL: lit.struct.decl @StructExample
struct StructExample(ImplicitlyCopyable, RegisterPassable):
    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self):
        pass

    # CHECK: lit.fn @"maybe_static({{.*}}SIMD[DType.int, 1])"(%x: !Int) {{.*}}isStatic
    @staticmethod
    def maybe_static(x: Int):
        # CHECK: %0 = {{.*}}{:scalar<index> 4}
        # CHECK: lit.call {{.*}}@StructExample::@"maybe_static{{.*}}"(%0)
        StructExample.maybe_static(4)
        pass

    # This isn't static.
    # CHECK: lit.fn @"maybe_static
    def maybe_static(self, x: EmptyStruct):
        # CHECK: %0 = {{.*}}{:scalar<index> 4}
        # CHECK: lit.call {{.*}}@StructExample::@"maybe_static{{.*}}"(%0)
        StructExample.maybe_static(4)
        pass

    # CHECK: lit.fn @"mutatingMethod{{.*}}(%self: !lit.ref<!StructExample, mut {{.*}}> mut) -> !kgen.none
    def mutatingMethod(mut self):
        pass


# CHECK-LABEL: lit.fn @"callMaybeStatic
def callMaybeStatic(a: Int, b: EmptyStruct):
    # CHECK-NEXT: lit.call {{.*}}@StructExample::@"maybe_static{{.*}}(%a)
    StructExample.maybe_static(a)

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@StructExample::@"__init__{{.*}}()
    # CHECK-NEXT: [[ANONSE:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[TMP]], [[ANONSE]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[ANONSE]]
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}([[TMP]], %b)
    StructExample.maybe_static(StructExample(), b)

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@"__init__{{.*}}()
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}(%a)
    StructExample().maybe_static(a)

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@StructExample::@"__init__{{.*}}()
    # CHECK-NEXT: [[ANONSE:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[TMP]], [[ANONSE]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[ANONSE]]
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}([[TMP]], %b)
    StructExample().maybe_static(b)


# CHECK-LABEL: lit.fn @"initializersAsFunctions
# See that we can take the address of initializers without a thunk.
def initializersAsFunctions():
    # Register passable trivial.
    # CHECK-NEXT: %def_ptr1 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure[{{.*}}:!lit.generator<("_a": !Int) -> !MyInt> @decls::@MyInt::@"__init__(::SIMD[DType.int, 1])")]()
    # CHECK-NEXT: lit.ref.store [[TMP]], %def_ptr1
    var def_ptr1: def (:Int) thin -> MyInt = MyInt.__init__

    # Register passable non-trivial.

    # CHECK-NEXT: %def_ptr2 = lit.var.decl "def_ptr2"
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure[!lit.generator<() -> !StructExample>: @decls::@StructExample::@"__init__()"]()
    # CHECK-NEXT: lit.ref.store [[TMP]], %def_ptr2
    var def_ptr2: def () thin -> StructExample = StructExample.__init__

    # CHECK-NEXT: %def_ptr4 = lit.var.decl "def_ptr4"
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure{{.*}}@StructExample::@"__init__(copy:{{.*}}"]()
    # CHECK-NEXT: lit.ref.store [[TMP]], %def_ptr4
    var def_ptr4: def (
        *, copy: StructExample
    ) thin -> StructExample = StructExample.__init__

    # Memory
    # CHECK-NEXT: %def_ptr5 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure[{{.*}}:!lit.generator<[1]("a": !Int, ?, "self": !lit.ref<!StructWithInit, mut *[0,0]> byref_result) -> !kgen.none> @decls::@StructWithInit::@"__init__(::SIMD[DType.int, 1])")
    # CHECK-NEXT: lit.ref.store [[TMP]], %def_ptr5
    var def_ptr5: def (Int) thin -> StructWithInit = StructWithInit.__init__


# CHECK-LABEL: lit.struct.decl @DelegatingInitMem
# Issue #12042
struct DelegatingInitMem(Movable where False):
    var value: Int

    # CHECK: lit.fn @"__init__{{.*}}({{.*}}%self
    @implicit
    def __init__(out self, value: Bool):
        # CHECK: lit.call {{.*}}__init__{{.*}}(%0, %self)
        self = Self(42)

    @implicit
    def __init__(out self, value: Int):
        self.value = value


# External issue #260
def nameOutsideStruct(x: Int, y: Int):
    pass


struct ShadowsOuterName(Movable where False):
    def nameOutsideStruct(self):
        nameOutsideStruct(1, 2)


struct LegacyInOutInit(Movable where False):
    # This should be accepted for compatibility, but "out" is the preferred
    # spelling.
    def __init__(out self):
        pass

##===----------------------------------------------------------------------===##
# async/await
##===----------------------------------------------------------------------===##


struct Container[T: AnyType](TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type[`!kgen.pointer<`, Self.T, `>`]
    var address: Self._mlir_type

    def __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]


async def load(server_ptr: Container[__mlir_type.index]):
    pass


# CHECK-LABEL: lit.fn @"awaitSomething()"
async def awaitSomething():
    var ptr = Container[__mlir_type.index]()
    # CHECK: [[CORO:%.*]] = lit.call {{.*}}@Coroutine::@"__init__{{.*}}<:!AnyType [{{.*}}], :origin.set {}>(%{{.*}}) :
    # CHECK-SAME: !lit.generator<("handle": !alias_AnyCoroutine1) -> !lit.struct<#Coroutine <:!AnyType
    await load(ptr)


# CHECK-LABEL: lit.fn @"coroutine
# CHECK-SAME: [mut [[LT:.*]]](?, %__result__: !lit.ref<:meta<!Int> #alias_Int, mut [[LT]]> byref_result) async -> !kgen.none
async def coroutine() -> Int:
    # CHECK: lit.ref.store %0, %__result__
    # CHECK: lit.return %none
    return 0


# CHECK-LABEL: lit.struct.decl @StructWithAsync
struct StructWithAsync(Movable where False):
    # CHECK-LABEL: lit.fn @"do_something{{.*}}({{.*}}) async
    async def do_something(self: StructWithAsync):
        # CHECK-NEXT: [[CORO:%.*]] = lit.async.call[!lit.generator<[1](?, "__result__": !lit.ref<:meta<!Int> #alias_Int, mut *[0,0]> byref_result) async -> !kgen.none>: @decls::@"coroutine()"][imm {}]()
        # CHECK-NEXT: %1 = kgen.rebind [[CORO]] : !co.routine to !alias_AnyCoroutine1
        # CHECK: lit.call {{.*}}@Coroutine::@"__init__{{.*}}<:!AnyType !Int, :origin.set {}>(%1)
        _ = coroutine()


# CHECK-LABEL: lit.fn @"call_struct_async
# CHECK-SAME: [imm [[LT:.*]], mut {{.*}}]{{.*}}) async -> !kgen.none
async def call_struct_async(f: StructWithAsync):
    # CHECK-NEXT: lit.async.call[!lit.generator<[2]({{.*}}, "__result__":{{.*}}) async -> !kgen.none>: @{{.*}}][imm [[LT]], imm {}](%f)
    _ = f.do_something()


struct Awaitable(Movable where False):
    def __init__(out self):
        pass

    def __await__(mut self) -> Int:
        return 0


# CHECK-LABEL: lit.fn @"awaitable()"
def awaitable() -> Int:
    # CHECK: call {{.*}}@Awaitable::@"__await__{{.*}}(%aw)
    var aw = Awaitable()
    return await aw


# COM: https://github.com/modular/mojo/issues/951
@always_inline
async def inline_async() -> Int:
    return 0


# CHECK-LABEL: lit.fn @"use_inline_async()"
async def use_inline_async() -> Int:
    # CHECK: [[ASYNC_RESULT:%.*]] = lit.async.call{{.*}}inline_async
    # CHECK: [[TMP2:%.*]] = kgen.rebind [[ASYNC_RESULT]] : !co.routine to !alias_AnyCoroutine1
    # CHECK: [[TMP:%.*]] = lit.call {{.*}}Coroutine{{.*}}__init__{{.*}}([[TMP2]]) :
    # CHECK: lit.ref.store [[TMP]], [[CORO:%.*]] : <
    # CHECK: lit.call {{.*}}Coroutine{{.*}}__await__{{.*}}([[CORO]], %{{.*}})
    return await inline_async()


async def capture_byref(mut x: Awaitable, y: Awaitable):
    pass


@fieldwise_init
struct LifetimeAccess[origin: __mlir_type.`!lit.origin<true>`](RegisterPassable):
    pass


async def lifetime_access(var x: LifetimeAccess[_]):
    pass


# CHECK-LABEL: lit.fn @"coroutine_origins
def coroutine_origins():
    # CHECK: var.decl "x" var : {{.*}}mut [[X_LT:.*]]>
    var x: Awaitable
    # CHECK: var.decl "y" var : {{.*}}mut [[Y_LT:.*]]>
    var y: Awaitable
    # CHECK: [[Y_IMM:%.*]] = lit.ref.immut %y
    # CHECK: [[CORO:%.*]] = lit.async.call[!lit.generator<[3]("x": !lit.ref<!Awaitable, mut *[0,0]> mut, "y": !lit.ref<!Awaitable, imm *[0,1]> imm_mem, ?, "__result__": !lit.ref<none, mut *[0,2]> byref_result) async -> !kgen.none>
    # CHECK-SAME: [mut [[X_LT]], muttoimm [[Y_LT]], imm {}](%x, [[Y_IMM]])
    # CHECK: [[CORO2:%.*]] = kgen.rebind [[CORO]] : !co.routine to !alias_AnyCoroutine1
    # CHECK: lit.call {{.*}}Coroutine::@"__init__{{.*}}<:!AnyType [{{.*}}@__MLIRType<:non_struct_type none>, none], :origin.set {mut [[X_LT]], mut [[Y_LT]]}>([[CORO2]])
    var coro = capture_byref(x, y)

    # CHECK: lit.async.call[!lit.generator<[2]("x": !lit.ref<!lit.struct<#LifetimeAccess <:origin<true> [[Y_LT]]>>,
    # CHECK-SAME: mut *[0,0]{{.*}}) async -> !kgen.none>: {{.*}}lifetime_access{{.*}}<:origin<true> [[Y_LT]]>]
    # CHECK: #Coroutine <:!AnyType [{{.*}}@__MLIRType<:non_struct_type none>, none], :origin.set {{{.*}}, mut [[Y_LT]]}>
    var access = lifetime_access(LifetimeAccess[origin_of(y)._mlir_origin]())


# CHECK-LABEL: lit.fn @"mem_result{{.*}}(?, %__result__: !lit.ref<!Awaitable, {{.*}}> byref_result) async -> !kgen.none
async def mem_result() -> Awaitable:
    # CHECK: [[CORO:%.*]] = lit.async.call[{{.*}}mem_result()"][imm {}]()
    # CHECK: [[CORO2:%.*]] = kgen.rebind [[CORO]] : !co.routine to !alias_AnyCoroutine1
    # CHECK: lit.call {{.*}}@Coroutine::@"__init__{{.*}}([[CORO2]])
    var coro = mem_result()


# CHECK-LABEL: lit.fn @"mem_raises{{.*}}(?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<:meta<!Int> #alias_Int, {{.*}}> byref_result) throws|async -> !kgen.scalar<bool>
async def mem_raises() raises -> Int:
    # CHECK: [[CORO:%.*]] = lit.async.call[{{.*}}mem_raises()"][imm {}, imm {}]()
    # CHECK: [[CORO2:%.*]] = kgen.rebind [[CORO]] : !co.routine to !alias_AnyCoroutine1
    # CHECK: lit.call {{.*}}@RaisingCoroutine::@"__init__{{.*}}([[CORO2]])
    var coro = mem_raises()


# CHECK-LABEL: lit.fn @"async_closure_capture
def async_closure_capture(x: String):
    @__parameter
    # CHECK: lit.fn *"capture_it
    async def capture_it():
        _ = x

    # CHECK: lit.async.call[{{.*}}capture_it
    # CHECK:  %coro = lit.var.decl{{.*}}Coroutine <{{.*}}{imm *"x`
    var coro = capture_it()


##===----------------------------------------------------------------------===##
# Nested Functions
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"topLevelFunction()"
def topLevelFunction() -> Int:
    var a = 0

    # CHECK: lit.fn *"nestedFunction()"
    @__parameter
    def nestedFunction() -> Int:
        # CHECK-NEXT: lit.ref.load %a
        return a

    # CHECK: lit.alias.decl *"b{{.*}}": !lit.generator<:{mut *"a`"}:() capturing -> !alias_Int1> = <*"nestedFunction()">
    comptime b = nestedFunction
    # CHECK: lit.call[!lit.generator<:{mut *"a`"}:() capturing -> !alias_Int1>: *"nestedFunction()"]()
    return nestedFunction()


# CHECK-LABEL: lit.struct.decl @SomeStruct
struct SomeStruct(Movable where False):
    # CHECK-LABEL: @"someMethod({{.*}})"
    def someMethod(self) -> Int:
        var a = 0

        # CHECK: lit.fn *"nestedFunction()"
        @__parameter
        def nestedFunction() -> Int:
            # CHECK-NEXT: lit.ref.load %a
            return a

        # CHECK: lit.alias.decl *"b{{.*}}": !lit.generator<:{mut [[A_LT:\*"a`.*"]]}:() capturing -> !alias_Int1> = <*"nestedFunction()">
        comptime b = nestedFunction
        # CHECK: lit.call[!lit.generator<:{mut [[A_LT]]}:() capturing -> !alias_Int1>: *"nestedFunction()"]()
        return nestedFunction()


# CHECK-LABEL: lit.fn @"closureParameter[def() capturing thin -> index]()"
# CHECK-SAME: capturing ->
def closureParameter[func: def () capturing -> __mlir_type.index]():
    pass


# CHECK-LABEL: lit.fn @"closureParameterCaptures
# CHECK-SAME: :*(0,0):
# CHECK-SAME: func: !lit.generator<:origins:() capturing -> !kgen.none>
def closureParameterCaptures[
    origins: OriginSet, //, func: def () capturing [origins] -> None
]():
    pass


struct HasParam[p: Int](TrivialRegisterPassable):
    pass


def closureParameterInference[
    p: Int, //, f: def () capturing -> None
](arg: HasParam[p]):
    pass


struct HasLifetimeParam[p: Origin[mut=True]](TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"explicitLifetime
# CHECK-SAME: #Origin <:!Bool {:scalar<bool> true}{{.*}}>> lt>
def explicitLifetime[lt: Origin[mut=True], //, arg: HasLifetimeParam[lt]]():
    pass


# CHECK-LABEL: lit.fn @"inaccessibleImplicitLifetimeParam
# CHECK-SAME: <?, *"arg.p._mlir_origin``": origin<true>,
def inaccessibleImplicitLifetimeParam(arg: HasLifetimeParam):
    pass


# CHECK-LABEL: lit.struct.decl @CapturingStruct
struct CapturingStruct[a: Int](Movable where False):
    @staticmethod
    def takeClosure[
        origins: OriginSet, //,
        f: def () capturing [origins] -> None,
    ]():
        pass


# CHECK-LABEL: lit.trait.decl @CapturingTrait
trait CapturingTrait:
    # CHECK: lit.fn @"takeClosure{{.*}}:*(0,0):
    def takeClosure[
        origins: OriginSet, //,
        f: def () capturing [origins] -> None,
    ](self):
        ...


# CHECK-LABEL: lit.struct.decl @CapturingStructTrait
struct CapturingStructTrait(CapturingTrait, RegisterPassable):
    # CHECK: lit.fn @"takeClosure{{.*}}:*(0,0):
    def takeClosure[
        origins: OriginSet, //,
        f: def () capturing [origins] -> None,
    ](self):
        pass


# CHECK-LABEL: lit.fn @"inferCaptureOrigins
def inferCaptureOrigins[
    lt: Origin[mut=True], param: HasLifetimeParam[lt]
](mut x: Int, mut y: Int, arg: HasParam):
    @__parameter
    def bareFunc():
        pass

    @__parameter
    def captureSomething():
        _ = x

    # CHECK: call {{.*}}closureParameterCaptures{{.*}}:origin.set {}),
    # CHECK-SAME: !lit.generator<() capturing -> !kgen.none>
    closureParameterCaptures[bareFunc]()
    # CHECK: call {{.*}}closureParameterCaptures{{.*}}:origin.set {mut *"x`1"}),
    # CHECK-SAME: !lit.generator<:{mut *"x`1"}:() capturing -> !kgen.none>
    closureParameterCaptures[captureSomething]()
    # CHECK: call {{.*}}closureParameterInference{{.*}}<:!Int *"arg.p`{{.*}}",
    # CHECK-SAME: rebind(:!lit.generator<:{mut *"x`1"}:{{.*}} *"captureSomething
    closureParameterInference[captureSomething](arg)

    # CHECK: lit.alias.decl *"unboundSet{{.*}}!kgen.func.literal<:!lit.fn<:*(0,0):
    comptime unboundSet = closureParameterCaptures
    # CHECK: lit.alias.decl *"boundSet{{.*}}!kgen.func.literal<:!lit.fn<:rebind(:origin.set {mut *"x`1"}):
    comptime boundSet = closureParameterCaptures[captureSomething]

    # CHECK: lit.alias.decl *"unboundSingleParam{{.*}}:origin<true> *(0,0)>
    comptime unboundSingleParam = explicitLifetime
    # CHECK: lit.alias.decl *"boundSingleParam{{.*}}:origin<true> *"lt._mlir_origin`"
    comptime boundSingleParam = explicitLifetime[param]

    # CHECK: lit.alias.decl *"memberFunction{{.*}}!kgen.func.literal<:!lit.fn<:*(0,1):
    comptime memberFunction = CapturingStruct.takeClosure

    # CHECK: lit.fn *"captureWithClosure
    # CHECK-SAME: :{mut *"y`{{.*}}", mut |*(0,0)|}:
    @__parameter
    def captureWithClosure[
        lts: OriginSet, //, f: def () capturing [lts] -> None
    ]():
        _ = y

    # CHECK: lit.alias.decl *"boundClosure{{.*}} !lit.generator<:{mut *"y`2", mut |rebind(:origin.set {mut *"x`1"})|}:
    comptime boundClosure = captureWithClosure[captureSomething]


# CHECK-LABEL: lit.fn @"testParameterCapture
def testParameterCapture(mut x: Int, mut y: Int):
    # CHECK: lit.fn *"capture()":{mut *"x`"}
    @__parameter
    def capture():
        _ = x

    # CHECK: lit.fn *"do_it()":{mut *"x`", mut *"y`
    @__parameter
    def do_it():
        _ = y
        capture()


# CHECK-LABEL: lit.fn @"topLevelParamFn[index]()"<a_param>
def topLevelParamFn[a_param: __mlir_type.index]():
    def nestedFunction[b_param: __mlir_type.index]():
        return

    # CHECK: lit.alias.decl *"thinref{{.*}}": !lit.generator<<"b_param": index>
    comptime thinref = nestedFunction
    # CHECK: lit.call tail @decls::@"nestedFunction[index](){{.*}}"<2>()
    nestedFunction[Int(2).__mlir_index__()]()

    var value = 0

    @__copy_capture(value)
    @__parameter
    def capturingNestedFunction() -> Int:
        return value

    # CHECK: lit.alias.decl *"fatRef{{.*}}": !lit.generator<() capturing -> !alias_Int1> = <*"capturingNestedFunction()">
    comptime fatRef = capturingNestedFunction


struct SomeParamStruct[c_param: Int](Movable where False):
    # CHECK-LABEL: lit.fn @"topLevelParamFn{{.*}}<a_param: !Int>
    def topLevelParamFn[a_param: Int](self):
        def nestedFunction[b_param: Int]():
            return

        # CHECK: lit.alias.decl *"reff{{.*}}": !lit.generator<<"b_param": !Int>
        comptime reff = nestedFunction
        # CHECK: lit.call tail @decls::@"nestedFunction[::SIMD[DType.int, 1]](){{.*}}"<{{.*}}2{{.*}}>()
        nestedFunction[2]()


##===----------------------------------------------------------------------===##
# Extern Functions
##===----------------------------------------------------------------------===##

# CHECK: lit.fn @"my_extern_add_one
# CHECK-SAME: external,
# CHECK-SAME: linkageName = #kgen.linkage_name<"add_one" : !kgen.string, false>
@extern("add_one")
def my_extern_add_one(x: Int) abi("Mojo") -> Int:
    ...

##===----------------------------------------------------------------------===##
# Implicit origins for result slots.
##===----------------------------------------------------------------------===##


struct MyStruct(Movable where False):
    def __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"getThing()"
# CHECK-SAME: [mut *"__result__`"](?, %__result__:
def getThing() -> MyStruct:
    # result slot parameter should get a different name to avoid conflict.
    # CHECK: lit.call tail @decls::@"localTest(){{.*}}"[mut *"__result__`"]
    def localTest() -> MyStruct:
        return MyStruct()

    return localTest()


# CHECK-LABEL: lit.fn @"callThing()"
# CHECK-SAME: [mut *"__result__`"](?, %__result__:
def callThing() -> MyStruct:
    return getThing()


##===----------------------------------------------------------------------===##
# Implicit Origin Parameters
##===----------------------------------------------------------------------===##


struct SomeType(Movable where False):
    pass


# COM: An implicit origin is passed into a struct parameter inside a trait
# COM: binding. Ensure this passes `-verify-parameters`.
# CHECK-LABEL: lit.fn @"implicit_origin_as_param
# CHECK-SAME: !lit.ref<{{.*}}<:!AnyType {{.*}}Match<:origin<false> *"arg`">>
def implicit_origin_as_param(
    arg: SomeType,
) -> Bound[Match[origin_of(arg)._mlir_origin]]:
    pass


struct Bound[T: AnyType](Movable where False):
    pass


@fieldwise_init
# CHECK: lit.struct.decl @Match
struct Match[lt: __mlir_type.`!lit.origin<false>`](Movable where False):
    pass
    # CHECK: kgen.conformance {{.*}}::@Deinitable
    # CHECK-NEXT: kgen.witness "__deinit__{{.*}}" : !lit.generator<[1]("self": !lit.ref<!lit.struct<#Match <:origin<false> lt>>, mut *[0,0]> deinit_mem,


##===----------------------------------------------------------------------===##
# Struct field with type of recursive parameter
# https://github.com/modularml/modular/issues/28580
##===----------------------------------------------------------------------===##


trait BarTrait:
    pass


struct Bar[T: BarTrait](Movable where False):
    def __init__(out self):
        pass


struct BarSelf(BarTrait, Movable where False):
    var bar: Bar[Self]

    def __init__(out self):
        # CHECK: [[V0:%.*]] = lit.ref.struct.ger %self
        # CHECK: lit.call{{.*}}__init__{{.*}}([[V0]])
        self.bar = Bar[Self]()


# CHECK-LABEL: lit.struct.decl @RegPassableInitSelfInit
struct RegPassableInitSelfInit(ImplicitlyCopyable, RegisterPassable):
    var a: Int

    # CHECK: lit.fn @"__init__
    # CHECK-SAME: () -> !RegPassableInitSelfInit
    def __init__(out self):
        self.a = 42

    # CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*, %copy:
    # CHECK-SAME: -> !RegPassableInitSelfInit
    def __init__(out self, *, copy: Self):
        self.a = copy.a


# CHECK-LABEL: testRegPassableInitSelf
def testRegPassableInitSelf():
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: %x = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[TMP]], %x
    var x = RegPassableInitSelfInit()
    # CHECK-NEXT: %x2 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: [[TMP2:%.*]] = lit.call {{.*}}__init__{{.*}}"{{.*}}([[TMP]]){{.*}}*, "copy"
    # CHECK-NEXT: lit.ref.store [[TMP2]], %x2
    var x2 = x

    # CHECK-NEXT: [[AP:%.*]] = lit.ref.struct.ger %x[a]
    # CHECK-NEXT: [[ONE:%.*]] = kgen.param.constant
    # CHECK-NEXT: lit.ref.store [[ONE:%.*]], [[AP]]
    x.a = 1


struct OverloadedKwArgs(Movable where False):
    var val: Int

    def __init__(out self, single: Int):
        self.val = single

    def __init__(out self, *, double: Int):
        self.val = double * 2

    def __init__(out self, *, triple: Int):
        self.val = triple * 3

    def __getitem__(self, idx: Int) -> Int:
        return self.val

    def __getitem__(self, *, idx2: Int) -> Int:
        return self.val * 2

    def __getitem__(self, *, idx3: Int) -> Bool:
        return self.val > 0

    def __setitem__(mut self, idx: Int, val: Int):
        self.val = val

    def __setitem__(mut self, val: Int, *, idx2: Int):
        self.val = val * 2

    def overloaded_fn(mut self, x: Int, *, y: Int, z: Int):
        self.val = x + y + z

    def overloaded_fn(mut self, x: Int, *, y2: Int, z: Int):
        self.val = x + y2 * 2 + z


# CHECK-LABEL: lit.fn @"testOverloadKwArgs
def testOverloadKwArgs():
    # CHECK-NEXT: %0 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %x = lit.var.decl
    # CHECK-NEXT: %1 = lit.call {{.*}}@OverloadedKwArgs{{.*}}single
    var x = OverloadedKwArgs(1)

    # CHECK-NEXT: %2 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %3 = lit.call {{.*}}@OverloadedKwArgs{{.*}}single
    x = OverloadedKwArgs(single=1)

    # CHECK-NEXT: %4 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %5 = lit.call {{.*}}@OverloadedKwArgs{{.*}}double
    x = OverloadedKwArgs(double=1)

    # CHECK-NEXT: %6 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %7 = lit.call {{.*}}@OverloadedKwArgs{{.*}}triple
    x = OverloadedKwArgs(triple=1)

    # CHECK-NEXT: %8 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %9 = kgen.param.constant: !Int = <{:scalar<index> 42}>
    # CHECK-NEXT: %10 = lit.call {{.*}}@OverloadedKwArgs::@"__setitem__{{.*}}"idx"
    x[1] = 42

    # CHECK-NEXT: %11 = kgen.param.constant: !Int = <{:scalar<index> 42}>
    # CHECK-NEXT: %12 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %13 = lit.call {{.*}}@OverloadedKwArgs::@"__setitem__{{.*}}"idx2"
    x[idx2=1] = 42

    # CHECK-NEXT: %14 = lit.ref.immut %x : <!OverloadedKwArgs, mut *"x`">
    # CHECK-NEXT: %15 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %16 = lit.call {{.*}}@OverloadedKwArgs::@"__getitem__{{.*}}idx
    # CHECK-NEXT: %y = lit.var.decl
    # CHECK-NEXT: lit.ref.store %16, %y : <:meta<!Int> #alias_Int, mut *"y`1">
    # CHECK-NEXT: %17 = lit.ref.immut %x : <!OverloadedKwArgs, mut *"x`">
    var y = x[1]

    # CHECK-NEXT: %18 = kgen.param.constant: !Int = <{:scalar<index> 1}
    # CHECK-NEXT: %19 = lit.call {{.*}}@OverloadedKwArgs::@"__getitem__{{.*}}idx2
    # CHECK-NEXT: lit.ref.store %19, %y : <:meta<!Int> #alias_Int, mut *"y`1">
    y = x[idx2=1]

    # CHECK-NEXT: %20 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %21 = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: %22 = kgen.param.constant: !Int = <{:scalar<index> 3}>
    # CHECK-NEXT: %23 = lit.call {{.*}}@OverloadedKwArgs{{.*}}"y"
    # CHECK-NEXT: %z = lit.var.decl "z" var : !lit.ref<none, mut *"z`2">
    # CHECK-NEXT: lit.ref.store %23, %z : <none, mut *"z`2">
    var z = x.overloaded_fn(1, y=2, z=3)

    # CHECK-NEXT: %24 = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: %25 = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: %26 = kgen.param.constant: !Int = <{:scalar<index> 3}>
    # CHECK-NEXT: %27 = lit.call {{.*}}@OverloadedKwArgs{{.*}}"y2"
    # CHECK-NEXT: lit.ref.store %27, %z : <none, mut *"z`2">
    z = x.overloaded_fn(1, y2=2, z=3)


# Can't generate the constructors for a type wrapping !lit.ref
struct MOCO1320[mut: Bool, //, origin: Origin[mut=mut]](Movable where False):
    comptime _mlir_type = __mlir_type[
        `!lit.ref<`,
        Int,
        `, `,
        Self.origin._mlir_origin,
        `>`,
    ]
    var _value: Self._mlir_type

    def __init__(out self, *, x: Self._mlir_type):
        self._value = x

    def __init__(out self, *, ref [Self.origin]to: Int):
        self._value = __get_mvalue_as_litref(to)


struct StructWithParam[a: Int](Movable where False):
    pass


# CHECK-LABEL: lit.fn @"autoparam_mangler_crash
def autoparam_mangler_crash[*types: Int, constraints: StructWithParam]():
    pass


##===----------------------------------------------------------------------===##
# Dependent Constraints
##===----------------------------------------------------------------------===##

def need_positive_int[x: Int]() where x > 0:
    pass

# CHECK-LABEL: lit.struct.decl @ConstraintStruct
# CHECK-SAME: <a: !Int,{{.*}}<sugar_preserved({{.*}}ge(:scalar<index> #lit.struct.extract<:!Int a, "_mlir_value">, 1)), #loc{{.*}}>
struct ConstraintStruct[a: Int] (Movable where False) where a > 0:
    comptime b = Self.a + 1

    def use_known_assumption(self):
        need_positive_int[self.a]()

    @staticmethod
    def static_use_known_assumption():
        need_positive_int[Self.a]()

# CHECK-LABEL: lit.fn @"use_constraint_struct
# CHECK-SAME: <x: !Int,{{.*}}<sugar_preserved({{.*}}ge(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">, 1)), #loc{{.*}}>
def use_constraint_struct[x: Int, cs: ConstraintStruct[x]]() where x > 0:
    need_positive_int[x]()

# CHECK-LABEL: lit.fn @"use_constraint_struct
# CHECK-SAME: <["cs.a`"]*"cs.a`": !Int, +, cs: !lit.struct<#ConstraintStruct <:!Int *"cs.a`">, <{<sugar_preserved(#lit.struct.extract<:!Bool apply(:!lit.generator<("self": !Int, "rhs": !Int) -> !Bool> @std::@builtin::@stubs::@SIMD::@"__gt__(::SIMD[$0, $1],::SIMD[$0, $1])"<:!DType {:dtype index}, :!SIMDLength {1}>, *"cs.a`", {:scalar<index> 0}), "_mlir_value">, ge(:scalar<index> #lit.struct.extract<:!Int *"cs.a`", "_mlir_value">, 1)), {{.*}}>}>>>() -> !kgen.none attributes {sourceName = "use_constraint_struct_autoparam", specialFnKind = 0 : i8} {
def use_constraint_struct_autoparam[cs: ConstraintStruct[_]]():
    pass

# CHECK-LABEL: lit.fn @"use_constraint_struct_in_constraint
# CHECK-SAME: <x: !Int, {<sugar_preserved({{.*}}ge(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">, 1)){{.*}}>, <sugar_preserved({{.*}}ge(:scalar<index> add(#lit.struct.extract<:!Int x, "_mlir_value">, 1), 2)){{.*}}>}
def use_constraint_struct_in_constraint[x: Int]()
    where x > 0
    where ConstraintStruct[x].b > 1:
    pass


def bin_pred(x: Int, y: Int) -> Bool:
    return x > y

@fieldwise_init
struct BinStruct[x: Int, y: Int] (Movable where False) where bin_pred(x, y):
    # Accessing the same parameter through an alias should work.
    comptime xx = Self.x
    # CHECK-LABEL: lit.fn @"get_with_z
    # CHECK-SAME: %__result__: !lit.ref<!lit.struct<#BinStruct
    def get_with_z[z: Int](self) -> BinStruct[Self.xx, z] where bin_pred(Self.x, z):
        return {}

##===----------------------------------------------------------------------===##
# Partially bound Contextual Types.
##===----------------------------------------------------------------------===##

struct PartiallyBoundParam[T: AnyType](Movable where False):
    @implicit
    def __init__(out self: PartiallyBoundParam[Self.T],  x:Self.T):
        pass


def infer_partially_bound_var_type():
    # CHECK: lit.var.decl "v" var : !lit.ref<!lit.struct<#PartiallyBoundParam <:!AnyType !Int>>, mut *"v`1">
    var v: PartiallyBoundParam = 1
