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

# RUN: %parse-mojo-isolated %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s

struct Empty(TrivialRegisterPassable):
    @always_inline("builtin")
    def __init__(out self):
        pass

def takeAnyType[T: AnyType](a: T):
    pass

##===----------------------------------------------------------------------===##
# Input parameters
##===----------------------------------------------------------------------===##

@fieldwise_init
struct StructWithIntParam[size: Int](RegisterPassable):
    pass

# CHECK-LABEL: lit.fn @"paramArith{{.*}}"<x: !Int>() -> !kgen.none
def paramArith[x: Int]():
    # CHECK: lit.alias.decl *"y`": !Bool = <sugar_builtin(apply({{.*}}{_mlir_value: scalar<bool> = eq(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">, 99)})>
    comptime y = x == 98 + 1

def take_3index(a: Int, b: Int, c: Int) -> Int:
    return a

# CHECK-LABEL: lit.fn @"fancy_signature{{.*}}"<dt: !DType, size: !Int>
# CHECK-SAME: (%x: {{.*}}#SIMD <:!DType dt, :!SIMDLength {{.*}}>{{.*}}>,
# CHECK-SAME: %exp: {{.*}}#SIMD <:!DType dt, :!SIMDLength {{.*}}>{{.*}}>) -> !alias_Int1
def fancy_signature[dt: DType, size: Int](
    x: SIMD[dt, size],
    exp: (SIMD)[dt, size]
) -> Int:
  # CHECK: %[[TMP1:.*]] = kgen.param.constant: !Int = <size>
  # CHECK: %[[TMP2:.*]] = kgen.param.constant: !Int = <size>
  # CHECK: %[[TMP3:.*]] = kgen.param.constant: !Int = <size>
  # CHECK: %[[RES:.*]] = lit.call {{.*}}@"take_3index{{.*}}(%[[TMP1]], %[[TMP2]], %[[TMP3]])
  # CHECK: %local = lit.var.decl "local" var
  # CHECK: lit.ref.store %[[RES]], %local
  var local = take_3index(size, size, size)

  # CHECK: %[[TMP:.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int sugar_builtin(apply({{.*}}{_mlir_value: scalar<index> = add(#lit.struct.extract<:!Int size, "_mlir_value">, 42)}))>
  # CHECK: lit.return %[[TMP]]
  return size+42


def generic_fn[a: DType, b: Int, c: TrivialRegisterPassable](d : Int):
  pass

# CHECK: lit.fn @"call_generic{{.*}}"<dt: !DType>()
def call_generic[dt: DType]():
  # CHECK: %[[C57:.*]] = {{.*}}constant{{.*}}57
  # CHECK: lit.call {{.*}}@"generic_fn{{.*}}"<:!DType dt, :!Int {{.*}}42{{.*}}, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !DType>(%[[C57]])
  generic_fn[dt, 42, DType](57)

  # CHECK: %[[C57_2:.*]] = {{.*}}constant{{.*}}57
  # CHECK: lit.call {{.*}}@"generic_fn{{.*}}"<:!DType dt, :!Int {:scalar<index> 13}, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable {{.*}}SIMD<:!DType dt, :!SIMDLength {4}>>(%[[C57_2]])
  generic_fn[dt, 13, SIMD[dt, 4]](57)

# CHECK-LABEL: lit.struct.decl @TestParamStruct<
# CHECK-SAME: [[A:.*]]: !Int>
@fieldwise_init
struct TestParamStruct[A: Int](TrivialRegisterPassable):

  # CHECK: lit.fn @"method{{.*}}"<B: !Int>(%self: !lit.struct<#TestParamStruct <:!Int [[A]]>
  # CHECK-SAME: %other: {{.*}}#TestParamStruct <:!Int sugar_builtin(apply({{.*}}{_mlir_value: scalar<index> = add(#lit.struct.extract<:!Int [[A]], "_mlir_value">, #lit.struct.extract<:!Int B, "_mlir_value">)})>>
  def method[B: Int](self: TestParamStruct[Self.A], other: TestParamStruct[Self.A + B]):
    pass

  # CHECK-LABEL: lit.fn @"aliases{{.*}}%x: {{.*}}#TestParamStruct <
  def aliases(self, x: TestParamStruct[TestParamStruct[Self.A].TypeLevelAlias]):
    # CHECK: lit.alias.decl [[B:.*]]: !Int = <sugar_builtin(apply({{.*}}{_mlir_value: scalar<index> = add(mul(#lit.struct.extract<:!Int *"A`", "_mlir_value">, 2), 1)})>
    comptime B = Self.A + Self.A + 1
    # CHECK: lit.alias.decl *"C{{.*}}: !Int =
    comptime C = B + Self.A
    # CHECK: lit.alias.decl [[D:.*]]: {{.*}}#TestParamStruct <:!Int {{.*}}1{{.*}}> =
    # CHECK-SAME: <apply(:!lit.generator<{{.*}}TestParamStruct <:!Int {:scalar<index> 1}>>> {{.*}}__init__()"<:!Int {:scalar<index> 1}>)>
    comptime D = TestParamStruct[1]()
    # CHECK: %temp = lit.var.decl {{.*}} : {{.*}}#TestParamStruct <:!Int
    var temp: TestParamStruct[C]

    # CHECK: lit.alias.decl *"intVal{{.*}}": !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
    comptime intVal : Int = 42

    # CHECK: %temp2 = lit.var.decl {{.*}} : {{.*}}sugar_member_alias({{.*}}{_mlir_value: scalar<index> = mul(#lit.struct.extract<:!Int *"A`", "_mlir_value">, 2)}{{.*}})
    var temp2: TestParamStruct[TestParamStruct[Self.A].TypeLevelAlias]

  # CHECK: lit.alias.decl *"TypeLevelAlias{{.*}}": !Int = <sugar_builtin(apply({{.*}}{_mlir_value: scalar<index> = mul(#lit.struct.extract<:!Int *"A`", "_mlir_value">, 2)}{{.*}})>
  comptime TypeLevelAlias = Self.A+Self.A

# Test that we support partially bound parameters.
# CHECK-LABEL: lit.fn @"testTestParamStruct
def testTestParamStruct(a: TestParamStruct[4]):
  # CHECK: %0 = lit.call {{.*}}@TestParamStruct::@"__init__{{.*}}<:!Int {{.*}}11{{.*}}>()
  # CHECK: %arg11 = lit.var.decl {{.*}} : {{.*}}#TestParamStruct <:!Int {{.*}}11
  var arg11 = TestParamStruct[11]()

  # CHECK: %1 = lit.ref.load %arg11
  # CHECK: lit.call {{.*}}@TestParamStruct::@"method{{.*}}<:!Int {:scalar<index> 4}, :!Int {:scalar<index> 7}>(%a, %1)
  a.method[7](arg11)

# CHECK-LABEL: lit.fn @"testSIMD(
def testSIMD(a: Float32,
            b: Int32,
            mut reff: Int32):
  # CHECK: %field1 = lit.var.decl {{.*}} : !lit.ref<scalar<f32>
  var field1 = a._mlir_value
  # CHECK: %field2 = lit.var.decl {{.*}} : !lit.ref<scalar<si32>
  var field2 = reff._mlir_value

  # Test calls to methods and operators on parameterized type.
  # CHECK: lit.call {{.*}}@SIMD::@"__add__{{.*}}<:!DType {{.*}}f32{{.*}}, :!SIMDLength {1}>(%a, %a)
  var x = a+a
  # CHECK: lit.call {{.*}}@SIMD::@"__add__{{.*}}<:!DType {{.*}}si32{{.*}}, :!SIMDLength {1}>(%b, %b)
  var y = b+b

# Show that forward references of parameter names can be correctly resolved.
#
# CHECK-LABEL: lit.fn @"paramResolution[
# CHECK-SAME: ::SIMD[DType.int, 1],
# CHECK-SAME: parameters::StructWithIntParam[$0],
# CHECK-SAME: ::SIMD[DType.int, 1],
# CHECK-SAME: parameters::StructWithIntParam[$2]
# CHECK-SAME: ]()"<
# CHECK-SAME: size1: !Int, a: !lit.struct<#StructWithIntParam <:!Int size1>>,
# CHECK-SAME: size2: !Int, b: !lit.struct<#StructWithIntParam <:!Int size2>>>()
def paramResolution[size1: Int, a: StructWithIntParam[size1],
                   size2: Int, b: StructWithIntParam[size2]]():
  pass

# Show that we can implicitly convert from 42's literal type to Int.
# CHECK-LABEL: lit.fn @"implConversion
# CHECK: <a: !lit.struct<#StructWithIntParam <:!Int {:scalar<index> 42}>>
def implConversion[a: StructWithIntParam[42]]():
  pass

# CHECK-LABEL: lit.struct.decl @Pair<dt: !DType>
struct Pair[dt: DType](RegisterPassable):
 # CHECK: lit.struct.field a : {{.*}}#SIMD <:!DType dt, :!SIMDLength {42}>{{.*}}>
 # CHECK: lit.struct.field b : !alias_Int1
  var a : SIMD[Self.dt, 42]
  var b : Int

  # CHECK: lit.fn @"__init__{{.*}}-> !lit.struct<#Pair <:!DType dt>>
  @implicit
  def __init__(out self, a: SIMD[Self.dt, 42]):
    self.a = a
    self.b = 4
  # CHECK: }

  def __init__(out self, *, copy: Self): pass

# CHECK: }

# CHECK: useParameterizedField
def useParameterizedField[x: Pair[.float32]]():
  # CHECK: lit.alias.decl *"y{{.*}}":
  comptime y : SIMD[.float32, 42] = x.a


# CHECK-LABEL: lit.struct.decl @TypeParameter
# CHECK-SAME: <[[TYPE:.*]]: non_struct_type>
struct TypeParameter[T: __mlir_type.`!kgen.non_struct_type`](Movable where False):
  # CHECK: @"bar(parameters::TypeParameter{{.*}}(%self: {{.*}} imm_mem, %val: !kgen.param<:non_struct_type [[TYPE]]>)
  def bar(self, val: Self.T):
    pass

# Test that parameter decls can refine subsequent ones in the same param list.
# CHECK-LABEL: lit.struct.decl @ParamSubst
# CHECK-SAME: <T: {{.*}}, shape: param_list<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable T>>
struct ParamSubst[
    T: TrivialRegisterPassable,
    shape: __mlir_type[`!kgen.param_list<`, T,`>`],
  ](Movable where False): pass

# CHECK-LABEL: lit.fn @"testParamSubst
def testParamSubst():
  # CHECK: %xx = lit.var.decl {{.*}} : !lit.ref<!lit.struct<#ParamSubst <:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable {{..*}}<:non_struct_type index>, {{.*}}:param_list<index> [1, 2]>>,
  var xx : ParamSubst[__mlir_type.index, __mlir_attr.`#kgen.param_list<1, 2> : !kgen.param_list<index>`]


# Test parameter substitution.
# CHECK-LABEL: lit.fn @"fnToCall{{.*}}"<size: !Int, arr: array<to_builtin(:scalar<index> #lit.struct.extract<:!Int size, "_mlir_value">), f32>>()
def fnToCall[size: Int, arr: __mlir_type[`!pop.array<`, size.__mlir_index__(), `, f32>`]]():
  pass

# CHECK: lit.fn @"fnWithCall{{.*}}"<array: array<10, f32>
def fnWithCall[array: __mlir_type[`!pop.array<10, f32>`]]():
   # CHECK: lit.call {{.*}}@"fnToCall{{.*}}"<:!Int {:scalar<index> 10}, :array<10, f32> array>()
   fnToCall[10, array]()

# CHECK-LABEL: lit.fn @"meta_str{{.*}}"<["type.value`"]*"type.value`": string, +, type: !lit.struct<#StringLiteral <:string *"type.value`">>>() -> !kgen.none
def meta_str[type: StringLiteral]():
  pass

# CHECK-LABEL: lit.fn @"str_input_param()"() -> !kgen.none
def str_input_param():
  # CHECK: %0 = lit.call {{.*}}@"meta_str{{.*}}"<{{.*}}!lit.struct<#StringLiteral <:string "123">>{{.*}}>()
  meta_str["123"]()

@fieldwise_init
struct TwoParams[a: Int, b: Int](TrivialRegisterPassable):
    pass

# CHECK-LABEL: lit.fn @"signature_capture{{.*}}"<
# CHECK-SAME: a: !Int,
# CHECK-SAME: f: !lit.generator<<"b": !Int>() -> {{.*}}TwoParams <:!Int a, :!Int *(0,0)>{{.*}}>
def signature_capture[a: Int, f: def[b: Int]() thin -> TwoParams[a, b]]():
    _ = f[2]()

# CHECK-LABEL: lit.fn @"my_constrained{{.*}}"<{{.*}}cond: !Bool, message: !lit.struct<#StringLiteral <:string *"message.value`">>>()
def my_constrained[cond: Bool, message: StringLiteral]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>, *"message.value`"
    __mlir_op.`kgen.param.assert`[cond=cond.__mlir_bool__(), message=message.value]()
    return


# CHECK-LABEL: lit.fn @"pass_str_param
def pass_str_param():
    # CHECK: lit.call {{.+}}my_constrained{{.*}}<:string "foo", :!Bool {:scalar<bool> true}, :!lit.struct<#StringLiteral <:string "foo">> {}>()
    my_constrained[1==1, "foo"]()

# CHECK-LABEL: lit.fn @"implicit_params
# CHECK-SAME: <?, [[VALUE0:.*]]: !Int, [[VALUE1:.*]]: !Int>
# CHECK-SAME: %value: {{.*}}#TwoParams <:!Int [[VALUE0]], :!Int [[VALUE1]]>
def implicit_params(value: TwoParams):
    pass

# CHECK-LABEL: lit.fn @"implicit_params_with_others
# CHECK-SAME: <a: !Int, ?, [[LHS0:.*]]: !Int, [[LHS1:.*]]: !Int, [[RHS0:.*]]: !Int, [[RHS1:.*]]: !Int>
# CHECK-SAME: %lhs: {{.*}}#TwoParams <:!Int [[LHS0]], :!Int [[LHS1]]>
# CHECK-SAME: %rhs: {{.*}}#TwoParams <:!Int [[RHS0]], :!Int [[RHS1]]>
def implicit_params_with_others[a: Int](lhs: TwoParams, rhs: TwoParams):
    pass

# CHECK-LABEL: lit.fn @"infer_implicit_params()"
def infer_implicit_params():
    # CHECK: call {{.*}}implicit_params{{.*}}<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}
    var one = TwoParams[1, 2]()
    implicit_params(one)
    var two = TwoParams[3, 4]()
    # CHECK: call {{.*}}implicit_params_with_others{{.*}}<:!Int {:scalar<index> 42},
    # CHECK-SAME: :!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}, :!Int {:scalar<index> 3}, :!Int {:scalar<index> 4}>
    implicit_params_with_others[42](one, two)

    # CHECK: alias.decl *"partial_bind{{.*}}: !lit.generator<<?, "lhs.a`": !Int, "lhs.b`1": !Int, "rhs.a`2": !Int, "rhs.b`3": !Int>
    # CHECK-SAME: implicit_params_with_others{{.*}}<:!Int {:scalar<index> 1}, :!Int *(0,0), :!Int *(0,1), :!Int *(0,2), :!Int *(0,3)>>
    comptime partial_bind = implicit_params_with_others[1]
    # CHECK: lit.call {{.*}}implicit_params_with_others{{.*}}<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}, :!Int {:scalar<index> 3}, :!Int {:scalar<index> 4}>
    partial_bind(one, two)

def implicit_params_with_var_params[*Ts: Int](s: TwoParams[1, _]): pass

# CHECK-LABEL: lit.fn @"test_implicit_params_with_var_params
def test_implicit_params_with_var_params():
    # CHECK: [[VAL0:%.*]] = lit.call {{.*}}@TwoParams::@"__init__{{.*}}<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>() :
    # CHECK: call {{.*}}@"implicit_params_with_var_params{{.*}}<:param_list<!Int> [], {{.*}}, :!Int {:scalar<index> 2}>([[VAL0]])
    implicit_params_with_var_params(TwoParams[1, 2]())

# CHECK-LABEL: lit.fn @"explicit_autoparameterization
# CHECK-SAME: "<?, [[V0:.*]]: !Int, [[W0:.*]]: !Int, [[W1:.*]]: !Int>(
# CHECK-SAME: %v: {{.*}}#TwoParams <:!Int {:scalar<index> 5}, :!Int [[V0]]>
# CHECK-SAME: %w: {{.*}}#TwoParams <:!Int [[W0]], :!Int [[W1]]>
def explicit_autoparameterization(v: TwoParams[5, _], w: TwoParams[b=_, a=_]):
    pass

comptime TwoParamsSwap[b: Int, a: Int] = TwoParams[a, b]

# CHECK-LABEL: lit.fn @"autoparam_param_alias
# CHECK-SAME: <?, [[B0:.*]]: !Int, [[A0:.*]]: !Int, [[B1:.*]]: !Int>
# CHECK-SAME: %x: {{.*}}#TwoParams <:!Int [[A0]], :!Int [[B0]]>
# CHECK-SAME: %y: {{.*}}#TwoParams <:!Int {:scalar<index> 2}, :!Int [[B1]]>
def autoparam_param_alias(x: TwoParamsSwap, y: TwoParamsSwap[_, 2]) -> Int:
    return x.a + y.b

# CHECK-LABEL: lit.fn @"autoparam_param_alias_params
# CHECK-SAME: <x: !Int, ["y.a`"][[A0:.*]]: !Int, +, y: {{.*}}TwoParams <:!Int [[A0]], :!Int {:scalar<index> 2}>
def autoparam_param_alias_params[x: Int, //, y: TwoParamsSwap[2, _]]():
    pass

struct IndexParam[x: Int](TrivialRegisterPassable):
    @implicit
    def __init__(out self, p: __mlir_type.`!kgen.none`):
        pass


# CHECK-LABEL: lit.fn @"autoparam_of_params
# CHECK-SAME: <a: !Int, ["b.x`"]*"b.x`": !Int, +, b: {{.*}}IndexParam <:!Int *"b.x`">>, c: {{.*}}IndexParam <:!Int a>
def autoparam_of_params[a: Int, //, b: IndexParam, c: IndexParam[a]]():
    pass

# CHECK-LABEL: lit.fn @"autoparam_of_struct_metatype_params
# CHECK-SAME: <["a.x`1"]*"a.x`1": !Int, +, a: meta<!lit.struct<#IndexParam <:!Int *"a.x`1">>>>
def autoparam_of_struct_metatype_params[a: type_of(IndexParam)]():
    pass

@fieldwise_init
struct DependentParams[x: Int, //, p: IndexParam[x]](TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"autoparam_of_dependent_params
# CHECK-SAME: <["dp.x`"]*"dp.x`": !Int, ["dp.p`1"]*"dp.p`1": {{.*}}IndexParam <:!Int *"dp.x`">>, +, dp: {{.*}}DependentParams <:!Int *"dp.x`", :{{.*}}IndexParam <:!Int *"dp.x`">> *"dp.p`1">>
def autoparam_of_dependent_params[dp: DependentParams]():
    pass


# CHECK-LABEL: lit.fn @"function_autoparam
# CHECK-SAME: :{mut |*(0,0)|, mut |*(0,1)|}:<["f.__origins__`"][[F_LT:.*]]: origin.set, ["g.__origins__`1"][[G_LT:.*]]: origin.set, +
# CHECK-SAME: f: !lit.generator<:[[F_LT]]:() capturing -> !kgen.none>
# CHECK-SAME: g: !lit.generator<:[[G_LT]]:() capturing -> !kgen.none>
def function_autoparam[f: def () capturing [_] -> None, g: def () capturing [_] -> None]():
    @__parameter
    def function():
        pass

    # CHECK: lit.alias.decl *"bind_one{{.*}}": !lit.generator<<>!kgen.func.literal<:!lit.fn<() capturing -> !kgen.none>
    # CHECK-SAME: function_autoparam{{.*}}<:origin.set {}, :origin.set {}, :{{.*}} *"function()", :{{.*}} *"function()">
    comptime bind_one = function_autoparam[function, function]


# CHECK-LABEL: lit.fn @"nonprop_capture_set
# CHECK-SAME: ()"<f: !lit.generator<<"g.__origins__`2x": origin.set, +, "g": !lit.generator<:*(1,0):() capturing -> !kgen.none>>:*(0,0):() -> !kgen.none>>()
def nonprop_capture_set[f: def[g: def () capturing [_] -> None] () thin -> None]():
    pass


# CHECK-LABEL: lit.fn @"autoparam_param_vararg
# CHECK-SAME: <["f.__origins__`"]*"f.__origins__`": origin.set, {{.*}} +, f: {{.*}}, x: !lit.struct<#ParameterList{{.*}} pos_vararg>
def autoparam_param_vararg[f: def () thin [_] -> None, *x: Int]():
    pass


# CHECK-LABEL: lit.fn @"auto_kw_default{{.*}}"<u: !Int = {:scalar<index> 3}, |, v: !Int = {:scalar<index> 3}, ?, {{.*}}, {{.*}}>(%a
def auto_kw_default[u: Int = 3, /, v: Int = 3](a: IndexParam, b: IndexParam):
  pass


# CHECK-LABEL: lit.fn @"test_auto_kw_default
# CHECK-SAME: <?, [[A:.*]]: !Int, [[B:.*]]: !Int>(%a
def test_auto_kw_default(a: IndexParam, b: IndexParam):
  # CHECK-NEXT: <:!Int {:scalar<index> 3}, :!Int {:scalar<index> 3}, :!Int [[A]], :!Int [[B]]>
  auto_kw_default(a, b)
  # CHECK-NEXT: <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 3}, :!Int [[A]], :!Int [[B]]>
  auto_kw_default[1](a, b)
  # CHECK-NEXT: <:!Int {:scalar<index> 3}, :!Int {:scalar<index> 2}, :!Int [[A]], :!Int [[B]]>
  auto_kw_default[v=2](a, b)
  # CHECK-NEXT: <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}, :!Int [[A]], :!Int [[B]]>
  auto_kw_default[1, v=2](a, b)


def default_with_parametric_value[i: Int](a: StructWithIntParam[i], b: StructWithIntParam[i] = StructWithIntParam[i]()):
     pass
def test_default_with_parametric_value(zzz: StructWithIntParam[1]):
    default_with_parametric_value(zzz)

struct HasInferredParamWithAutoParam[value: StructWithIntParam, //](Movable where False):
    pass
def test_autoparam_inferred[x: StructWithIntParam]():
    var arg: HasInferredParamWithAutoParam[value=x]


trait ASuperTrait:
    pass


trait ASubTrait(ASuperTrait):
    pass


struct StructWithTraitParam[T: ASuperTrait](Movable where False):
    pass

    def __init__(out self: StructWithTraitParam[Self.T]):
        pass


# CHECK-LABEL: lit.fn @"test_upcast_trait
def test_upcast_trait[T: ASubTrait](tuples: StructWithTraitParam[T]):
    pass

struct TakeSWIP[XYZ: StructWithIntParam = StructWithIntParam[1]()](Movable where False):
    pass

struct TestTakeSWIP[CO: StructWithIntParam](Movable where False):
    var c: TakeSWIP[Self.CO]

##===----------------------------------------------------------------------===##
# Memory-only parameters
##===----------------------------------------------------------------------===##

struct MemoryType(ImplicitlyCopyable):
    var value: Int

    @always_inline("nodebug")
    @implicit
    def __init__(out self, value: Int):
        self.value = value

struct NonMovableMemoryType(ImplicitlyCopyable):
    var value: Int

    @always_inline
    @implicit
    def __init__(out self, value: Int):
        self.value = value

def makeMemoryValue(x: Int) -> MemoryType:
    return x

def passMemoryValue(x: MemoryType) -> MemoryType:
    return x

@always_inline
def readMemoryValue(x: NonMovableMemoryType) -> Int:
    return x.value

# CHECK-LABEL: lit.fn @"callMemoryValueParam
def callMemoryValueParam():
    # CHECK: lit.alias.decl [[PARAM_VALUE1:.*]]: {{.*}}MemoryType = <apply_result_slot({{.*}}makeMemoryValue{{.*}}, {{.*}}1234
    comptime paramValue = makeMemoryValue(1234)
    # CHECK: %dynamicLet = lit.var.decl
    # CHECK: %[[PARAM_VALUE2:.*]] = kgen.param.materialize: !MemoryType =
    # CHECK: lit.ref.store %[[PARAM_VALUE2]], %dynamicLet
    var dynamicLet = paramValue

    comptime nonMovable = NonMovableMemoryType(42)
    # CHECK: %dynamicVar = lit.var.decl
    # CHECK: %[[NON_MOVABLE:.*]] = kgen.param.materialize: !NonMovableMemoryType
    # CHECK: lit.ref.store %[[NON_MOVABLE]], %dynamicVar
    var dynamicVar = nonMovable

    # CHECK: lit.alias.decl [[COPY:.*]]: {{.*}}MemoryType = <apply_result_slot({{.*}}passMemoryValue{{.*}} store_to_mem(
    comptime copy = passMemoryValue(paramValue)
    # CHECK: [[MVALUE:%.*]] = lit.var.decl "anonymous*"
    # CHECK: [[PVALUE:%.*]] = kgen.param.materialize: !MemoryType =
    # CHECK: lit.ref.store [[PVALUE]], [[MVALUE]]
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[MVALUE]]
    # CHECK: lit.var.decl
    # CHECK: lit.call {{.*}}passMemoryValue{{.*}}([[IMMREF]], %{{.*}})
    _ = passMemoryValue(copy)

    # CHECK: lit.call {{.*}}MemoryType::@"__init__(::SIMD[DType.int, 1])"), {:scalar<index> 22})>
    memoryParam[MemoryType(22)]()

    # CHECK: dontFoldMemoryCall{{.*}}{:scalar<index> 42})))
    comptime dontFoldMemoryCall = readMemoryValue(NonMovableMemoryType(42))._mlir_value

# CHECK-LABEL: lit.fn @"memoryParam{{.*}}"<value: !MemoryType>()
def memoryParam[value: MemoryType]():
    pass

struct InitSelfCtor(TrivialRegisterPassable):
    var x: Int

    @always_inline("builtin")
    @implicit
    def __init__(out self, x: Int):
        self.x = x

    @always_inline("builtin")
    def __add__(self, rhs: Self) -> Self:
        return self.x + rhs.x

struct InitSelfParam[x: InitSelfCtor](TrivialRegisterPassable):
    pass


@fieldwise_init("implicit")
struct IntBox(Movable where False):
    var x: Int


@always_inline
def intbox_memory_result(x: Int) -> IntBox:
    return x


# CHECK-LABEL: lit.fn @"interpret_initself_ctor
# CHECK-SAME: %arg: !lit.struct<#InitSelfParam <:!InitSelfCtor {{.*}}{x: !Int = {:scalar<index> 42}})>
def interpret_initself_ctor(arg: InitSelfParam[InitSelfCtor(42)]):
    # CHECK-NEXT: !lit.generator<<>!kgen.func.literal<{{.*}}!lit.struct<#InitSelfParam <:!InitSelfCtor
    comptime refined_fn = refine_memory_only_results[1, 2]

    # CHECK: [[CST:%.*]] = kgen.param.constant: !InitSelfCtor = <{{.*}}{x: !Int = {:scalar<index> 42}})>
    # CHECK-NEXT: store [[CST]], %inlined_initself_call
    var inlined_initself_call = InitSelfCtor(42)

    # CHECK-NEXT: [[CST:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 24}>
    # CHECK: %inlined_byrefresult_call = lit.var.decl "inlined_byrefresult_call"
    # CHECK-NEXT: lit.call{{.*}}intbox_memory_result{{.*}}([[CST]], %inlined_byrefresult_call)
    var inlined_byrefresult_call = intbox_memory_result(24)


def refine_memory_only_results[a: InitSelfCtor, b: InitSelfCtor]() -> InitSelfParam[a + b]:
    pass


struct ConvertFromIntLiteral(Movable where False):
    @implicit
    def __init__(out self, x: IntLiteral):
        pass


def nonmaterializable_arg(x: IntLiteral) -> ConvertFromIntLiteral:
    return x


# CHECK-LABEL: lit.fn @"parameter_memoryonly_call
def parameter_memoryonly_call():
    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: [[TWO:%.*]] = kgen.param.constant: {{.*}}IntLiteral <:!pop.int_literal 2>
    # CHECK-NEXT: [[CST:%.*]] = lit.call {{.*}}ConvertFromIntLiteral::@"__init__{{.*}}([[TWO]], %x)
    var x: ConvertFromIntLiteral = 2
    # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}IntLiteral <:!pop.int_literal 4>
    # CHECK-NEXT: %y = lit.var.decl "y"
    # CHECK-NEXT: [[CST:%.*]] = lit.call {{.*}}nonmaterializable_arg{{.*}}([[FOUR]], %y)
    var y = nonmaterializable_arg(4)


struct IntBoxParam[b: IntBox](Movable where False): pass
def takeIntBoxParam[size: IntBox](a: IntBoxParam[size]): pass
def selectIntBoxFromVariadic(*values: IntBox) -> IntBox: pass


# CHECK-LABEL: lit.fn @"parameter_call_drop_dangling_implicit_origins
def parameter_call_drop_dangling_implicit_origins[b: IntBox]():
    comptime res = selectIntBoxFromVariadic(b)
    var wrapper : IntBoxParam[res]
    takeIntBoxParam[res](wrapper)

# https://github.com/modular/modular/issues/4362 + MOCO-187
# Function call with IntLiteral incorrectly eliminated despite side-effects
def take_nonmat(x: IntLiteral):
    _ = x

# CHECK-LABEL: lit.fn @"call_take_nonmat
def call_take_nonmat():
  comptime a = 1
  # CHECK: lit.call {{.*}}take_nonmat
  take_nonmat(a)

##===----------------------------------------------------------------------===##
# First-class functions as parameters.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"takeCallable{{.*}}"<
# CHECK-SAME: callable: !lit.generator<(!Int, |) -> !alias_Int1>>(%a: !Int) -> !alias_Int1
def takeCallable[
     callable: def(Int) thin -> Int
   ](a: Int) -> Int:
  # CHECK-NEXT: %0 = lit.call tail[!lit.generator<(!Int, |) -> !alias_Int1>: callable](%a)
  # CHECK-NEXT: lit.return %0
  return callable(a)

def takeAndReturnIndex(x: Int) -> Int:
  return x

def posOnlyArg(x: Int, /):
  pass

# CHECK-LABEL: lit.fn @"takeAndReturnIndex
def passFunction(a: Int) -> Int:
  # CHECK: rebind(:!lit.generator<("x": !Int, |) -> !kgen.none> {{.*}}posOnlyArg
  comptime changeKw: def(x: Int) thin -> None = posOnlyArg

  # CHECK: lit.call {{.*}}@"takeCallable{{.*}}<:!lit.generator<(!Int, |) -> !alias_Int1>
  # CHECK-SAME: rebind(:!lit.generator<("x": !Int) -> !alias_Int1> {{.*}}takeAndReturnIndex{{.*}}")>(%a)
  return takeCallable[takeAndReturnIndex](a)

# CHECK-LABEL: lit.fn @"callableWithParam{{.*}}"<type: dtype>() -> !kgen.none
def callableWithParam[type: __mlir_type.`!kgen.dtype`]():
  pass

# CHECK-LABEL: lit.fn @"takeCallable2
def takeCallable2[
      func: def[dt: __mlir_type.`!kgen.dtype`]() thin -> None
  ]():
      pass

# CHECK-LABEL: lit.fn @"passFunctionParam2
def passFunctionParam2():
  # CHECK: lit.call {{.*}}takeCallable2
  # CHECK-SAME: :!lit.generator<<"dt": dtype>() -> !kgen.none> rebind(:!lit.generator<<"type": dtype>() -> !kgen.none> @parameters::@"callableWithParam
  takeCallable2[callableWithParam]()


struct ParamType[x: Int](TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"dependent_function_type
def dependent_function_type[a: Int, f: def (ParamType[a]) thin -> None]():
    comptime func = dependent_function_type
    # CHECK: lit.call{{.*}}dependent_function_type
    func[a, f]()

def overloaded_function():
    pass

def overloaded_function(a: Int):
    pass

struct ParamFuncType[f: def() thin -> None](Movable where False):
    pass

def bind_twice[f: def() thin -> None, g: def(Int) thin -> None]():
    pass

def variadic_func_param[*fs: def() thin -> None]():
    pass

# CHECK-LABEL: lit.fn @"bind_overloaded_fn
def bind_overloaded_fn[f: def[f: def () thin -> None] () thin -> None]():
    # CHECK-NEXT: meta<!lit.struct<#ParamFuncType <:!lit.generator<() -> !kgen.none> {{.*}}@"overloaded_function()"
    comptime T = ParamFuncType[overloaded_function]
    # CHECK-NEXT: meta<!lit.struct<#ParamFuncType <:!lit.generator<() -> !kgen.none> {{.*}}@"overloaded_function()"
    comptime U = ParamFuncType[f=overloaded_function]

    # CHECK-NEXT: bind_params(:{{.*}} f, {{.*}}@"overloaded_function()")
    comptime g = f[overloaded_function]
    # CHECK-NEXT: bind_params(:{{.*}} f, {{.*}}@"overloaded_function()")
    comptime h = f[f=overloaded_function]

    # CHECK-NEXT: bind_twice{{.*}}<:!lit.generator<() -> !kgen.none> {{.*}}@"overloaded_function()", :!lit.generator<(!Int, |) -> !kgen.none> {{.*}}overloaded_function(::SIMD[DType.int, 1])")>
    comptime bound = bind_twice[overloaded_function, overloaded_function]

    # CHECK-NEXT: variadic_func_param{{.*}}<:param_list<{{.*}}> [{{.*}}@"overloaded_function()", {{.*}}@"overloaded_function()"]>
    comptime bind_variadic = variadic_func_param[overloaded_function, overloaded_function]

# Make sure overload resolution can overload things based on parameter bindings.
def paramOverload[*, x: Int](zzz: Int): pass
def paramOverload[y: Int](): pass
def testParamOverload():
    takeAnyType(paramOverload[x=4])
    takeAnyType(paramOverload[y=4])

##===----------------------------------------------------------------------===##
# Alias resolution
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl *"boolDtype{{.*}}": dtype = <bool>
comptime boolDtype = __mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`
# CHECK: lit.alias.decl *"FORTY_TWO{{.*}}": {{.*}}<:!pop.int_literal 42>
comptime FORTY_TWO = 42

# CHECK-LABEL: lit.struct.decl @A
# CHECK-SAME: <v: !Int>
struct A[v: Int](Movable where False):
  # CHECK: lit.alias.decl *"member{{.*}}": !Int = <sugar_builtin(apply({{.*}}{_mlir_value: scalar<index> = add(#lit.struct.extract<:!Int v, "_mlir_value">, 42)}{{.*}})>
  comptime member = Self.v + FORTY_TWO

# CHECK-LABEL: lit.fn @"testUseOfAliases
def testUseOfAliases():
  # This type checks.
  SIMD[DType(boolDtype), 4].splat()
  # CHECK: lit.alias.decl *"y{{.*}}": !Int = <{{.*}}44
  comptime y = A[2].member

struct MyDType(RegisterPassable):
  var state : Int

  def __init__(out self, *, copy: Self):
    self.state = self.state

  @implicit
  def __init__(out self, value: Int):
     self.state = value

  def __eq__(self, rhs: MyDType) -> Bool:
     return __mlir_attr.true

  comptime ui8 = MyDType(Int(1))
  comptime float32 = MyDType(Int(2))
  comptime float64 = MyDType(Int(3))

struct MyVector[size: Int, dtype: MyDType](Movable where False):
    pass

def testMyDType[dt: MyDType](a: MyVector[4, MyDType.float32],
                            b: MyVector[4, dt]):
    pass

# Issue #6828: Unqualified name lookup into structs doesn't work
# CHECK-LABEL: lit.struct.decl @UnqualAliasLookup<param: !Int>
struct UnqualAliasLookup[param: Int](Movable where False):
  # CHECK: lit.alias.decl *"member{{.*}}": !Int = <sugar_builtin(apply({{.*}}{_mlir_value: scalar<index> = add(#lit.struct.extract<:!Int param, "_mlir_value">, 1)}{{.*}})>
  comptime member = Self.param + 1
  def get(self) -> Int:
    # CHECK: %0 = kgen.param.constant: !alias_Int1 = <rebind(:!Int sugar_member_alias({{.*}}{_mlir_value: scalar<index> = add(#lit.struct.extract<:!Int param, "_mlir_value">, 1)}{{.*}})>
    return Self.member

##===----------------------------------------------------------------------===##
# Variadic parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"fnWithVariadics
# CHECK-SAME: <["b.values`"]*"b.values`": param_list<!Int>, +,
# CHECK-SAME: b: !lit.struct<#ParameterList <:!AnyType !Int, :param_list<!Int> *"b.values`">> pos_vararg>
def fnWithVariadics[*b: Int]():
  pass

# CHECK-LABEL: lit.struct.decl @StructWithVariadics
# CHECK-SAME: <["b.values`"]*"b.values`": param_list<!Int>, +,
# CHECK-SAME: b: !lit.struct<#ParameterList <:!AnyType !Int, :param_list<!Int> *"b.values`">> pos_vararg>
struct StructWithVariadics[*b: Int](Movable where False):
    @implicit
    def __init__(out self, i: Int):
        pass

# CHECK-LABEL: lit.fn @"useParamVariadics
def useParamVariadics():
  # CHECK-NEXT: lit.call {{.*}}@"fnWithVariadics{{.*}}"<:param_list<!Int> []
  fnWithVariadics()

  # CHECK: lit.call {{.*}}@"fnWithVariadics{{.*}}"<:param_list<!Int> [{:scalar<index> 1}]
  fnWithVariadics[1]()
  # CHECK: lit.call {{.*}}@"fnWithVariadics{{.*}}"<:param_list<!Int> [{:scalar<index> 1}, {:scalar<index> 2}]
  fnWithVariadics[1, 2]()

  # This keeps the parameters unbound, allowing them to be used with different length..
  # CHECK-NEXT: lit.alias.decl *"defAlias{{[^"]*}}": !lit.generator<<"b.values`": param_list<!Int>{{.*}}>!kgen.func.literal<{{.*}}@"fnWithVariadics
  comptime defAlias = fnWithVariadics

  # Use of an unbound thing in a DRValue context binds an empty variadic list.
  # FIXME(#29495): Pack references aren't working right.
  # HECK-NEXT: [[TMP:%.*]] = kgen.create_closure[!lit.generator<() -> !kgen.none>: @parameters::@"fnWithVariadics{{.*}}"<:param_list<!Int> []>]()
  # HECK-NEXT: %defLet = lit.var.decl "fnLet" : {{.*}}!lit.generator<() -> !kgen.none>
  # HECK-NEXT: lit.ref.store [[TMP]], %defLet
  # var defLet = fnWithVariadics

  # CHECK-NEXT: %a = lit.var.decl {{.*}} : !lit.ref<{{.*}}StructWithVariadics <:param_list<!Int> []
  var a: StructWithVariadics[]
  # CHECK-NEXT: %b = lit.var.decl {{.*}} : !lit.ref<{{.*}}StructWithVariadics <:param_list<!Int> [{:scalar<index> 1}]
  var b: StructWithVariadics[1]
  # CHECK-NEXT: %c = lit.var.decl {{.*}} : !lit.ref<{{.*}}StructWithVariadics <:param_list<!Int> [{:scalar<index> 1}, {:scalar<index> 2}]
  var c: StructWithVariadics[1, 2]

  # TODO(16040): fix symbol name mangling to erase parameter name 'b'
  # CHECK: lit.call {{.*}}@StructWithVariadics::@"__init__({{.*}}<:param_list<!Int> [{:scalar<index> 1}]
  var d = StructWithVariadics[1](2)
  # CHECK: lit.call {{.*}}@StructWithVariadics::@"__init__({{.*}}<:param_list<!Int> []
  var e = StructWithVariadics(3)


# CHECK-LABEL: lit.fn @"unpack_variadic
def unpack_variadic[*a: Int]():
    # CHECK-NEXT: @StructWithVariadics<:param_list<!Int> *"a.values`", :!lit.struct<#ParameterList <:!AnyType !Int, :param_list<!Int> *"a.values`">> a>>
    comptime T = StructWithVariadics[*a]
    # CHECK-NEXT: fnWithVariadics{{.*}}<:param_list<!Int> *"a.values`", :!lit.struct<#ParameterList <:!AnyType !Int, :param_list<!Int> *"a.values`">> a>>
    comptime f = fnWithVariadics[*a]


# CHECK-LABEL: lit.fn @"variadic_parameter{{.*}}"<elems: param_list<index>>
def variadic_parameter[elems: __mlir_type.`!kgen.param_list<index>`]() -> Int:
    return 3

def dependent_variadic_parameter[
    type: TrivialRegisterPassable, //, *values: type
](): pass

# CHECK-LABEL: lit.fn @"pass_variadic{{.*}}"<elems: param_list<index>>
def pass_variadic[elems: __mlir_type.`!kgen.param_list<index>`]():
    # CHECK-NEXT: lit.call {{.*}}@"variadic_parameter{{.*}}"<:param_list<index> elems>
    _ = variadic_parameter[elems]()
    # CHECK: lit.call {{.*}}@"dependent_variadic_parameter{{.*}}"<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int, :param_list<!Int>
    _ = dependent_variadic_parameter[type=Int, 1, 2]()


# Variadic list initialization of List does not work in alias domain
# https://github.com/modularml/modular/issues/33579

# CHECK-LABEL: lit.fn @"init_self_memory_variadics
def init_self_memory_variadics():
    # 1 and 2 need to be passed through memory in the variadics.
    # CHECK-NEXT: lit.alias.decl *"x`":
    # CHECK-SAME:  [store_to_mem({:scalar<index> 1}), store_to_mem({:scalar<index> 2})]
    comptime x = MyList[Int](1, 2)

struct MyList[T: ImplicitlyCopyable](Movable where False):
    @implicit
    def __init__(out self, *values: Self.T): pass

# Infer-only parameters should be bindable with keywords
comptime ImmMyStringSlice = MyStringSlice[mut=False, ...]
struct MyStringSlice[mut: Bool, //, origin: Origin[mut=mut]](Movable where False):  pass

# This only binds to immutable things.
# CHECK-LABEL: lit.fn @"test_imm_string_slice
# CHECK-SAME: (%a: !lit.ref<{{.*}}MyStringSlice <:!Bool {:scalar<bool> false}
def test_imm_string_slice(a: ImmMyStringSlice):
    pass





##===----------------------------------------------------------------------===##
# Function Overloading on Parameters
##===----------------------------------------------------------------------===##


def parameter_overloading[param: Int]():
    pass

def parameter_overloading[param: DType]():
    pass

def partial_parameter_overloading[param: Int, other: Int]():
    pass

def partial_parameter_overloading[param: DType, other: DType]():
    pass

# CHECK-LABEL: lit.fn @"form_reference_to_overloaded
def form_reference_to_overloaded():
    # CHECK-NEXT: @"parameter_overloading[[[INT:.*]]]()"<:!Int {:scalar<index> 1}>
    comptime refresult = parameter_overloading[1]
    # CHECK-NEXT: !lit.generator<<"other": !Int>!kgen.func.literal<{{.*}}@"partial_parameter_overloading[[[INT]],[[INT]]]()"<:!Int {:scalar<index> 1}, :!Int *(0,0)>>
    comptime partial = partial_parameter_overloading[1, _]

##===----------------------------------------------------------------------===##
# Parameter Inference
##===----------------------------------------------------------------------===##

struct StaticVec[size: Int](TrivialRegisterPassable):
  def __init__[type: __mlir_type.`!kgen.dtype`](out self, v: __mlir_type[`!kgen.simd<`, Self.size.__mlir_index__(), `, `, type, `>`]):
      pass

  @staticmethod
  def thing[type: __mlir_type.`!kgen.dtype`](v: __mlir_type[`!kgen.simd<`, Self.size.__mlir_index__(), `, `, type, `>`]):
      return

def callee1[size: Int](v: StaticVec[size]): pass
def callee2[T: TrivialRegisterPassable](v: T): pass
def callee3[size: Int, type: __mlir_type.`!kgen.dtype`]
   (v:  __mlir_type[`!kgen.simd<`, size.__mlir_index__(), `, `, type, `>`]): pass
def callee4[T: __mlir_type.`!kgen.non_struct_type`]
   (v:  __mlir_type[`!kgen.pointer<`, T, `>`]): pass

# CHECK-LABEL: lit.fn @"testParamInference{{.*}}"<size: !Int>(
def testParamInference[size: Int](a: StaticVec[4], b: StaticVec[size],
                                 b2: StaticVec[size+2],
                                 c: __mlir_type.`!kgen.simd<17, f32>`,
                                 d: __mlir_type.`!kgen.pointer<f32>`):
  # CHECK-NEXT: lit.call {{.*}}callee1{{.*}}<{{.*}}4{{.*}}>(%a)
  callee1(a)
  # CHECK-NEXT: lit.call {{.*}}callee1{{.*}}<:!Int size>(%b)
  callee1(b)
  # CHECK-NEXT: lit.call tail {{.*}}callee1{{.*}}{_mlir_value: scalar<index> = add(#lit.struct.extract<:!Int size, "_mlir_value">, 2)}{{.*}}>(%b2)
  callee1(b2)
  # CHECK-NEXT: lit.call {{.*}}callee2{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable {{.*}}StaticVec<:!Int size>>(%b)
  callee2(b)
  # CHECK-NEXT: lit.call {{.*}}callee3{{.*}}<:!Int {:scalar<index> 17}, :dtype f32>(%c)
  callee3[17](c)
  # CHECK-NEXT: lit.call {{.*}}callee4{{.*}}<:non_struct_type f32>(%d)
  callee4(d)

# CHECK-LABEL: lit.struct.decl @Abstraction
# CHECK-SAMEL <[[A:.*]]: !Int>
@fieldwise_init
struct Abstraction[a: Int](TrivialRegisterPassable):
  comptime val = Self.a._mlir_value

  @implicit
  def __init__(out self, arg: Int):
    pass

  @staticmethod
  def push[b: Int]() -> Abstraction[Self.a + b]:
      return Abstraction[Self.a + b]()

  @staticmethod
  def pull[b: Int](value: Abstraction[Self.a + b]):
      return

# CHECK-LABEL: lit.fn @"testDependentType{{.*}}"<
# CHECK-SAME: rank: !Int, shape: array<to_builtin(:scalar<index> #lit.struct.extract<:!Int rank, "_mlir_value">)
def testDependentType[
    rank: Int,
    shape: __mlir_type[`!pop.array<`, rank.__mlir_index__(), `, index>`],
]():
    pass

@no_inline
def dont_interpret():
  pass

# CHECK-LABEL: lit.fn @"testParameterEvaluator()"
def testParameterEvaluator():
  # CHECK-NEXT: lit.alias.decl *"x{{.*}}": scalar<index> = <sugar_member_alias(!lit.struct<#Abstraction <:!Int {:scalar<index> 1}>>, "val", 1)>
  comptime x = Abstraction[1].val
  # CHECK-NEXT: %0 = lit.call {{.*}}@Abstraction::@"push{{.*}}"<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>
  # CHECK-NEXT: %y = lit.var.decl "y"
  # CHECK-NEXT: lit.ref.store %0, %y
  var y : Abstraction[3] = Abstraction[1].push[2]()
  # CHECK-NEXT: [[Y:%.*]] = lit.ref.load %y : {{.*}}Abstraction <:!Int {:scalar<index> 3}>>,
  # CHECK-NEXT: lit.call {{.*}}@Abstraction::@"pull{{.*}}"<{{.*}}>([[Y]])
  Abstraction[1].pull[2](y)
  # CHECK-NEXT: lit.call {{.*}}@"testDependentType{{.*}}"<:!Int {:scalar<index> 1}, :array<1, index> [0]>
  testDependentType[1, __mlir_attr.`#pop.array<0> : !pop.array<1, index>`]()

  # CHECK: lit.call {{.*}}dont_interpret
  dont_interpret()


def takeAbstraction2(value: Abstraction[2]):
    return

struct AnotherAbstraction[a: Int](RegisterPassable):
    var value : Abstraction[Self.a + 1]

    def __init__(out self):
        self.value = Abstraction[Self.a + 1]()

    def __init__(out self, *, copy: Self):
        self.value = copy.value

# CHECK-LABEL: lit.fn @"testDependentField()"
def testDependentField():
    var lvalue = AnotherAbstraction[1]()
    # CHECK: [[VALUE_PTR:%.*]] = lit.ref.struct.ger %lvalue[value] {{.*}}AnotherAbstraction <:!Int {:scalar<index> 1}>>,{{.*}}Abstraction <:!Int {:scalar<index> 2}>>
    takeAbstraction2(lvalue.value)

struct LeafToRootEval[a: Int, b: Int](Movable where False):
    var value: Abstraction[Self.a + Self.b + Self.a]

# CHECK-LABEL: lit.fn @"refine_type_leaf_to_root
def refine_type_leaf_to_root(e: LeafToRootEval[2, 3]):
    # CHECK: lit.var.decl "value" {{.*}}Abstraction <:!Int {:scalar<index> 7}>
    var value = e.value

def tail_types[T: TrivialRegisterPassable, *U: AnyType](a: T, *b: *U):
    pass

# CHECK-LABEL: lit.fn @"call_with_tail_types()"
def call_with_tail_types():
    # CHECK: call {{.*}}tail_types{{.*}}<:param_list<!AnyType> [], :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int,
    tail_types(1)
    # CHECK: call {{.*}}tail_types{{.*}}<:param_list<!AnyType> [!FloatDyn], :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int,
    tail_types(1, 1.2)
    # CHECK: call {{.*}}tail_types{{.*}}<:param_list<!AnyType> [!Int], :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int,
    tail_types(1, 77)

# COM: We can't infer parameters from the default value, but we need to test if
# COM: if other parameters are inferred correctly in their presence.
def infer_with_default_arg[T: TrivialRegisterPassable](a: T, b: Int = 7):
    pass

# CHECK-LABEL: lit.fn @"test_infer_with_default_arg()"
def test_infer_with_default_arg():
    # lit.call {{.*}}::@"infer_with_default_arg[TrivialRegisterPassable]($0,::SIMD[DType.int, 1])"<:non_struct_type !Int>
    infer_with_default_arg(128)


struct InferredDefaultType[dtype: DType = .uint32](Movable where False):
    var value: SIMD[Self.dtype, 1]

    # This default depends on a previous default (on the struct).
    def __init__[v_dtype: DType = Self.dtype](
        out self: InferredDefaultType[v_dtype], value: SIMD[v_dtype, 1]
    ):
        self.value = value


# CHECK-LABEL: lit.fn @"test_dependent_default_param_inference()"() -> !kgen.none
def test_dependent_default_param_inference():
    # CHECK: lit.call {{.*}}@InferredDefaultType::@"__init__[::DType](::SIMD[$1, 1])"{{.*}}<:!DType {:dtype index}, :!DType {:dtype index}>
    _ = InferredDefaultType(3)

# CHECK-LABEL: lit.fn @"indirect_call_infer_params
def indirect_call_infer_params[callee: def[x: Int](y: Abstraction[x]) thin -> None]():
    # CHECK: lit.call tail[!lit.generator<("y": {{.*}}#Abstraction <:!Int {:scalar<index> 2}>
    # CHECK-SAME: bind_params(:!lit.generator<<"x": !Int>("y": {{.*}}Abstraction <:!Int *(0,0)>
    # CHECK-SAME: callee, :!Int {:scalar<index> 2}
    callee(Abstraction[2]())

# COM: test parameter inference through signatureType,
# COM: from issue https://github.com/modular/mojo/issues/1362
def mapSingle[A: AnyType, B: AnyType, R: AnyType](
  f: def(x: A, y: B) thin -> R,
  a: A, b: B
) -> R:
  return f(a, b)
def useMapSingle() -> String:
  def f(x: String, y: String) -> String:
    return String()
  # CHECK: lit.call {{.*}}mapSingle{{.*}}<:!AnyType !String, :!AnyType !String, :!AnyType !String>
  return mapSingle(f, "a", "b")


# COM: Test that keyword-only parameter can be inferred after variadic.
# COM: Issue https://github.com/modularml/modular/issues/33939
def deduce_kw_only[*Ts: Int, x: Int](y: Abstraction[x]):
    pass


# CHECK-LABEL: lit.fn @"out_of_order_kw
def out_of_order_kw[x: Int, y: IndexParam[x]]():
    # CHECK-NEXT: out_of_order_kw{{.*}}<{{.*}}0{{.*}}, :{{.*}}IndexParam <{{.*}}0{{.*}}>> {{.*}}IndexParam::@"__init__{{.*}}<{{.*}}0{{.*}}>, #kgen.none)>>
    comptime bound = out_of_order_kw[y=None, x=0]


# CHECK-LABEL: lit.fn @"test_deduce_kw_only
def test_deduce_kw_only(a: Abstraction[3]):
    # CHECK: call {{.*}}@"deduce_kw_only{{.*}}:param_list<!Int> [{:scalar<index> 1}, {:scalar<index> 2}], {{.*}}:!Int {:scalar<index> 3}>(%a)
    deduce_kw_only[1, 2](a)

# Make sure the +1 in the 'a' argument doesn't break inference.
def test_infer_add(a: SIMD[.float32, 4], b: SIMD[.int32, 5]):
   _ = take_two(a, b)

struct CallableArg[ArgT: TrivialRegisterPassable](Movable where False):
    def __call__(self, arg: Self.ArgT):
        pass

# CHECK-LABEL: lit.fn @"infer_conversion_arg_type
def infer_conversion_arg_type(callable: CallableArg[NoneType]):
    # CHECK: lit.call {{.*}}CallableArg::@"__call__{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !NoneType>
    callable(None)

def take_two[a_type: DType, c_type: DType, width: SIMDLength](
    c: SIMD[c_type, width], a: SIMD[a_type, width + 1],
) -> SIMD[c_type, width]: pass

def implicit_signature[
    type: DType,
    rank: Int, //,
    func: def[width: Int](Abstraction[rank]) thin -> SIMD[type, width],
]():
    pass

# CHECK-LABEL: lit.fn @"signature_inference
def signature_inference[dt: DType, rank: Int]():
    def func[width: Int](idx: Abstraction[rank]) -> SIMD[dt, width]:
        pass

    # CHECK: call {{.*}}implicit_signature{{.*}}<:!DType dt, :!Int rank,
    # CHECK-SAME: :!lit.generator<<"width": !Int>(!lit.struct<#Abstraction <:!Int rank>
    # CHECK-SAME: -> !lit.struct<#SIMD <:!DType dt, :!SIMDLength sugar_builtin(apply(:!lit.generator<("value": !Int, |) -> !SIMDLength> @std::@builtin::@stubs::@SIMDLength::@"__init__(::SIMD[DType.int, 1])", *(0,0)), {_mlir_value = to_builtin(:scalar<index> #lit.struct.extract<:!Int *(0,0), "_mlir_value">)})>>
    implicit_signature[func]()


struct ClosureParam[lt: Origin[mut=True], f: def () capturing [lt._mlir_origin] -> None](Movable):
  pass


# CHECK-LABEL: lit.fn @"infer_implicit_params
def infer_implicit_params(var p: ClosureParam):
    # CHECK: lit.call {{.*}}ClosureParam::@"__init__
    # CHECK-SAME: *, "move"
    # CHECK-SAME: <:origin<true> *"p.lt._mlir_origin``",
    # CHECK-SAME: :!lit.generator<:{mut *"p.lt._mlir_origin``"}:() capturing -> !kgen.none> *"p.f`2">
    var tmp = p^
    _ = tmp^


trait ToInt:
    def to_int(self) -> Int:
        ...

@fieldwise_init
struct HasToInt(TrivialRegisterPassable, ToInt):
    var inner: Int
    @always_inline("nodebug")
    def to_int(self) -> Int:
        return self.inner

# COM: https://linear.app/modularml/issue/MOCO-885/crash-when-using-autoparam-in-parametrized-structs
@fieldwise_init
struct MixedInferAndPosParam[size: Int](TrivialRegisterPassable):
    var f0: Int

    # CHECK-LABEL: lit.fn @"__init__[{{.*}}ToInt & ::AnyType](
    # CHECK-SAME: T0: !ToInt_AnyType, T1: !ToInt_AnyType
    def __init__[T0: ToInt, T1: ToInt, //](out self, a: T0, b: T1):
        self.f0 = a.to_int()

@fieldwise_init
struct MixedInferAndPosParamWithInferredOnStruct[ST: ToInt, //, size: Int](TrivialRegisterPassable):
    var f0: Int

    # CHECK-LABEL: lit.fn @"__init__[{{.*}}ToInt & ::AnyType](
    # CHECK-SAME: T0: !ToInt_AnyType, T1: !ToInt_AnyType
    def __init__[T0: ToInt, T1: ToInt, //](out self, z: Self.ST, a: T0, b: T1):
        self.f0 = a.to_int()

# CHECK-LABEL: lit.fn @"useMixedInferAndPosParam()"
def useMixedInferAndPosParam():
    # CHECK: lit.call {{.*}}::@MixedInferAndPosParam::@"__init__{{.*}}<:!Int {:scalar<index> 27}, :!ToInt_AnyType !HasToInt, :!ToInt_AnyType !HasToInt
    _ = MixedInferAndPosParam[27](HasToInt(37), HasToInt(47))
    # CHECK: lit.call {{.*}}::@MixedInferAndPosParamWithInferredOnStruct::@"__init__{{.*}}<:!ToInt_AnyType !HasToInt, :!Int {:scalar<index> 27}, :!ToInt_AnyType !HasToInt, :!ToInt_AnyType !HasToInt
    _ = MixedInferAndPosParamWithInferredOnStruct[27](HasToInt(99), HasToInt(37), HasToInt(47))

struct Box[T: AnyType](TrivialRegisterPassable):
    @implicit
    def __init__(out self, x: Self.T):
        pass

# CHECK-LABEL: lit.fn @"infer_box_type
def infer_box_type[T: AnyType, //, box: Box[T]]():
    # CHECK-NEXT: lit.call {{.*}}infer_box_type{{.*}}<:!AnyType !Int,
    infer_box_type[Int()]()

# MOCO-1457: Support struct param inference for origins
struct OriginStructInferenceImm[origin: ImmOrigin](Movable where False):
    def __init__(out self, ref [Self.origin._mlir_origin]data: Int):  pass
struct OriginStructInferencePar[mut: Bool, //, origin: Origin[mut=mut]](Movable where False):
    def __init__(out self, ref [Self.origin._mlir_origin]data: Int):  pass
struct OriginStructInferenceParWrapped[mut: Bool, //, origin: Origin[mut=mut]](Movable where False):
    def __init__(out self, ref [Self.origin]data: Int):  pass
struct OriginStructInferenceParSpecialized[mut: Bool, //, origin: Origin[mut=mut]](Movable where False):
    def __init__[O: ImmOrigin](out self: OriginStructInferenceParSpecialized[O], ref [O]data: Int):  pass

# CHECK-LABEL: lit.fn @"test_origin_struct_inf
def test_origin_struct_inf[imm_data: Int](mut data: Int):
   # This needs to infer the origin through an implicit conversion
   # CHECK: %0 = lit.ref.immut %data
   # CHECK: lit.call {{.*}}OriginStructInferenceImm::@"__init__
   # CHECK-SAME: :origin<false> (mutcast mut *"data`"){{.*}}>(%0, %immTest)
   var immTest = OriginStructInferenceImm(data)

   # CHECK: lit.call {{.*}}OriginStructInferencePar::@"__init__
   # CHECK-SAME: :origin<true> *"data`">> {}>(%data, %parTest)
   var parTest = OriginStructInferencePar(data)

   # CHECK: lit.call {{.*}}OriginStructInferenceParWrapped::@"__init__
   # CHECK-SAME: :origin<true> *"data`">> {}>(%data, %parWrappedTest)
   var parWrappedTest = OriginStructInferenceParWrapped(data)

   # CHECK: %[[IMMUT:.+]] = lit.ref.immut {{.*}} : <!Int, mut [[IMMUT_REF:.+]]>
   # CHECK: lit.call {{.*}}OriginStructInferenceParSpecialized::@"__init__
   # CHECK-SAME: :!Bool {:scalar<bool> false},
   # CHECK-SAME: (%[[IMMUT]], %parSpecializedTest)
   var parSpecializedTest = OriginStructInferenceParSpecialized(imm_data)


# MOCO-2194: Parameter inference correctly folds contextually-evaluated params.
trait WithAnAlias:
    comptime A: AnyType

# Struct that conforms to `WithAnAlias`
struct SomeStruct(WithAnAlias, Movable where False):
    comptime A = Int

# Needs a struct that conforms to `WithAnAlias`
# Will infer `a` and `b` automatically from `__init__`.
struct SomeWrapper[t: WithAnAlias, a: AnyType, b: AnyType](Movable where False):
    @staticmethod
    def __init__(out self: SomeWrapper[Self.t, Self.t.A, Self.t.A]):
        pass

def test_param_inference_contextual_fold():
    # CHECK: lit.call {{.*}}SomeWrapper::@"__init__
    # CHECK-SAME: <:!WithAnAlias_AnyType !SomeStruct,
    # CHECK-SAME: :!AnyType sugar_member_alias(!SomeStruct, "A", !Int),
    # CHECK-SAME: :!AnyType sugar_member_alias(!SomeStruct, "A", !Int)>
    var sw = SomeWrapper[SomeStruct]()


##===----------------------------------------------------------------------===##
# Access parameter through structure
##===----------------------------------------------------------------------===##

struct MultiStruct[p1: Int, p2: Int, p3: Int](Movable where False):
    def __init__(out self): pass

def foo[x: Int]():
  pass

def bar(x : Int):
  pass

# CHECK-LABEL: lit.fn @"reference_params_through_struct
def reference_params_through_struct():
    var x = MultiStruct[52, 9, 33]()

    # CHECK: %[[Y:.*]] = lit.var.decl "y"
    # CHECK-NEXT: %[[P:.*]] = kgen.param.constant: {{.*}} <{:scalar<index> 52}
    # CHECK-NEXT: lit.ref.store %[[P]], %[[Y]]
    var y = x.p1

    # CHECK: %[[P:.*]] = kgen.param.constant: {{.*}} <{:scalar<index> 9}
    # CHECK-NEXT: lit.call {{.*}}bar({{.*}})"(%[[P]])
    bar(x.p2)

    # CHECK: lit.call {{.*}}foo{{.*}}<:!Int {:scalar<index> 33}>
    foo[x.p3]()

    # CHECK: %[[Z:.*]] = lit.var.decl "z"
    # CHECK-NEXT: %[[P:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: lit.ref.store %[[P]], %[[Z]]
    var z = MultiStruct[1, 2, 3].p1

    # CHECK: %[[P:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: lit.call {{.*}}bar({{.*}})"(%[[P]])
    bar(MultiStruct[1, 2, 3].p2)

    # CHECK: lit.call {{.*}}foo{{.*}}<:!Int {:scalar<index> 3}>
    foo[MultiStruct[1, 2, 3].p3]()


struct DependentParam[x: Int, y: ParamType[x]](RegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"auto_param_dependent
# CHECK-SAME: <?, [[Y0:.*]]: !Int, [[Y1:.*]]: {{.*}}#ParamType <:!Int [[Y0]]>>
def auto_param_dependent(value: DependentParam[...]):
    # CHECK-NEXT: ParamType <:!Int [[Y0]]>> = <*"value.y`1">
    comptime param = value.y


##===----------------------------------------------------------------------===##
# Default function parameters
##===----------------------------------------------------------------------===##

def default_params[a: Int, b: Int = 7, c: String = "woof"]():
    pass


# CHECK-LABEL: lit.fn @"test_default_params()"
def test_default_params():
    # CHECK: lit.call {{.*}}@"default_params[::SIMD[DType.int, 1],::SIMD[DType.int, 1],::String]()"
    # CHECK-SAME: <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 7}, {{.*}}#StringLiteral <:string "woof">
    default_params[1]()

    # CHECK: lit.call {{.*}}@"default_params[::SIMD[DType.int, 1],::SIMD[DType.int, 1],::String]()"
    # CHECK-SAME: <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 8}, {{.*}}#StringLiteral <:string "woof">
    default_params[2, 8]()

    # CHECK: lit.call {{.*}}@"default_params[::SIMD[DType.int, 1],::SIMD[DType.int, 1],::String]()"
    # CHECK-SAME: <:!Int {:scalar<index> 4}, :!Int {:scalar<index> 9}, {{.*}}#StringLiteral <:string "meow">
    default_params[4, 9, "meow"]()


def test_indirect_default_params[
    callee: def[a: Int, b: Int = 7, c: String = "woof"]() thin -> None]():

    # CHECK: lit.call tail[!lit.generator<() -> !kgen.none>: bind_params(:!lit.generator<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> callee,
    # CHECK-SAME: :!Int {:scalar<index> 1}, :!Int {:scalar<index> 7}, {{.*}}#StringLiteral <:string "woof"
    callee[1]()

    # CHECK: lit.call tail[!lit.generator<() -> !kgen.none>: bind_params(:!lit.generator<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> callee,
    # CHECK-SAME: :!Int {:scalar<index> 2}, :!Int {:scalar<index> 8}, {{.*}}#StringLiteral <:string "woof"
    callee[2, 8]()

    # CHECK: lit.call tail[!lit.generator<() -> !kgen.none>: bind_params(:!lit.generator<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> callee,
    # CHECK-SAME: :!Int {:scalar<index> 4}, :!Int {:scalar<index> 9}, {{.*}}#StringLiteral <:string "meow"
    callee[4, 9, "meow"]()


struct ContextDefault[
    a: Int = 7,
    b: Int = a,
](Movable where False) where a > 0 where b >= a:
    @staticmethod
    def inner[c: Int = Self.b, d: Int = c]()
        where Self.b > 0
        where c >= Self.b:
        pass


# COM: Check that defaults on prepended contextual params survive full-signature
# COM: reconstruction when referencing a function on a bound parametric type.
# CHECK-LABEL: lit.fn @"test_contextual_default_fn_ref()"
def test_contextual_default_fn_ref():
    # CHECK: lit.alias.decl *"contextual_default_fn`": !lit.generator<
    # CHECK-SAME: "a": !Int = {:scalar<index> 7}
    # CHECK-SAME: "b": !Int = *(0,0)
    # CHECK-SAME: "c": !Int = *(0,1)
    # CHECK-SAME: "d": !Int = *(0,2)
    # CHECK-SAME: {{.*}}ge(:scalar<index> #lit.struct.extract<:!Int *(0,0), "_mlir_value">, 1)
    # CHECK-SAME: {{.*}}le(:scalar<index> #lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">)
    # CHECK-SAME: {{.*}}ge(:scalar<index> #lit.struct.extract<:!Int *(0,1), "_mlir_value">, 1)
    # CHECK-SAME: {{.*}}le(:scalar<index> #lit.struct.extract<:!Int *(0,1), "_mlir_value">, #lit.struct.extract<:!Int *(0,2), "_mlir_value">)
    comptime contextual_default_fn = ContextDefault.inner


# COM: check that inferred parameter values take precedence over defaults
# CHECK-LABEL: lit.fn @"inferred_default_param
def inferred_default_param[dt: DType, w: SIMDLength = 8](a: SIMD[dt, w]):
    pass


# CHECK: lit.fn @"test_inferred_default_param{{.*}}"<x: !Int>
# CHECK: lit.call {{.*}}@"inferred_default_param{{.*}}"<:!DType {{.*}}f32{{.*}}, :!SIMDLength {4}>
# CHECK: lit.call {{.*}}@"inferred_default_param{{.*}}"<:!DType {{.*}}f32{{.*}}, :!SIMDLength sugar_builtin(apply(:!lit.generator<("value": !Int, |) -> !SIMDLength> @std::@builtin::@stubs::@SIMDLength::@"__init__(::SIMD[DType.int, 1])", x), {_mlir_value = to_builtin(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">)})>
def test_inferred_default_param[
    x: Int
](concrete: SIMD[.float32, 4], p: SIMD[.float32, x]):
    inferred_default_param(concrete)
    inferred_default_param(p)


# COM: basic check for memory-only default parameters
@fieldwise_init
struct MemoryOnlyType(Movable where False):
    pass


# CHECK: lit.fn @"mem_only_default_param[{{.*}}MemoryOnlyType::@"__init__()
def mem_only_default_param[x: MemoryOnlyType = MemoryOnlyType()]():
    pass

# CHECK-LABEL: lit.fn @"test_mem_only_default_param()"
# CHECK: lit.call {{.*}}@"mem_only_default_param[{{.*}}MemoryOnlyType::@"__init__()
def test_mem_only_default_param():
    mem_only_default_param()

# CHECK-LABEL: lit.fn @"param_default{{.*}}"<
# CHECK-SAME: x: !Int = {:scalar<index> 1}>(%y: !Int = x)
def param_default[x: Int = 1](y: Int = x):
    pass

# CHECK-LABEL: lit.fn @"test_param_default
def test_param_default():
    # CHECK: [[C:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 4}>
    # CHECK-NEXT: call {{.*}}param_default{{.*}}<:!Int {:scalar<index> 4}>([[C]]
    param_default[4]()
    # CHECK: [[C:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: call {{.*}}param_default{{.*}}<:!Int {:scalar<index> 1}>([[C]]
    param_default()

struct Optional[T: AnyType](Movable where False):
    @implicit
    def __init__(out self, none: __mlir_type.`!kgen.none`):
        pass

    @implicit
    def __init__(out self, value: Self.T):
        pass

def default_on_infer_failure[p: Int = 0](a: Optional[ParamType[p]] = None):
    pass

# CHECK-LABEL: lit.fn @"test_optional_inference
def test_optional_inference(value: ParamType[3]):
    # CHECK-NEXT: %none = kgen.param.constant
    # CHECK-NEXT: [[NONEVD:%.*]] = lit.var.decl {{.*}}ParamType<:!Int {:scalar<index> 0}>
    # CHECK-NEXT: lit.call {{.*}}@Optional::@"__init__{{.*}}(%none, [[NONEVD]])
    # CHECK: [[IMMUT:%.*]] = lit.ref.immut [[NONEVD]]
    # CHECK-NEXT: call {{.*}}default_on_infer_failure{{.*}}<:!Int {:scalar<index> 0}>([[IMMUT]])
    default_on_infer_failure()

    # CHECK: call {{.*}}default_on_infer_failure{{.*}}<:!Int {:scalar<index> 0}>
    default_on_infer_failure(None)

    # CHECK: call {{.*}}default_on_infer_failure{{.*}}<:!Int {:scalar<index> 3}>
    default_on_infer_failure(value)

# The default value is a different type (non-parameterized) the argument. This
# allows callers to infer O from the default value.

# CHECK-LABEL: lit.fn @"default_inferring_param
# CHECK: (%str: !lit.struct<#StringSpan <{{.*}} O>> = :!alias_StaticString1 apply
def default_inferring_param[O: ImmOrigin](str: StringSpan[O] = StaticString("")):
    pass

# CHECK-LABEL: lit.fn @"test_default_inferring_param
def test_default_inferring_param(b: String):
    # Infers O to default value.
    # CHECK: %0 = kgen.param.constant: !lit.struct<#StringSpan <:!Bool {:scalar<bool> false}, :origin<false> #lit.origin.field<#lit.static.origin : !lit.origin<false>, "__constants__">,
    # CHECK-NEXT: lit.call {{.*}}default_inferring_param{{.*}}(%0)
    default_inferring_param()
    default_inferring_param(StaticString("a"))
    default_inferring_param(b)

##===----------------------------------------------------------------------===##
# Default struct parameters
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.decl @DefaultParams<{{.*}}: !Int, {{.*}}: !Int = {:scalar<index> 7}, {{.*}}: {{.*}}#StringLiteral <:string "woof">
@fieldwise_init
struct DefaultParams[a: Int, b: Int = 7, msg: String = "woof"](Movable where False): pass

# CHECK-LABEL: lit.fn @"test_default_param_struct()"
def test_default_param_struct():
    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int {:scalar<index> 1}, :!Int {:scalar<index> 7}, {{.*}}#StringLiteral <:string "woof">
    comptime T = DefaultParams[1]
    # CHECK-NEXT: %[[INIT:.*]] = lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#DefaultParams <
    # CHECK-SAME:   :!Int {:scalar<index> 1}, :!Int {:scalar<index> 7}, {{.*}}#StringLiteral <:string "woof">
    # CHECK-NEXT: lit.call {{.*}}@DefaultParams::@"__init__({{.*}}<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 7}, {{.*}}#StringLiteral <:string "woof">
    _ = DefaultParams[1]()

    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int {:scalar<index> 2}, :!Int {:scalar<index> 3}, {{.*}}#StringLiteral <:string "woof">
    comptime U = DefaultParams[2, 3]
    # CHECK-NEXT: %[[INIT:.*]] = lit.var.decl {{.*}} synth : !lit.ref<{{.*}}DefaultParams <
    # CHECK-SAME:   :!Int {:scalar<index> 2}, :!Int {:scalar<index> 3}, {{.*}}#StringLiteral <:string "woof">
    # CHECK-NEXT: lit.call {{.*}}@DefaultParams::@"__init__({{.*}}<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 3}, {{.*}}#StringLiteral <:string "woof">
    _ = DefaultParams[2, 3]()

    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int {:scalar<index> 4}, :!Int {:scalar<index> 5}, {{.*}}#StringLiteral <:string "meow">
    comptime S = DefaultParams[4, 5, "meow"]
    # CHECK-NEXT: %[[INIT:.*]] = lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#DefaultParams <
    # CHECK-SAME:   :!Int {:scalar<index> 4}, :!Int {:scalar<index> 5}, {{.*}}#StringLiteral <:string "meow">
    # CHECK-NEXT: lit.call {{.*}}@DefaultParams::@"__init__({{.*}}<:!Int {:scalar<index> 4}, :!Int {:scalar<index> 5}, {{.*}}#StringLiteral <:string "meow">
    _ = DefaultParams[4, 5, "meow"]()


# CHECK: lit.struct.decl @AllDefaultParams<{{.*}}: !Int = {:scalar<index> 0}, {{.*}}MemoryOnlyType::@"__init__()
@fieldwise_init
struct AllDefaultParams[x: Int = 0, v: MemoryOnlyType = MemoryOnlyType()](Movable where False): pass

# CHECK-LABEL: lit.fn @"test_default_param_struct_all_default()"
def test_default_param_struct_all_default():
    # CHECK: lit.alias.decl *"T{{.*}}": meta<!lit.struct<{{.*}}#AllDefaultParams{{.*}}>> = <@{{.*}}::@AllDefaultParams<
    # CHECK-SAME: :!Int {:scalar<index> 0},
    # CHECK-SAME: :!MemoryOnlyType {{.*}}MemoryOnlyType::@"__init__()
    comptime T = AllDefaultParams[]

    # CHECK: %[[INIT:.*]] = lit.var.decl {{.*}} : !lit.ref<{{.*}}#AllDefaultParams <
    # CHECK-SAME:   :!Int {:scalar<index> 0}, :!MemoryOnlyType {{.*}}MemoryOnlyType::@"__init__()
    # CHECK-NEXT: = lit.call {{.*}}::@AllDefaultParams::@"__init__({{.*}}<:!Int {:scalar<index> 0}, :!MemoryOnlyType
    _ = AllDefaultParams[]()


# COM: Issue #22763
def IntForType[T: TrivialRegisterPassable]() -> Int:
    return 1

struct StructWithParametricDefaultValue[T: TrivialRegisterPassable, N: Int = IntForType[T]()](Movable where False):
    pass

# CHECK-LABEL: lit.fn @"test_struct_with_parametric_default_value()"
def test_struct_with_parametric_default_value():
    # CHECK: lit.alias.decl *"a{{.*}}": meta<!lit.struct<{{.*}}>> = <@{{.*}}::@StructWithParametricDefaultValue<
    # CHECK-SAME: :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int,
    # CHECK-SAME: :!Int apply(:!lit.generator<() -> !Int> @{{.*}}::@"IntForType[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]()"{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int>)>
    comptime a = StructWithParametricDefaultValue[Int]

##===----------------------------------------------------------------------===##
# Struct keyword parameters
##===----------------------------------------------------------------------===##

@fieldwise_init
struct KwParamStruct[a: Int, b: Int = 2, c: Int = 3](Movable where False): pass

# CHECK-LABEL: lit.fn @"test_struct_kw_params()"
def test_struct_kw_params():
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#KwParamStruct <:!Int {:scalar<index> 5}, :!Int {:scalar<index> 7}, :!Int {:scalar<index> 3}
    _ = KwParamStruct[5, b=7]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#KwParamStruct <:!Int {:scalar<index> 5}, :!Int {:scalar<index> 7}, :!Int {:scalar<index> 9}
    _ = KwParamStruct[5, b=7, c=9]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#KwParamStruct <:!Int {:scalar<index> 5}, :!Int {:scalar<index> 2}, :!Int {:scalar<index> 9}
    _ = KwParamStruct[5, c=9]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#KwParamStruct <:!Int {:scalar<index> 5}, :!Int {:scalar<index> 7}, :!Int {:scalar<index> 9}
    _ = KwParamStruct[5, c=9, b=7]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#KwParamStruct <:!Int {:scalar<index> 5}, :!Int {:scalar<index> 7}, :!Int {:scalar<index> 9}
    _ = KwParamStruct[a=5, c=9, b=7]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<{{.*}}#KwParamStruct <:!Int {:scalar<index> 5}, :!Int {:scalar<index> 7}, :!Int {:scalar<index> 9}
    _ = KwParamStruct[c=9, b=7, a=5]()

##===----------------------------------------------------------------------===##
# Partial binding
##===----------------------------------------------------------------------===##

@fieldwise_init
struct Thing[v: Int](Movable where False): pass

struct CtadStruct[a: Int, b: Int](Movable where False):
    @implicit
    def __init__(out self, x: Thing[Self.a]): pass

    def __init__(out self, x: Thing[Self.a], y: Thing[Self.b]): pass

    @staticmethod
    def foo(x: Thing[Self.a]): pass

    @staticmethod
    def foo(x: Thing[Self.a], y: Thing[Self.b]): pass

struct CtadStructWithDefault[a: Int, b: Int, c: Int = 8](Movable where False):
    @implicit
    def __init__(out self, x: Thing[Self.a]): pass

    def __init__(out self, x: Thing[Self.a], y: Thing[Self.b]): pass

    @staticmethod
    def foo(x: Thing[Self.a]): pass

    @staticmethod
    def foo(x: Thing[Self.a], y: Thing[Self.b]): pass


struct CtadStructWithMultiDefault[a: Int, b: Int = 6, c: Int = 8, d: Int = 10](Movable where False):
    @implicit
    def __init__(out self, x: CtadStructWithMultiDefault[Self.a]): pass


# CHECK-LABEL: lit.fn @"test_partial_binding_CTAD(
def test_partial_binding_CTAD(multi: CtadStructWithMultiDefault[5]):
    # CHECK: call @{{.*}}::@CtadStruct::@"__init__({{.*}})"{{.*}}<:!Int {:scalar<index> 6}, :!Int {:scalar<index> 7}>
    _ = CtadStruct[b=7](Thing[6]())
    # CHECK: call @{{.*}}::@CtadStruct::@"__init__({{.*}})"{{.*}}<:!Int {:scalar<index> 8}, :!Int {:scalar<index> 9}>
    _ = CtadStruct[](Thing[8](), Thing[9]())
    # CHECK: call @{{.*}}::@CtadStruct::@"foo({{.*}}<:!Int {:scalar<index> 6}, :!Int {:scalar<index> 7}>
    CtadStruct[b=7].foo(Thing[6]())
    # CHECK: call @{{.*}}::@CtadStruct::@"foo({{.*}}<:!Int {:scalar<index> 8}, :!Int {:scalar<index> 9}>
    CtadStruct[].foo(Thing[8](), Thing[9]())

    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int {:scalar<index> 6}, :!Int {:scalar<index> 7}, :!Int {:scalar<index> 8}>
    _ = CtadStructWithDefault[b=7](Thing[6]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 8}>
    _ = CtadStructWithDefault[](y=Thing[1](), x=Thing[2]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int {:scalar<index> 6}, :!Int {:scalar<index> 9}, :!Int {:scalar<index> 8}>
    _ = CtadStructWithDefault(Thing[6](), Thing[9]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int {:scalar<index> 6}, :!Int {:scalar<index> 7}, :!Int {:scalar<index> 8}>
    CtadStructWithDefault[b=7].foo(Thing[6]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 8}>
    CtadStructWithDefault[].foo(y=Thing[1](), x=Thing[2]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int {:scalar<index> 4}, :!Int {:scalar<index> 3}, :!Int {:scalar<index> 8}>
    CtadStructWithDefault.foo(y=Thing[3](), x=Thing[4]())

    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {:scalar<index> 5}, :!Int {:scalar<index> 6}, :!Int {:scalar<index> 9}, :!Int {:scalar<index> 10}>
    _ = CtadStructWithMultiDefault[_, _, 9, _](multi)
    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {:scalar<index> 5}, :!Int {:scalar<index> 6}, :!Int {:scalar<index> 8}, :!Int {:scalar<index> 9}>
    _ = CtadStructWithMultiDefault[_, _, _, 9](multi)
    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {:scalar<index> 5}, :!Int {:scalar<index> 3}, :!Int {:scalar<index> 8}, :!Int {:scalar<index> 9}>
    _ = CtadStructWithMultiDefault[_, 3, _, 9](multi)
    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {:scalar<index> 5}, :!Int {:scalar<index> 6}, :!Int {:scalar<index> 8}, :!Int {:scalar<index> 10}>
    _ = CtadStructWithMultiDefault[5, _, _, _](multi)


# COM: https://github.com/modular/mojo/issues/1227
# COM: Ensure default parameters are rebound during CTAD.
@fieldwise_init
struct DependentDefault[x: Int = 1, y: Int = x](TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"dependent_default_ctad
def dependent_default_ctad():
    # CHECK-NEXT: value{{.*}}: {{.*}}#DependentDefault <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 1}>
    comptime value = DependentDefault()


comptime Scalar = SIMD[_, 1]


# CHECK-LABEL: lit.fn @"scalar_type{{.*}}"<dt: !DType>
def scalar_type[dt: DType]():
    # CHECK: alias.decl [[T:.*]]: meta<{{.*}}SIMD<:!DType dt, :!SIMDLength {1}>>
    comptime T = Scalar[dt]

    #FIXME(29495): reenable.
    # https://github.com/modularml/modular/issues/29495
    # HECK: lit.var.decl "value" = %{{.*}} : !lit.struct<{{.*}}@SIMD<:!DType dt,
    #var value: T = 1
    # HECK: call {{.*}}<:!DType dt, {{.*}}, :!DType dt>(%value)
    #_ = value.cast[dt]()

# CHECK-LABEL: lit.fn @"funct_partial_binding{{.*}}"<x: !Empty, F:
def funct_partial_binding[x: Empty, F: def[t: Empty, s: Empty] () thin -> None]():
    # CHECK: !lit.generator<<"u": !Empty, "v": !Empty>() -> !kgen.none> = <rebind(
    # CHECK-SAME: :!lit.generator<<"t": !Empty, "s": !Empty>() -> !kgen.none>
    # CHECK-SAME: bind_params(:!lit.generator<<"t": !Empty, "s": !Empty>() -> !kgen.none> F, ?, ?)

    comptime G: def[u: Empty, v: Empty] () thin -> None = F[s=_, t=_]
    # CHECK: !lit.generator<<"u": !Empty>() -> !kgen.none> = <rebind(
    # CHECK-SAME: :!lit.generator<<"s": !Empty>() -> !kgen.none>
    # CHECK-SAME: bind_params(:!lit.generator<<"t": !Empty, "s": !Empty>() -> !kgen.none> F, :!Empty x, ?))>
    comptime H: def[u: Empty] () thin -> None = F[x, _]

struct StructWithSpecificSelfInitTypes[size: Int](Movable where False):
    def __init__(out self: StructWithSpecificSelfInitTypes[0]): pass
    @implicit
    def __init__(out self: StructWithSpecificSelfInitTypes[1], a: Int): pass
    def __init__(out self: StructWithSpecificSelfInitTypes[2], a: Int, b: Int): pass

struct DependentSpecificInitSelf[T: AnyType](Movable where False):
    @implicit
    def __init__[U: Movable](out self: DependentSpecificInitSelf[U], var value: U):
        pass

def implicit_convert_specific_Self(value: StructWithSpecificSelfInitTypes[1]):
    pass

# CHECK-LABEL: lit.fn @"test_inference_from_Self_type
def test_inference_from_Self_type(x: Int):
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {:scalar<index> 0}>([[TMP]])
  # CHECK-NEXT: lit.ownership.use [[TMP]]
  _ = StructWithSpecificSelfInitTypes()
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {:scalar<index> 1}>(%x, [[TMP]])
  # CHECK-NEXT: lit.ownership.use [[TMP]]
  _ = StructWithSpecificSelfInitTypes(x)
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {:scalar<index> 2}>(%x, %x, [[TMP]])
  # CHECK-NEXT: lit.ownership.use [[TMP]]
  _ = StructWithSpecificSelfInitTypes(x, x)

  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {:scalar<index> 1}>(%x, [[TMP]])
  # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[TMP]]
  # CHECK-NEXT: call {{.*}}implicit_convert_specific_Self{{.*}}([[IMM]])
  implicit_convert_specific_Self(x)

  # CHECK: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK: lit.call {{.*}}__init__{{.*}}<:!AnyType !Int, :!AnyType_Movable !Int>{{.*}}({{.*}}, [[TMP]])
  _ = DependentSpecificInitSelf(x)

struct AutoParamDefault[value: Int, param: Int, default: Int = param](Movable where False):
    @implicit
    def __init__(out self, ptr: ParamType[Self.value]): pass
    def __init__(out self, *, other: Self): pass
    def method(self, other: ParamType[Self.value]): pass
    def method(self, other: AutoParamDefault[Self.value, ...]): pass

# CHECK-LABEL: lit.fn @"implicit_conversion_overload
def implicit_conversion_overload(x: AutoParamDefault[1, ...], ptr: ParamType[1]):
    # CHECK: call {{.*}}method{{.*}}(%x, %ptr)
    x.method(ptr)

# MOCO-1144
# https://linear.app/modularml/issue/MOCO-1144/[mojo-lang]-crash-on-partially-bound-parameter-list
def takeAnyTypeReturnInt[t: AnyType]() -> Int: pass
struct MOCO1144[
    mut: Bool,
    type: AnyType,
    alignment: Int = takeAnyTypeReturnInt[type]()
](Movable where False): pass
comptime MOCO1144Bound = MOCO1144[True, _, _]

def getMOCO1144Bound() -> MOCO1144Bound[Int]: pass

# CHECK-LABEL: lit.fn @"tryCallingAThingReturningMOCO1144Bound
def tryCallingAThingReturningMOCO1144Bound():
    # CHECK-NEXT:  lit.var.decl "x" {{.*}}#MOCO1144 <:!Bool {:scalar<bool> true}, :!AnyType !Int{{.*}}takeAnyTypeReturnInt[::AnyType]()"<:!AnyType !Int
    var x = getMOCO1144Bound()

struct HasAutoParam[arg: StructWithIntParam](Movable where False):
    pass
# Solving this requires understanding that self_a = shadow_a and
# self_param = shadow_param, even though they're resolved out of order.
struct Ex1[self_a: Int, //, self_param: StructWithIntParam[self_a]](Movable where False):
    def __init__[
        shadow_a: Int, //, shadow_param: StructWithIntParam[shadow_a]
    ](out self: Ex1[shadow_param], data: HasAutoParam[shadow_param]):
        pass

def testEx1(imm_data: HasAutoParam[StructWithIntParam[1]()]):
    var parSpecializedTest = Ex1(imm_data)

# MOCO-1826: Improve parameter inference from other parameter bindings
struct MyTypeWithOrigin[
    elt_is_mutable: Bool,
    origin: Origin[mut=elt_is_mutable], //
](Movable where False): pass
def testMOCO1826[o: ImmOrigin](a: MyTypeWithOrigin[origin=o]): pass

# Test variadic param inference.
def vararg_example(*args: Int) -> Int: pass
def variadic_inf[N: Int](a: StructWithIntParam[vararg_example(N, 2)]): pass
def test_variadic_inf():
    variadic_inf(StructWithIntParam[vararg_example(1, 2)]())


##===----------------------------------------------------------------------===##
# Origin Parameters
##===----------------------------------------------------------------------===##

struct SomeReference[lt: __mlir_type.`!lit.origin<false>`](TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"unbound_origin
# CHECK-SAME: <?, [[R:.*]]: origin<false>>
# CHECK-SAME: #SomeReference <:origin<false> [[R]]>
def unbound_origin(r: SomeReference[_]):
    pass

# #33498: Variadics can't infer types for function pointers
def indirect_function(x: Int):  pass
def take_variadic_pack[*ArgTypes: AnyType](*args: *ArgTypes):  pass

# CHECK-LABEL: call_variadic_pack_with_function
def call_variadic_pack_with_function():
  # CHECK: [[FP:%.*]] = kgen.create_closure[!lit.generator<("x": !Int) -> !kgen.none>: @parameters::@"indirect_function
  # CHECK: lit.call {{.*}}take_variadic_pack
  var x = take_variadic_pack(indirect_function)


# MOCO-1065: Crash handling self conditional conformance inference.
@fieldwise_init
struct MOCO1065[
    mut: Bool, //,
    T: ImplicitlyCopyable,
    o: Origin[mut=mut],
](Movable where False):
    def __init__(out self: MOCO1065[UInt8, Self.o], ref [Self.o] string: Empty):
        pass

def test_MOCO1065[p: Empty](t: Empty):
    var s = MOCO1065(t)
    comptime a = MOCO1065(p)


### Complex dependent type inference problem.
@fieldwise_init
struct DepValue[a: Int](Movable where False): pass
struct DepUser[b: Int](Movable where False):
    def foo(self):
        # This should infer
        var x : DepUser[2] = self.xyz(DepValue[1]())
    def xyz(self, rhs: DepUser) -> type_of(rhs): pass
    @implicit
    def __init__[x: Int](value: DepValue[x], out result: DepUser[x+1]):
        pass


# Ensure we can infer a variadic parameter from inside an incoming
# parameter-value.

def infer_variadic[
    ArgTypes: TypeList[Trait=Movable, ...], //,
    T: type_of(Tuple[*ArgTypes]),
]():
    pass


# CHECK-LABEL:     lit.fn @"test_infer_variadic()"
def test_infer_variadic():
    # CHECK: lit.call {{.*}}@"infer_variadic{{.*}}"<:param_list<!AnyType_Movable>
    # CHECK-SAME: [!Int, !Bool]
    # CHECK-SAME: :meta<!lit.struct<#Tuple <:param_list<!AnyType_Movable>
    infer_variadic[Tuple[Int, Bool]]()

# Make sure we can store RP types with different sugars correctly.
# CHECK-LABEL: lit.fn @"test_sugar_rebind
def test_sugar_rebind[N: Int](a: SIMD[.int32, 2 * N]):
    # CHECK-NEXT: %x = lit.var.decl
    # CHECK-NEXT: %0 = kgen.rebind %a
    # CHECK-NEXT: lit.ref.store %0, %x
    var x: SIMD[.int32, N * 2] = a

@fieldwise_init
struct CanonicalTypesInDel[input_rank: Int, conv_attr: StructWithIntParam[input_rank - 2]](Movable where False):
    pass

# Test remapping of parameter decls in the face of sugar.
struct IO(TrivialRegisterPassable):
    @always_inline("builtin") # Sugared ctor.
    def __init__(out self):
        pass
@fieldwise_init
struct IOSpec[input: IO](TrivialRegisterPassable):
    pass
@fieldwise_init
struct ManagedTensorSlice[input: IO, //, io_spec: IOSpec[input]](TrivialRegisterPassable):
    pass
comptime InputTensor = ManagedTensorSlice[IOSpec[IO()]()]


# We should be able to infer parameter from generator type.

# This specifies the type for a generator that generate a generator type.
comptime GeneratorTypeGeneratorType[
    From: type_of(AnyType), To: type_of(AnyType)
] = __generator_type[From: From] To

comptime Generator[From: Copyable] = Int


def takeGenerator[
    F: type_of(AnyType),
    T: type_of(AnyType), //,
    G: GeneratorTypeGeneratorType[F, T],
]():
    pass

# CHECK-LABEL: lit.fn @"myDriver()"()
def myDriver():
    # We should be able to infer From -> Copyable, To -> mt_Int
    # CHECK: %0 = lit.call {{.*}}@"takeGenerator{{.*}}"<:meta<!AnyType> !AnyType_Copyable_Movable, :meta<!AnyType> meta<!Int>, :!lit.generator<<"From": !AnyType_Copyable_Movable>meta<!Int>>
    takeGenerator[Generator]()

# This should not crash, even though TakesXOrigin has an inferred parameter
# that depends on MUT.
struct TakesBool[B: Bool](Movable where False):
    pass
struct XOrigin[mut: Bool, *, value: TakesBool[mut]](Movable where False):
    pass
struct TakesXOrigin[MUT: Bool, //, O: XOrigin[MUT, ...]](Movable where False):

    # This should be fine, even though MUT is from the struct.
    def foo[P: XOrigin[Self.MUT, ...]](self):
        pass

struct HasInferred[a: XOrigin](Movable where False):
    def also_has_inferred[b: XOrigin](self):
        pass
def test_inferred_mixing[b: XOrigin](a: HasInferred):
   a.also_has_inferred[b]()

# This makes sure we can associate 'origin' back to the inferred result type of
# the Pointer construction.
struct FindOriginFromKGENOrigin[X: AnyType, origin: ImmOrigin](Movable where False):
    var writable: Pointer[Self.X, Self.origin]
    def __init__(out self, ref [Self.origin]w: Self.X):
        self.writable = Pointer(to=w)


# Test non-materializable target of a meta type.
def infer_non_materializable_target_of_meta_type[
    type_type: __mlir_type.`!kgen.type`,
    //,
    type: type_type,
]():
    pass

def test_non_materializable_target_of_meta_type():
    # CHECK:      lit.call tail {{.*}}"infer_non_materializable_target_of_meta_type{{.*}}"
    # CHECK-SAME: <:type meta<!lit.struct<#IntLiteral <:!pop.int_literal 0>
    infer_non_materializable_target_of_meta_type[type_of(0)]()


struct HasParamList[*values: Int](Movable where False):
    def __init__(out self):
        pass


struct HasDefaultParam[strides: HasParamList[...] = HasParamList[4]()](Movable where False):
    pass


# CHECK-LABEL: lit.alias.decl *"WithDefaultParam
# CHECK-SAME: meta<!lit.struct<#HasDefaultParam <:param_list<!Int> [{:scalar<index> 4}], {{.*}}:!lit.struct<#HasParamList <:param_list<!Int> [{:scalar<index> 4}]
comptime WithDefaultParam = HasDefaultParam[]


struct DimList[*values: Int](Movable where False):
    pass


struct NDBuffer[shape: DimList](Movable where False):
    pass


# Make sure we don't default auto-parameterized `shape.values` to empty
def _get_element_idx2(buff: NDBuffer[_]):
    pass


def _copy_nd_buffer_to_layout_tensor[shape: DimList](src: NDBuffer[shape]):
    # CHECK:      lit.call tail @parameters::@"_get_element_idx2
    # CHECK-SAME: <:param_list<!Int> *"shape.values.values``", {{.*}}:!lit.struct<#DimList <:param_list<!Int> *"shape.values.values``"{{.*}}>> shape>
    _get_element_idx2(src)


# Make sure default inferred value is installed.

# CHECK-LABEL: lit.alias.decl *"ImmOrigin{{.*}}": meta<!lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> ?>, <"_mlir_origin": origin<false>, +>>>
comptime ImmOrigin = Origin[mut=False]
# CHECK-LABEL: lit.alias.decl *"ImmAnyOrigin{{.*}}": !lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> #lit.any.origin>>
comptime ImmAnyOrigin = AnyOrigin[mut=False]
# CHECK-LABEL: lit.alias.decl *"ImmUntrackedOrigin{{.*}}": !lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> {}>>
comptime ImmUntrackedOrigin = UntrackedOrigin[mut=False]


def upcast_typelist_callee[y: TypeList[Trait=AnyType, ...]]():
    pass

def upcast_typelist_caller[x: TypeList[Trait=Movable, ...]]():
    # TODO(MOCO-3712): Causes segfault.
    #comptime a: TypeList[Trait=AnyType, ...] = x
    upcast_typelist_callee[x]()
