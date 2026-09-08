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

# RUN: %parse-mojo-isolated -verify-diagnostics %s | FileCheck %s

# CHECK: module {

def noop(): pass


# CHECK-LABEL: lit.struct.decl @MemoryOnlyInt
struct MemoryOnlyInt(ImplicitlyCopyable):
  var x: Int

  # CHECK-LABEL: lit.fn @"__init__
  @implicit
  def __init__(out self, a: Int = 42):
    # CHECK: %0 = lit.ref.struct.ger %self[x]
    # CHECK: %1 = {{.*}}constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
    # CHECK: lit.ref.store %1, %0
    self.x = 1
  def __deinit__(deinit self): pass

  # CHECK-LABEL: lit.fn @"__init__{{.*}}*, %copy
  def __init__(out self, *, copy: Self):
    self.x = copy.x

  @staticmethod
  def variadic(*value: MemoryOnlyInt):
    pass

def consume(var a: MemoryOnlyInt): pass

# This type is used to test implicit conversion from MemoryOnlyInt
struct MemoryOnlyFloat64(Movable where False):
  var x: Float64
  @implicit
  def __init__(out self, value: MemoryOnlyInt):
    self.x = 1.0

# CHECK-LABEL: lit.struct.decl @MemoryOnlyPair
struct MemoryOnlyPair(ImplicitlyCopyable):
  var x: MemoryOnlyInt
  var y: Int

  # CHECK: lit.fn @"__init__{{.*}}(*, %copy: !lit.ref<!MemoryOnlyPair, imm {{.*}}> imm_mem,
  # CHECK-SAME: %self: !lit.ref<!MemoryOnlyPair, mut {{.*}}> byref_result)
  def __init__(out self, *, copy: MemoryOnlyPair):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[x]
    # CHECK-NEXT: %1 = lit.ref.struct.ger %copy[x]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}"{{.*}}(%1, %0){{.*}}*, "copy"
    # CHECK-NEXT: [[SY:%.*]] = lit.ref.struct.ger %self[y]
    # CHECK-NEXT: [[OY:%.*]] = lit.ref.struct.ger %copy[y]
    # CHECK-NEXT: [[OY_VAL:%.*]] = lit.ref.load [[OY]]
    # CHECK-NEXT: lit.ref.store [[OY_VAL]], [[SY]]
    self.x = copy.x
    self.y = copy.y

  # CHECK: lit.fn @"method{{.*}}(
  # CHECK-SAME: %self: !lit.ref<!MemoryOnlyPair, mut {{.*}}> owned_in_mem,
  # CHECK-SAME: %arg: !lit.ref<!MemoryOnlyInt, mut {{.*}}> owned_in_mem)
  def method(var self, var arg: MemoryOnlyInt):
    # CHECK: %0 = lit.ref.struct.ger %self[y]
    # CHECK: %1 = lit.ref.struct.ger %arg[x]
    # CHECK: %4 = lit.ref.load %2
    # CHECK: %5 = lit.ref.load %3
    # CHECK: %6 = lit.call {{.*}}__add__{{.*}}(%4, %5)
    _ = self.y+arg.x

def inferred_function_with_memory_result[
  width: SIMDLength](x: SIMD[.float32, width]) -> MemoryOnlyInt: pass

# CHECK-LABEL: lit.fn @"memoryOnlyOps
def memoryOnlyOps(mut a: MemoryOnlyPair) -> MemoryOnlyPair:
  # CHECK-NEXT: %v1 = lit.var.decl {{.*}} var : !lit.ref<!MemoryOnlyPair,
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}"{{.*}}([[IMMREF]], %v1){{.*}}*, "copy"
  var v1 = a

  # CHECK-NEXT: %v2 = lit.var.decl "v2"
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}"{{.*}}([[IMMREF]], %v2){{.*}}*, "copy"
  var v2 : MemoryOnlyPair = a

  # CHECK-NEXT: lit.ownership.use %a
  _ = a

  a  # expected-warning {{'MemoryOnlyPair' value is unused; assign to '_' to discard the result}}

  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: %regX = lit.var.decl {{.*}}
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[AX]]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}"{{.*}}([[IMMREF]], %regX){{.*}}*, "copy"
  var regX = a.x

  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}"{{.*}}([[IMMREF]], [[AX]]){{.*}}*, "copy"
  a.x = regX

  # Pass memory only things by value as arguments.

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: [[TMPPAIR:%.*]] = lit.var.decl {{.*}}!MemoryOnlyPair
  # CHECK-NEXT: lit.call {{.*}}@"__init__{{.*}}"{{.*}}([[IMMREF]], [[TMPPAIR]]){{.*}}*, "copy"
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[TMPINT:%.*]] = lit.var.decl {{.*}}!MemoryOnlyInt
  # CHECK-NEXT: lit.call {{.*}}@"__init__{{.*}}"{{.*}}([[IMMREF]], [[TMPINT]]){{.*}}*, "copy"
  # CHECK-NEXT: lit.call {{.*}}@"method{{.*}}([[TMPPAIR]], [[TMPINT]])
  a.method(regX)

  # Drill into rvalue without cloning intermediate values.
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: [[V2XX:%.*]] = lit.ref.struct.ger [[V2X]][x]
  # CHECK-NEXT: %v2xx = lit.var.decl "v2xx"
  # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load [[V2XX]]
  # CHECK-NEXT: lit.ref.store [[VAL]], %v2xx
  var v2xx = v2.x.x

  # Implicit conversion between memory-only types.
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: %mpFloat = lit.var.decl
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[V2X]]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[IMMREF]], %mpFloat)
  var mpFloat : MemoryOnlyFloat64 = v2.x

  # CHECK-NEXT: [[SIMDVAL:%.*]] = lit.call {{.*}}SIMD::@"__init__{{.*}}()

  # CHECK: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call {{.*}}inferred_function_with_memory_result{{.*}}([[SIMDVAL]], [[TMP]])
  _ = inferred_function_with_memory_result(SIMD[.float32, 4]())
  # CHECK-NEXT: lit.ownership.use [[TMP]]

  # Memory-only default argument with memory-only result.
  # CHECK-NEXT: %[[C42:.*]] = {{.*}}constant: !Int = <{:scalar<index> 42}>
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%[[C42]], [[TMP]])
  _ = MemoryOnlyInt()
  # CHECK-NEXT: lit.ownership.use [[TMP]]

  # CHECK-NEXT: [[IMMREF1:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[IMMREF2:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: {{%.*}} = lit.var.decl "__passed_varargs__"
  # CHECK-NEXT: {{%.*}} = pop.array.create [[[IMMREF1]], [[IMMREF2]]]
  # CHECK: lit.call {{.*}}VariadicList::@"__init__
  # CHECK: lit.call {{.*}}MemoryOnlyInt::@"variadic
  MemoryOnlyInt.variadic(regX, regX)

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %v2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}"{{.*}}([[IMMREF]], %__result__){{.*}}*, "copy"
  # CHECK-NEXT: [[NONEVAL:%.*]] = kgen.param.constant: none = <#kgen.none>
  # CHECK-NEXT: lit.return [[NONEVAL]]
  return v2

struct DirectInit(Movable where False):
  def __init__(out self):
    pass

def direct_call_init():
  var value: DirectInit
  # This is a call of a static method on an instance, so 'value' is unused.
  # expected-warning @+1 {{'DirectInit' value is unused; assign to '_' to discard the result}}
  value.__init__()

struct DummyFunc(Movable where False):
    @implicit
    def __init__(out self, f: def(Int) thin raises):
        pass

def func_arg_conversion(f: DummyFunc): pass

# CHECK-LABEL: lit.fn @"implicit_func_conversion()"
def implicit_func_conversion():
    def take_int(x: Int) raises:
        pass

    # CHECK: %f = lit.var.decl "f"
    # CHECK: [[CLOSURE:%.*]] = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}([[CLOSURE]], %f)
    var f: DummyFunc = take_int
    # CHECK: [[CLOSURE:%.*]] = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}([[CLOSURE]], %__call_result_tmp__)
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %__call_result_tmp__
    # CHECK: call {{.*}}func_arg_conversion{{.*}}([[IMMREF]])
    func_arg_conversion(take_int)

# CHECK-LABEL: lit.struct.decl @RegPassable
struct RegPassable(ImplicitlyCopyable, RegisterPassable):
  var value: Int
  # CHECK-LABEL: lit.fn @"__init__
  # CHECK-NEXT: %self = lit.var.decl "self" initoutarg
  # CHECK-NEXT: [[VALREF:%.*]] = lit.ref.struct.ger %self[value]
  # CHECK-NEXT: [[VAL:%.*]] = kgen.rebind %value
  # CHECK-NEXT: lit.ref.store [[VAL]], [[VALREF]]
  # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %self
  # CHECK-NEXT: lit.return [[TMP]]
  @implicit
  def __init__(out self, value: Int):
    self.value = value

  def __deinit__(deinit self): pass
  def __neg__(self) -> Self: pass
  def __add__(self, rhs: Self) -> Self: pass
  def __matmul__(self, rhs: Self) -> Self: pass
  def __rmatmul__(lhs, rhs: Self) -> Self: pass

# CHECK-LABEL: lit.struct.decl @StructWithFuncParam<comparator: !lit.generator
# CHECK-SAME: <"T": !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable>(!kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(0,0)>, |)
struct StructWithFuncParam[comparator: def[T: TrivialRegisterPassable] (T) thin -> None](Movable where False):
    # CHECK-LABEL: lit.fn @"f
    # CHECK-SAME: %self: !lit.ref<{{.*}}<:!lit.generator<<"T": !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable>(!kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(0,0)>
    def f(self):
        pass

    # CHECK-LABEL: lit.fn @"g
    def g(self):
        # CHECK: lit.call {{.*}}[imm *"self`2x"]<:!lit.generator<<"T": !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable>(!kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(0,0)>, |)
        # CHECK-SAME: !lit.ref<{{.*}}<"T": !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable>(!kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(0,0)>, |)
        self.f()

# CHECK-LABEL: lit.fn @"simpleMath
def simpleMath(a: Int, b: Int) -> Int:
  # CHECK: %0 = lit.call {{.*}}SIMD::@"__mul__{{.*}}(%b, %a)
  # CHECK: %1 = lit.call {{.*}}SIMD::@"__sub__{{.*}}(%a, %0)
  # CHECK: lit.return %2 : !alias_Int1
  return a-b*a

# CHECK-LABEL: lit.fn @"precedence_associativity
def precedence_associativity(a: Int):
  # CHECK: %z = lit.var.decl "z" var
  var z: Int = 0

  # CHECK: [[SEVENTEENINT:%.*]] = kgen{{.*}}{:scalar<index> 17}
  # CHECK-NEXT: lit.ref.store [[SEVENTEENINT]], %z
  z = 17  # Implicit conversion

  # CHECK-NEXT: %[[ZREF:.*]] = kgen.rebind %z
  # CHECK-NEXT: %[[Z:.*]] = lit.ref.load %[[ZREF]]
  # CHECK-NEXT: %[[POW0:.*]] = lit.call {{.*}}SIMD::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[POW0A:.*]] = kgen.rebind %[[POW0]]
  # CHECK-NEXT: %[[POW0B:.*]] = kgen.rebind %[[POW0A]]
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}{:scalar<index> 2}
  # CHECK-NEXT: %[[POW1:.*]] = lit.call {{.*}}SIMD::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0B]])
  # CHECK-NEXT: %[[POW1R:.*]] = kgen.rebind %[[POW1]]
  # CHECK-NEXT: lit.ref.store %[[POW1R]], %z
  z = 2**(a**z)
  # CHECK-NEXT: %[[ZREF:.*]] = kgen.rebind %z
  # CHECK-NEXT: %[[Z:.*]] = lit.ref.load %[[ZREF]]
  # CHECK-NEXT: %[[POW0:.*]] = lit.call {{.*}}SIMD::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[POW0A:.*]] = kgen.rebind %[[POW0]]
  # CHECK-NEXT: %[[POW0B:.*]] = kgen.rebind %[[POW0A]]
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}{:scalar<index> 2}
  # CHECK-NEXT: %[[POW1:.*]] = lit.call {{.*}}SIMD::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0B]])
  # CHECK-NEXT: %[[POW1R:.*]] = kgen.rebind %[[POW1]]
  # CHECK-NEXT: lit.ref.store %[[POW1R]], %z
  z = 2**a**z
  # CHECK-NEXT:  %[[ZREF:.*]] = kgen.rebind %z
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %[[ZREF]]
  # CHECK-NEXT:  %[[MUL:.*]] = kgen.param.constant: !Int = <{:scalar<index> -6}
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}SIMD::@"__add__{{.*}}(%[[Z]], %[[MUL]])
  # CHECK-NEXT:  %[[ADDR:.*]] = kgen.rebind %[[ADD]]
  # CHECK-NEXT:  lit.ref.store %[[ADDR]], %z
  z = z + 3 * -2
  # CHECK-NEXT:  %[[ZREF:.*]] = kgen.rebind %z
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %[[ZREF]]
  # CHECK-NEXT:  %[[FLOOR_DIV:.*]] = kgen.param.constant: !Int = <{:scalar<index> -2}>
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}SIMD::@"__add__{{.*}}(%[[Z]], %[[FLOOR_DIV]])
  # CHECK-NEXT:  %[[ADDR:.*]] = kgen.rebind %[[ADD]]
  # CHECK-NEXT:  lit.ref.store %[[ADDR]], %z
  z = z + 3 // -2
  # CHECK-NEXT:  %[[ZREF:.*]] = kgen.rebind %z
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %[[ZREF]]
  # CHECK-NEXT:  %[[INT_THREE:.*]] = kgen{{.*}}{:scalar<index> 3}
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}SIMD::@"__add__{{.*}}(%[[Z]], %[[INT_THREE]])
  # CHECK-NEXT:  %[[ADDA:.*]] = kgen.rebind %[[ADD]]
  # CHECK-NEXT:  %[[ADDB:.*]] = kgen.rebind %[[ADDA]]
  # CHECK-NEXT:  %[[NEG:.*]] = kgen{{.*}}{:scalar<index> -2}
  # CHECK-NEXT:  %[[MUL:.*]] =  lit.call {{.*}}SIMD::@"__mul__{{.*}}(%[[ADDB]], %[[NEG]])
  # CHECK-NEXT:  %[[MULR:.*]] = kgen.rebind %[[MUL]]
  # CHECK-NEXT:  lit.ref.store %[[MULR]], %z
  z = (z + 3) * -+2
  # CHECK-NEXT:  %[[ZREF:.*]] = kgen.rebind %z
  # CHECK-NEXT:  %[[INT_TWO:.*]] = kgen{{.*}}{:scalar<index> 2}
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %[[ZREF]]
  # CHECK-NEXT:  %[[POW:.*]] = lit.call {{.*}}SIMD::@"__pow__{{.*}}(%[[INT_TWO]], %[[Z]])
  # CHECK-NEXT:  %[[POWA:.*]] = kgen.rebind %[[POW]]
  # CHECK-NEXT:  %[[POWB:.*]] = kgen.rebind %[[POWA]]
  # CHECK-NEXT:  %[[NEG:.*]] = lit.call {{.*}}SIMD::@"__neg__{{.*}}(%[[POWB]])
  # CHECK-NEXT:  %[[NEGR:.*]] = kgen.rebind %[[NEG]]
  # CHECK-NEXT:  lit.ref.store %[[NEGR]], %z
  z = -2**z

  # div tests
  # CHECK: lit.call {{.*}}__truediv__
  var r0 = Float32(33.0) / Float32(42.0)

  # CHECK: lit.call {{.*}}__truediv__
  var r1 = Float32(33.0) / 42.0

  # COM: test if-else operator associativity
  # CHECK: %[[CREF:.*]] = kgen.rebind %c
  # CHECK-NEXT: %[[C:.*]] = lit.ref.load %[[CREF]]
  # CHECK-NEXT: %[[TEN:.*]] = kgen.param.constant: !Int = <{:scalar<index> 10}>
  # CHECK-NEXT: %[[EQ:.*]] = lit.call {{.*}}__eq__{{.*}}(%[[C]], %[[TEN]])
  # CHECK-NEXT: %[[EQSB:.*]] = lit.call {{.*}}__mlir_bool__{{.*}}%[[EQ]]
  # CHECK-NEXT: %[[RESULT:.*]] = hlcf.elif %[[EQSB]] -> !alias_Int1 {
  # CHECK-NEXT:   %[[ZERO:.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 0})>
  # CHECK-NEXT:   hlcf.yield %[[ZERO]] : !alias_Int1
  # CHECK-NEXT: } else {
  # CHECK-NEXT:  %[[CREF:.*]] = kgen.rebind %c
  # CHECK-NEXT:  %[[C:.*]] = lit.ref.load %[[CREF]]
  # CHECK-NEXT:  %[[ELEVEN:.*]] = kgen.param.constant: !Int = <{:scalar<index> 11}>
  # CHECK-NEXT:  %[[EQ:.*]] = lit.call {{.*}}__eq__{{.*}}(%[[C]], %[[ELEVEN]])
  # CHECK-NEXT:  %[[EQSB:.*]] = lit.call {{.*}}__mlir_bool__{{.*}}(%[[EQ]])
  # CHECK-NEXT:  %[[RIGHT_IF_RESULT:.*]] = hlcf.elif %[[EQSB]] -> !alias_Int1 {
  # CHECK-NEXT:    %[[ONE:.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
  # CHECK-NEXT:    hlcf.yield %[[ONE]] : !alias_Int1
  # CHECK-NEXT:  } else {
  # CHECK-NEXT:    %[[TWO:.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 2})>
  # CHECK-NEXT:    hlcf.yield %[[TWO]] : !alias_Int1
  # CHECK-NEXT:  }
  # CHECK-NEXT:  hlcf.yield %[[RIGHT_IF_RESULT]] : !alias_Int1
  # CHECK-NEXT:}
  var c = 10
  z = 0 if c == 10 else 1 if c == 11 else 2


struct LHS(Movable where False):
  @implicit
  def __init__(out self, value: Int):
    pass

struct RHS(ImplicitlyCopyable):
  def __radd__(self, lhs: LHS) -> RHS: return self
  def __rsub__(self, lhs: LHS) -> RHS: return self
  def __rmul__(self, lhs: LHS) -> RHS: return self
  def __rfloordiv__(self, lhs: LHS) -> RHS: return self
  def __rmod__(self, lhs: LHS) -> RHS: return self
  def __rpow__(self, lhs: LHS) -> RHS: return self
  def __rlshift__(self, lhs: LHS) -> RHS: return self
  def __rrshift__(self, lhs: LHS) -> RHS: return self
  def __rand__(self, lhs: LHS) -> RHS: return self
  def __ror__(self, lhs: LHS) -> RHS: return self
  def __rxor__(self, lhs: LHS) -> RHS: return self

# CHECK-LABEL: lit.fn @"reverse_operators
def reverse_operators(a: RHS):
  # CHECK: lit.call {{.*}}RHS::@"__radd__(expressions::RHS,expressions::LHS)"
  var z = Int(1) + a

  # CHECK: lit.call {{.*}}RHS::@"__rsub__(expressions::RHS,expressions::LHS)"
  z = Int(2) - z

  # CHECK: lit.call {{.*}}RHS::@"__rmul__(expressions::RHS,expressions::LHS)"
  z = Int(3) * z

  # div tests
  # CHECK: lit.call {{.*}}__rtruediv__
  # CHECK: lit.call {{.*}}RHS::@"__rfloordiv__(expressions::RHS,expressions::LHS)"
  var r1 = 33.0 / Float32(42.0)
  z = Int(33) // z

  # CHECK: lit.call {{.*}}RHS::@"__rmod__(expressions::RHS,expressions::LHS)"
  var i0 = Int(10) % z

  # CHECK: lit.call {{.*}}RHS::@"__rpow__(expressions::RHS,expressions::LHS)"
  var i1 = Int(3) ** z

  # CHECK: lit.call {{.*}}RHS::@"__rlshift__(expressions::RHS,expressions::LHS)"
  var i2 = Int(1) << z

  # CHECK: lit.call {{.*}}RHS::@"__rrshift__(expressions::RHS,expressions::LHS)"
  var i3 = Int(1) >> z

  # CHECK: lit.call {{.*}}RHS::@"__rand__(expressions::RHS,expressions::LHS)"
  z = Int(1) & z

  # CHECK: lit.call {{.*}}RHS::@"__ror__(expressions::RHS,expressions::LHS)"
  z = Int(2) | z

  # CHECK: lit.call {{.*}}RHS::@"__rxor__(expressions::RHS,expressions::LHS)"
  z = Int(3) ^ z

# CHECK-LABEL: lit.fn @"precedence_matmul
def precedence_matmul(z: RegPassable) -> RegPassable:
  # CHECK:  [[THREE:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 3}>
  # CHECK-NEXT:  [[THREERP:%.*]] = lit.call {{.*}}@RegPassable::@"__init__{{.*}}([[THREE]])
  # CHECK-NEXT:  [[TWO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
  # CHECK-NEXT:  [[TWORP:%.*]] = lit.call {{.*}}@RegPassable::@"__init__{{.*}}([[TWO]])
  # CHECK-NEXT:  [[TWOTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[TWORP]], [[TWOTMP]]
  # CHECK-NEXT:  [[TWOTMP_IMM:%.*]] = lit.ref.immut [[TWOTMP]]
  # CHECK-NEXT:  [[NEG:%.*]] = lit.call {{.*}}@RegPassable::@"__neg__{{.*}}([[TWOTMP_IMM]])

  # CHECK-NEXT:  [[THREETMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[THREERP]], [[THREETMP]]

  # CHECK-NEXT:  [[NEGTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[NEG]], [[NEGTMP]]
  # CHECK-NEXT:  [[THREETMP_IMM:%.*]] = lit.ref.immut [[THREETMP]]
  # CHECK-NEXT:  [[NEGTMP_IMM:%.*]] = lit.ref.immut [[NEGTMP]]
  # CHECK-NEXT:  [[MATMUL:%.*]] = lit.call {{.*}}@RegPassable::@"__matmul__{{.*}}([[THREETMP_IMM]], [[NEGTMP_IMM]])
  # CHECK-NEXT:  [[MMTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[MATMUL]], [[MMTMP]]
  # CHECK-NEXT:  [[MMTMP_IMM:%.*]] = lit.ref.immut [[MMTMP]]
  # CHECK-NEXT:  [[ADD:%.*]] = lit.call {{.*}}@RegPassable::@"__add__{{.*}}(%z, [[MMTMP_IMM]])
  # CHECK-NEXT:  lit.return [[ADD]] : !RegPassable
  return z + RegPassable(3) @ -RegPassable(2)

# CHECK-LABEL: lit.fn @"precedence_bitwise
def precedence_bitwise(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}{:scalar<index> 2}
  # CHECK-NEXT: %[[MUL:.*]] = lit.call {{.*}}SIMD::@"__mul__{{.*}}(%a, %[[INT_TWO]])
  # CHECK-NEXT: %[[AND:.*]] = lit.call {{.*}}SIMD::@"__and__{{.*}}(%[[MUL]], %b)
  # CHECK-NEXT: %[[INT_FOUR:.*]] = kgen{{.*}}{:scalar<index> 4}
  # CHECK-NEXT: %[[XOR:.*]] = lit.call {{.*}}SIMD::@"__xor__{{.*}}(%[[INT_FOUR]], %c)
  # CHECK-NEXT: %[[OR:.*]] = lit.call {{.*}}SIMD::@"__or__{{.*}}(%[[AND]], %[[XOR]])
  # CHECK-NEXT: %[[ORR:.*]] = kgen.rebind %[[OR]]
  # CHECK-NEXT: lit.return %[[ORR]]
  return a * 2 & b | 4 ^ c

# CHECK-LABEL: @"comparisons
def comparisons(a: Int, b: Int):
   var res: Bool
   # CHECK: lit.call {{.*}}SIMD::@"__lt__{{.*}}(%a, %b)
   res = a < b
   # CHECK: lit.call {{.*}}SIMD::@"__le__{{.*}}(%a, %b)
   res = a <= b
   # CHECK: lit.call {{.*}}SIMD::@"__gt__{{.*}}(%a, %b)
   res = a > b
   # CHECK: lit.call {{.*}}SIMD::@"__ge__{{.*}}(%a, %b)
   res = a >= b
   # CHECK: lit.call {{.*}}SIMD::@"__eq__{{.*}}(%a, %b)
   res = a == b
   # CHECK: lit.call {{.*}}SIMD::@"__ne__{{.*}}(%a, %b)
   res = a != b

trait Boolable:
    def __bool__(self) -> Bool:
        ...

struct Boolish(Boolable, ImplicitlyCopyable, RegisterPassable):
  def __init__(out self, *, copy: Self): pass
  def __bool__(self) -> Bool: return True

struct MemBoolish(ImplicitlyCopyable):
  @implicit
  def __init__(out self, value: Boolish): pass
  def __init__(out self, *, copy: Self): pass
  def __bool__(self) -> Bool: return True

# CHECK-LABEL: @"unary
def unary(a: Bool, b: Int, c: Boolish, d: MemBoolish):
  # CHECK: %0 = lit.call {{.*}}Bool::@"__bool__({{.*}}Bool)"(%a)
  # CHECK: %1 = lit.call {{.*}}Bool::@"__invert__({{.*}}Bool)"(%0)
  _ = not a

  # CHECK: [[EQ:%.*]] = lit.call {{.*}}SIMD::@"__eq__(::SIMD[$0, $1],::SIMD[$0, $1])"
  # CHECK: [[EQBOOL:%.*]] = lit.call {{.*}}Bool::@"__bool__({{.*}}Bool)"([[EQ]])
  # CHECK:  = lit.call {{.*}}Bool::@"__invert__({{.*}}Bool)"([[EQBOOL]])
  _ = not b == 0

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%c)
  # CHECK:  = lit.call {{.*}}Bool::@"__invert__({{.*}}Bool)"([[BOOL]])
  _ = not c

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}@"__bool__{{.*}}(%d)
  # CHECK-NEXT: lit.call {{.*}}__invert__{{.*}}([[BOOL]])
  _ = not d

# CHECK-LABEL: lit.fn @"andOr1
def andOr1(a: Boolish, b: Boolish):
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%a)
  # CHECK: [[SB:%.*]] = lit.call {{.*}}__mlir_bool__{{.*}}([[BOOL]])
  # CHECK: hlcf.elif [[SB]] -> !Boolish {
  # CHECK:   = lit.call {{.*}}__init__{{.*}}"{{.*}}(%b){{.*}}*, "copy"
  # CHECK:   hlcf.yield
  # CHECK: } else {
  # CHECK:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}"{{.*}}(%a){{.*}}*, "copy"
  # CHECK:   hlcf.yield [[TMP]]
  # CHECK: }
  _ = a and b


# CHECK-LABEL: lit.fn @"andOr2
def andOr2(a: Boolish, b: Boolish):
  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.  Boolish is defined with copy ctor so it must be invoked.

  # CHECK: [[ABOOL:%.*]] = lit.call {{.*}}Boolish::@"__bool__{{.*}}(
  # CHECK-NEXT: [[SB:%.*]] = lit.call {{.*}}@Bool::@"__mlir_bool__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.elif [[SB]] -> !Boolish {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}Boolish::@"__init__{{.*}}"{{.*}}(%a){{.*}}*, "copy"
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK:        [[TMP:%.*]] = lit.call {{.*}}Boolish::@"__init__{{.*}}"{{.*}}(%b){{.*}}*, "copy"
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = a or b

# CHECK-LABEL: lit.fn @"andOr3
def andOr3(a: Boolish, c: Bool):
  # Testing two different logic'y types returns the common bool type if present.

  # CHECK: [[ABOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%a)
  # CHECK-NEXT: [[SB:%.*]] = lit.call {{.*}}__mlir_bool__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.elif [[SB]] -> !Bool {
  # CHECK-NEXT:   hlcf.yield %c
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}([[SB]])
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = a and c

# CHECK-LABEL: lit.fn @"andOr4
def andOr4(b: Boolish, c: Bool):
  # Check incompatible types that are nevertheless boolish.
  # CHECK: [[BBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%b)
  # CHECK-NEXT: [[BSB:%.*]] = lit.call {{.*}}__mlir_bool__{{.*}}([[BBOOL]])
  # CHECK-NEXT: = hlcf.elif [[BSB]] -> !Bool {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}([[BSB]])
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK: } else {
  # CHECK-NEXT: hlcf.yield %c : !Bool
  # CHECK-NEXT: }
  _ = b or c

# CHECK-LABEL: lit.fn @"andOr2
def andOr2(b: Boolish, d: MemBoolish):
  # Check memory-only boolish types.
  # Boolish and MemBoolish has a common type of MemBoolish.

  # CHECK: [[DBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%d)
  # CHECK-NEXT: [[DSB:%.*]] = lit.call {{.*}}__mlir_bool__{{.*}}([[DBOOL]])
  # CHECK-NEXT: [[IFRESULT:%.*]] = lit.var.decl {{.*}} : !lit.ref<!MemBoolish
  # CHECK-NEXT: hlcf.elif [[DSB]] {
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}"{{.*}}(%d, [[ANON:%[^)]*]]){{.*}}*, "copy"
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMPMEM:%.*]] = lit.var.decl
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}(%b, [[TMPMEM]])
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}"{{.*}}([[TMPMEM]], [[ANON]]){{.*}}*, "move"
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  _ = d or b

# CHECK-LABEL: lit.fn @"paramAndOr{{.*}}"<a: !Boolish, b: !Boolish>
def paramAndOr[a: Boolish, b: Boolish]():
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: lit.alias.decl *"c{{.*}}": !Boolish = <cond(
  # CHECK-SAME: apply({{.*}}Boolish::@"__bool__{{.*}}"), store_to_mem(a)), "_mlir_value">{{.*}}, b, a)>
  comptime c = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.

  # CHECK: lit.alias.decl *"d{{.*}}": !Boolish = <cond({{.*}}apply({{.*}}Boolish::@"__bool__{{.*}}"), store_to_mem(a)), "_mlir_value">{{.*}}, a, b)>
  comptime d = a or b

# CHECK-LABEL: lit.fn @"do_math
def do_math(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %z = lit.var.decl "z" var
  var z : Int
  # CHECK-NEXT: %[[INT_5:.*]] = kgen{{.*}}{:scalar<index> 5}
  # CHECK-NEXT: %[[MUL:.*]] = lit.call {{.*}}SIMD::@"__mul__{{.*}}(%[[INT_5]], %a)
  # CHECK-NEXT: %[[INT_42:.*]] = kgen{{.*}}{:scalar<index> 42}
  # CHECK-NEXT: %[[ADD:.*]] = lit.call {{.*}}SIMD::@"__add__{{.*}}(%[[INT_42]], %[[MUL]])
  # CHECK-NEXT: %[[ADDR:.*]] = kgen.rebind %[[ADD]]
  # CHECK-NEXT: lit.ref.store %[[ADDR]], %z
  z = 42 + 5*a

  # CHECK-NEXT: %x = lit.var.decl "x" var
  # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %z
  # CHECK-NEXT: lit.ref.store [[TMP]], %x
  # This is checking the lexer handles \ at end of line correctly.
  var x : Int
  x = \
z

  # CHECK-NEXT: lit.call {{.*}}noop()"()
  noop()

  # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %x
  # CHECK-NEXT: lit.return [[TMP]]
  return x

# CHECK-LABEL: lit.fn @"test_if_cond
def test_if_cond(var cond: Bool, memCond: MemBoolish):
    # CHECK: %[[COND:.*]] = lit.ref.load %cond
    # CHECK: %[[LIT_BOOLSB:.*]] = lit.call {{.*}}__mlir_bool__{{.*}}(%[[COND]])
    # CHECK-NEXT: %[[IF_RES:.*]] = hlcf.elif %[[LIT_BOOLSB]]
    # CHECK-NEXT:   %[[INT_TWO:.*]] = kgen{{.*}}{:scalar<index> 2}
    # CHECK-NEXT:   hlcf.yield %[[INT_TWO]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %[[INT_THREE:.*]] = kgen{{.*}}{:scalar<index> 3}
    # CHECK-NEXT:   hlcf.yield %[[INT_THREE]]
    # CHECK-NEXT: }
    # CHECK-NEXT: %i = lit.var.decl "i"
    # CHECK-NEXT: lit.ref.store %[[IF_RES]], %i
    var i: Int = 2 if cond else 3

    # CHECK: [[TRUEB:%.+]] = kgen{{.*}}{:scalar<bool> true}
    # CHECK-NEXT: lit.ref.store [[TRUEB]], %cond
    cond = True
    i += i
    if cond:     # 'if' stmt, not an 'if' expression.
        i += 1

# CHECK-LABEL: lit.fn @"test_param_if_cond{{.*}}"<cond: !Bool>
def test_param_if_cond[cond: Bool]() -> Int:
  # CHECK-NEXT: lit.alias.decl [[I_ALIAS:.*]]: !alias_Int1 = <cond(#lit.struct.extract<:!Bool cond, "_mlir_value">, rebind(:!Int {:scalar<index> 2}), rebind(:!Int {:scalar<index> 3}))>
  comptime i = 2 if cond else 3

  # CHECK-NEXT: lit.alias.decl *"j{{.*}} = <cond({{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">
  # CHECK-SAME: :!pop.float_literal #pop.float_literal<2|1>{{.*}}{:scalar<index> 3})>
  comptime j = 2.0 if cond else Int(3)

  # CHECK-NEXT: %[[I:.*]] = kgen.param.constant: !alias_Int1 = <#alias_i>
  return i

# CHECK-LABEL: lit.fn @"callable_mv[def(::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]](::SIMD[DType.int, 1])"
# CHECK-SAME: <callable: !lit.generator<(!Int, |) -> !alias_Int1>>(%a: !Int) -> !alias_Int1
def callable_mv[callable: def (Int) thin -> Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call tail[!lit.generator<(!Int, |) -> !alias_Int1>: callable](%a)
  return callable(a)

# CHECK-LABEL: lit.fn @"callable_mv_inputs{{.*}})"<
# CHECK-SAME: callable: !lit.generator<<"x": !Int>(!Int, |) -> !alias_Int1>, b: !Int>(%a: !Int) -> !alias_Int1
def callable_mv_inputs[callable: def[x: Int](Int) thin -> Int, b: Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call tail[!lit.generator<(!Int, |) -> !alias_Int1>: bind_params({{.*}}callable, :!Int b)](%a)
  return callable[b](a)

# CHECK-LABEL: lit.fn @"takeIndexParam{{.*}}"<a: !Int>() -> !alias_Int1
def takeIndexParam[a: Int]() -> Int:
  return a + 1

# CHECK-LABEL: lit.fn @"returnIndex()"() -> !alias_Int1
def returnIndex() -> Int:
  return 0

# CHECK-LABEL: lit.fn @"returnIndex2()"() -> !alias_Int1
def returnIndex2() -> Int:
  # CHECK-NEXT: %0 = lit.call {{.*}}takeIndexParam{{.*}}"<:!Int apply({{.*}}@{{.*}}returnIndex()")>()
  # CHECK-NEXT: return %0
  return takeIndexParam[returnIndex()]()

# CHECK-LABEL: lit.fn @"callInParam[def[::SIMD[DType.int, 1]](::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]]()"
# CHECK-SAME: <callable: !lit.generator<<"x": !Int>(!Int, |) -> !alias_Int1>>() -> !alias_Int1
def callInParam[callable: def[x: Int](Int) thin -> Int]() -> Int:
  # CHECK-NEXT: %0 = lit.call {{.*}}takeIndexParam{{.*}}()"<:!Int apply({{.*}}bind_params({{.*}}callable, :!Int {:scalar<index> 1}), {:scalar<index> 1})>()
  # CHECK-NEXT: return %0
  return takeIndexParam[callable[1](1)]()

# CHECK-LABEL: lit.fn @"parameterExprs{{.*}}()"
# CHECK-SAME: <a: !Int, a2: !Int>
def parameterExprs[a: Int, a2: Int]():
  # CHECK: lit.alias.decl *"b{{.*}}": !Int = <{:scalar<index> 0}>
  comptime b = a-a
  # CHECK: lit.alias.decl *"c{{.*}}": !Int = <{{.*}}add(#lit.struct.extract<:!Int a, "_mlir_value">, 42){{.*}}>
  comptime c = a+42
  # CHECK: lit.alias.decl *"d{{.*}}": !Int = <{{.*}}mul(#lit.struct.extract<:!Int a, "_mlir_value">, #lit.struct.extract<:!Int a2, "_mlir_value">){{.*}}>
  comptime d = a*a2

##===----------------------------------------------------------------------===##
# Patterns, LValues and RValues
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"patterns()
def patterns():
  # CHECK: %z2 = lit.var.decl "z2" var
  var z2: Int

  (((z2))) = 42  # Paren patterns
  # CHECK: [[TMP:%.*]] = {{.*}}constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
  # CHECK: lit.ref.store [[TMP]], %z2

  var someInt : Int
  (someInt) += someInt
  # CHECK: %someInt = lit.var.decl "someInt" var
  # CHECK:  %2 = lit.ref.load %1
  # CHECK:   = lit.call {{.*}}SIMD::@"__iadd__{{.*}}(%3, %2)

  # Discard pattern with different types.
  (_) = someInt
  (_) = 1.0

  # CHECK: %someFloat32 = lit.var.decl "someFloat32" var
  # CHECK-NEXT: [[TMP:%.*]] = kgen.rebind %someFloat32
  # CHECK: [[Float32:%.*]] = lit.ref.load [[TMP]]
  # CHECK-NEXT: [[TMP2:%.*]] = kgen.rebind %someFloat32
  # CHECK: {{%.*}} = lit.call {{.*}}__iadd__{{.*}}([[TMP2]], [[Float32]])
  var someFloat32 : Float32
  (someFloat32) += someFloat32

  # CHECK: %someSIMD = lit.var.decl "someSIMD" var
  # CHECK: [[SIMD:%.*]] = lit.ref.load %someSIMD
  # CHECK: {{%.*}} = lit.call {{.*}}@SIMD::@"__iadd__({{.*}}(%someSIMD, [[SIMD]])
  var someSIMD : SIMD[.float64, 4]
  (someSIMD) += someSIMD

# CHECK-LABEL: lit.fn @"byval_byref_function(::SIMD[DType.int, 1],::SIMD[DType.int, 1])"{{.*}}(%a: !Int, %b: !lit.ref<!Int, mut {{.*}}> mut) -> !kgen.none
def byval_byref_function(a: Int, mut b: Int):
  # CHECK-NEXT: lit.ref.store %a, %b
  b = a

  # CHECK-NEXT: %x = lit.var.decl "x" var
  var x : Int
  # This needs to load 'b' to pass it by value for the first arg, but pass its
  # address in directly for the second.
  # CHECK: [[TMP:%.*]] = lit.ref.load %b
  # CHECK: = lit.call {{.*}}::@"byval_byref_function{{.*}}([[TMP]], %b)
  byval_byref_function(b, b)

# CHECK-LABEL: lit.fn @"lvaluesAndRValues()
def lvaluesAndRValues() -> __mlir_type.index:
  # CHECK: [[VALUE:%.*]] = kgen.param.constant = <4>
  # CHECK: lit.return [[VALUE]] : index
  return Int(4).__mlir_index__()

# CHECK-LABEL: lit.fn @"mvalueStructField()"
def mvalueStructField():
  # CHECK: lit.alias.decl [[INT:.*]]: !Int = <{:scalar<index> 4}>
  comptime Index = Int(4)
  # CHECK: lit.alias.decl *"value{{.*}} = <sugar_preserved(#lit.struct.extract<{{.*}}, 4)>
  comptime value = Index._mlir_value
  comptime foldToValue = Int(5)._mlir_value

##===----------------------------------------------------------------------===##
# Augmented Assignments
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"basic_assignments
def basic_assignments(a0: Int, b: Int, c: RegPassable, d: RegPassable):
  var a = a0
  # CHECK-NEXT:      %a = lit.var.decl "a" var
  # CHECK-NEXT: lit.ref.store %a0, %a
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__iadd__{{.*}}(%a, %b)
  a += b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__isub__{{.*}}(%a, %b)
  a -= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__imul__{{.*}}(%a, %b)
  a *= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__ifloordiv__{{.*}}(%a, %b)
  a //= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__imod__{{.*}}(%a, %b)
  a %= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__ipow__{{.*}}(%a, %b)
  a **= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__irshift__{{.*}}(%a, %b)
  a >>= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__ilshift__{{.*}}(%a, %b)
  a <<= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__iand__{{.*}}(%a, %b)
  a &= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__ixor__{{.*}}(%a, %b)
  a ^= b
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__ior__{{.*}}(%a, %b)
  a |= b

  var x: Int
  # CHECK-NEXT: %x = lit.var.decl
  # CHECK-NEXT: %[[FOUR:.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 4})>
  # CHECK-NEXT: lit.ref.store %[[FOUR]], %x
  # CHECK-NEXT: %[[FOURI:.*]] = kgen.rebind %[[FOUR]]
  # CHECK-NEXT: lit.ref.store %[[FOURI]], %a
  a = x = 4

  # Walrus into an MLValue yields the stored value, not a re-read of the LHS.
  # CHECK-NEXT: %[[SEVEN:.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 7})>
  # CHECK-NEXT: lit.ref.store %[[SEVEN]], %x
  # CHECK-NEXT: %[[SEVENI:.*]] = kgen.rebind %[[SEVEN]]
  # CHECK-NEXT: %[[A:.*]] = lit.ref.load %a
  # CHECK-NEXT: lit.call {{.*}}simpleMath{{.*}}(%[[A]], %[[SEVENI]])
  _ = simpleMath(a, x := 7)

# Walrus into an MLValue stores the RHS and yields it.
# CHECK-LABEL: lit.fn @"walrus_assign
def walrus_assign() raises:
  # CHECK: %a = lit.var.decl "a"
  # CHECK: %b = lit.var.decl "b"
  # CHECK: %c = lit.var.decl "c"
  # CHECK: %d = lit.var.decl "d"
  var a: Int
  var b: Int
  var c: Int
  var d: Int

  # CHECK: [[THREE:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 3})>
  # CHECK-NEXT: lit.ref.store [[THREE]], %a
  # CHECK-NEXT: [[THREEI:%.*]] = kgen.rebind [[THREE]]
  # CHECK-NEXT: [[AREF:%.*]] = kgen.rebind %a
  # CHECK-NEXT: [[VAR_A:%.*]] = lit.ref.load [[AREF]]
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}simpleMath{{.*}}([[THREEI]], [[VAR_A]])
  _ = simpleMath(a := 3, a)
  # CHECK-NEXT: lit.ownership.use [[TMP]]

  # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 4})>
  # CHECK-NEXT: lit.ref.store [[FOUR]], %b
  if b := 4:
    b = simpleMath(b, b)

  # CHECK: [[FIVE:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 5})>
  # CHECK-NEXT: lit.ref.store [[FIVE]], %c
  # CHECK-NEXT: lit.ref.store [[FIVE]], %d
  d = c := 5

##===----------------------------------------------------------------------===##
# Literals
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"literals
def literals() raises:
    var a = 5             # CHECK: 5
    a = 55            # CHECK: 55
    a = 10500         # CHECK: 10500
    a = 12_500        # CHECK: 12500
    a = 0             # CHECK: 0
    a = 00            # CHECK: 0
    a = 0____0__0_0   # CHECK: 0
    a = 0__           # CHECK: 0
    a = 00__0_0       # CHECK: 0
    a = 1__9_         # CHECK: 19
    a = 0x123         # CHECK: 291
    a = 0X123         # CHECK: 291
    a = 0b10101       # CHECK: 21
    a = 0B10101       # CHECK: 21
    a = 0o711         # CHECK: 457
    a = 0O711         # CHECK: 457
    # Test parsing for this value with lots of underscores here because mblack
    # can't handle it.
    comptime b = 1_2.3__1e+1_1 # CHECK: #pop.float_literal<1231000000000|1>
    var c = False         # CHECK: !Bool = <{:scalar<bool> false}>
    c = True          # CHECK: !Bool = <{:scalar<bool> true}>

# CHECK-LABEL: lit.fn @"_strings
def _strings():
   """
      Various tests on strings
   """

    var a = ""                 # CHECK: ""
    # CHECK: "hello world"
    var a2 = "hello \
world"

    # COM: match newline hex values via regex since they vary between OSs
    # CHECK: "hello \\{{[\\0-9A-Z]+}}world"
    var a3 = r"hello \
world"

    # CHECK:  "1'{{(\\0D)?}}\0A2"
    var a4 = """1'
2"""

    # CHECK:  "1\222"
    var a5 = '''1"\
2'''

    # CHECK:   "1\22{{(\\0D)?}}\0A2"
    var a6 = '''1"
2'''

    # CHECK:   "1\22\0A2"
    var a7 = '1"\n2'

    # CHECK: "hello concat world"
    var a8 = "hello " "concat " "world"

    var a9 = "Hello"            # CHECK: "Hello"
    var a10 = "Hello 'world'"    # CHECK: "Hello 'world'"
    var a11 = "A\x42"            # CHECK: "AB"
    var a12 = "A\x423"           # CHECK: "AB3"
    var a13 = "A\102"            # CHECK: "AB"
    var a14 = "A\1023"           # CHECK: "AB3"
    # \xhh and \ooo denote Unicode code points (Python str semantics). Values
    # >= 0x80 expand to their UTF-8 encoding rather than a single raw byte,
    # so `\x85` is U+0085 (NEL), encoded as bytes C2 85.
    var a11b = "\x85"            # CHECK: "\C2\85"
    var a11c = "\xff"            # CHECK: "\C3\BF"
    var a13b = "\205"            # CHECK: "\C2\85"

    # COM: the MLIR textual representation escapes strings, so below \ is \\ and " is \"
    var a15 = 'Hello "world"'    # CHECK: "Hello \22world\22"
    var a16 = r"A\x42"           # CHECK: "A\\x42"
    var a17 = R"A\x42"           # CHECK: "A\\x42"
    var a18 = r"AB\\"            # CHECK: "AB\\\\"
    var a19 = r"A\x"             # CHECK: "A\\x"
    var a20 = "AB\\"             # CHECK: "AB\\"
    var a21 = r"A\"B"            # CHECK: "A\\\22B"
    var a22 = r'A\'B'            # CHECK: "A\\'B"
    var a23 = "A\"B"             # CHECK: "A\22B"
    var a24 = 'A\'B'             # CHECK: "A'B"
    var a25 = r"A\zB"            # CHECK: "A\\zB"

    # Issue #201: https://github.com/modular/mojo/issues/201
    def hello() -> StaticString:
        return "123"
    # expected-warning @+1 {{'StringLiteral["other comment"]' value is unused; assign to '_' to discard the result}}
    """other comment"""


##===----------------------------------------------------------------------===##
# Computed Properties and Subscripts
##===----------------------------------------------------------------------===##

# This is an array that has elements of MemoryOnlyInt.
struct MemoryOnlyIntArray(Movable where False):
  def __getitem__(mut self, x: Int) -> MemoryOnlyInt: pass
  def __setitem__(mut self, x: Int, var value: MemoryOnlyInt): pass

# CHECK-LABEL: lit.fn @"testMemoryOnlyIntArray
def testMemoryOnlyIntArray(mut arr: MemoryOnlyIntArray, x: Int, var moi: MemoryOnlyInt):
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %moi)
  arr[x] = moi^
  # CHECK: [[ANON:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, %__call_result_tmp__
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %__call_result_tmp__
  arr[x] = arr[x]

  # CHECK: [[ANON:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-SAME: : !lit.ref<!MemoryOnlyInt, mut *"__call_result_tmp__`
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[ANON]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[ANON]][x]
  # CHECK: %[[C1:.*]] = {{.*}}constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
  # CHECK: lit.ref.store %[[C1:.*]], [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x].x = 1

  # Initialize in memory through a temp + setitem.
  # CHECK: [[ANON:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK: lit.call {{.*}}__init__{{.*}}({{.*}}, [[ANON]])
  # CHECK: lit.call {{.*}}"__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x] = MemoryOnlyInt(42)

  noop() # CHECK: lit.call {{.*}}noop{{.*}}

  # This is yuck, we're rematerializing the base for the rewrite back multiple
  # times: see the "Generalizing Mojo Writeback to Refs" doc in notion.

  # CHECK: [[STORETMP:%.*]] = lit.var.decl "__call_result_tmp__" {{.*}} : !lit.ref<!MemoryOnlyInt,
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[STORETMP]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[STORETMP]][x]
  # CHECK: lit.ref.load [[XP]]
  # CHECK: lit.call {{.*}}SIMD::@"__iadd__
  # CHECK: [[STORETMP:%.*]] = lit.var.decl "__call_result_tmp__" {{.*}} : !lit.ref<!MemoryOnlyInt,
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[STORETMP]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[STORETMP]][x]
  # CHECK: lit.ref.store {{.*}}, [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[STORETMP]])
  arr[x].x += 1

# Walrus into a DLValue copies the RHS into the setter, then yields the
# original RHS (moved) as the RValue.
struct StrArray(Movable where False):
  def __getitem__(mut self, x: Int) -> String: pass
  def __setitem__(mut self, x: Int, var value: String): pass

# CHECK-LABEL: lit.fn @"walrus_dlvalue
def walrus_dlvalue(mut arr: StrArray, x: Int, var str: String) raises:
  # Copy of the RHS into the setter's `var` argument.
  # CHECK: [[STR_IMM:%.*]] = lit.ref.immut %str
  # CHECK: [[SETTER_TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK: lit.call {{.*}}String{{.*}}__init__{{.*}}copy{{.*}}([[STR_IMM]], [[SETTER_TMP]])
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[SETTER_TMP]])
  # Move the original RHS into the result binding.
  # CHECK: [[Y:%.*]] = lit.var.decl "y"
  # CHECK: lit.call {{.*}}String{{.*}}__init__{{.*}}move{{.*}}(%str, [[Y]])
  var y = arr[x] := str^

# CHECK-LABEL: lit.struct.decl @MyInlineIntInit
struct MyInlineIntInit(Movable where False):
    var value: MemoryOnlyInt
    # CHECK-LABEL: lit.fn @"__init__(expressions::MemoryOnlyInt)"
    # CHECK-SAME: (%value: !lit.ref<!MemoryOnlyInt, imm {{.*}}> imm_mem, ?, %self: !lit.ref<!MyInlineIntInit, mut {{.*}}> byref_result) -> !kgen.none
    @implicit
    def __init__(out self, value: MemoryOnlyInt):
        # CHECK: %0 = lit.ref.struct.ger %self[value]
        # CHECK: lit.call {{.*}}__init__{{.*}}"{{.*}}(%value, %0){{.*}}*, "copy"
        self.value = value

struct ConstDynamicObject(RegisterPassable):
    def __init__(out self):
        return

    def __getattr__(self, name: StringLiteral) -> Int:
        return 0

struct DynamicObject(Movable where False):
    def __init__(out self):
        pass

    def __getattr__(self, name: StringLiteral) -> Int:
        return 0

    def __setattr__(self, name: StringLiteral, value: Int):
        pass


# CHECK-LABEL: lit.fn @"dynamic_attribute()"
def dynamic_attribute():
    # CHECK: %const_obj = lit.var.decl "const_obj"
    var const_obj = ConstDynamicObject()
    # CHECK: call {{.*}}@ConstDynamicObject::@"__getattr__{{.*}}<:string "dynamic_attribute">(
    _ = const_obj.dynamic_attribute

    var obj = DynamicObject()
    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: call {{.*}}@DynamicObject::@"__getattr__{{.*}}<:string "some_attr">([[IMMREF]],
    var a = obj.some_attr

    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: %[[VALUE:.*]] = kgen.param.constant: !Int = <{:scalar<index> 42}>
    # CHECK: call {{.*}}@DynamicObject::@"__setattr__{{.*}}<:string "some_attr">([[IMMREF]], {{.*}}, %[[VALUE]])
    obj.some_attr = 42


struct CallableStruct(Movable where False):
    var value: Int

    @implicit
    def __init__(out self, value: Int):
        self.value = value

    def __call__(self, rhs: Int) -> Int:
        return self.value + rhs

# CHECK-LABEL: lit.fn @"test_call_method()"
def test_call_method():
    # CHECK: %[[C2:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: lit.call {{.*}}@"__call__{{.*}}(%{{.*}}, %[[C2]])
    var value = CallableStruct(5)
    _ = value(2)

struct MemoryType(Movable where False):
  def __init__(out self, *, copy: Self):
    pass

struct RegType(RegisterPassable): pass

# CHECK-LABEL: lit.struct.decl @ParamType
# CHECK-SAME: <a: !Int>
struct ParamType[a: Int](TrivialRegisterPassable): pass

# CHECK-LABEL: lit.fn @"function_types
def function_types():
  # CHECK: lit.alias.decl *"p0{{.*}}<<"a": !Int>(!lit.struct<#ParamType <:!Int *(0,0)>{{.*}}>, |) -> !kgen.none
  comptime p0 = def[a: Int](ParamType[a]) thin -> None

  # CHECK: lit.alias.decl *"p1{{.*}}<<"a": !Int, "b": {{.*}}#ParamType <:!Int *(0,0)>>>[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>
  comptime p1 = def[a: Int, b: ParamType[a]]() thin raises -> None

  # CHECK: lit.alias.decl *"p2{{.*}}"Ts": !lit.struct<#TypeList{{.*}} pos_vararg{{.*}}(!lit.ref<{{.*}}#VariadicPack
  # CHECK-SAME: <:!Bool {:scalar<bool> false},  :origin<false> *(0,2){{.*}}, :meta<!AnyType> !AnyType, :param_list<!AnyType> *(0,0), :!Bool {:scalar<bool> false}, {{.*}}>>, imm *[0,0]>
  # CHECK-SAME: imm_mem|pack_vararg, ?, "__result__": !lit.ref<none, mut *[0,1]> byref_result) async
  comptime p2 = async def[*Ts: AnyType](* *Ts) thin -> None

  # CHECK: lit.var.decl "float0"{{.*}}(!Int, |) -> !alias_Int1
  var float0: def(Int) thin -> Int

  # CHECK: lit.var.decl "float1"{{.*}}(!lit.ref<!MemoryType, imm {{.*}}> imm_mem, |, ?, "__result__": !lit.ref<!MemoryType, mut {{.*}}> byref_result) -> !kgen.none
  var float1: def(MemoryType) thin -> MemoryType

  # CHECK: lit.var.decl "float2"{{.*}}(!lit.ref<!RegType, mut *[0,0]> owned_in_mem, |) -> !RegType
  var float2: def(var RegType) thin -> RegType

  # CHECK: lit.var.decl "float3"{{.*}}(!lit.ref<!MemoryType, mut *[0,0]> owned_in_mem, |) -> !kgen.none
  var float3: def(var MemoryType) thin -> None

  # CHECK: lit.var.decl "float4"{{.*}}(!lit.ref<!Int, mut *[0,0]> mut, |) -> !kgen.none
  var float4: def(mut Int) thin -> None

  # CHECK: lit.var.decl "float5"{{.*}}(!Int, |, ?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>
  var float5: def(Int) thin raises -> None

  # CHECK: lit.var.decl "float6"{{.*}}(!Int, |, ?, "__result__": !lit.ref<none, mut *[0,0]> byref_result) async|capturing -> !kgen.none
  var float6: async def(Int) capturing thin -> None

  # CHECK: lit.var.decl "float7"{{.*}}(!lit.ref<!lit.struct<#VariadicList <:!Bool {:scalar<bool> false}, :origin<false> *(0,0), :!lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> *(0,0)>> *(0,1), :!AnyType !Int, :!Bool {:scalar<bool> false}>>, imm *[0,0]> imm_mem|pos_vararg, ?, {{.*}}) throws -> !kgen.scalar<bool>
  var float7: def(*Int) thin raises -> None

  # CHECK: lit.var.decl "float12"{{.*}}<(!Int = {:scalar<index> 10}, {{.*}}StringLiteral <:string "foo">
  # CHECK-SAME: , |) -> !kgen.none>
  var float12: def(Int = 10, StaticString = "foo") thin -> None

  # CHECK: lit.var.decl "named"{{.*}}<[1]("x": !lit.ref<!MemoryType, imm {{.*}}> imm_mem) -> !alias_Int1>
  var named: def(x: MemoryType) thin -> Int

# CHECK-LABEL: lit.struct.decl @Mem
# CHECK:         lit.alias.decl *"x{{.*}}": non_struct_type = <i8>
# CHECK-NEXT:    lit.alias.decl *"B{{.*}}": non_struct_type = <!lit.generator<("foo": !kgen.param<:non_struct_type sugar_member_alias(!Mem, "x", i8)>) -> !kgen.none>>
struct Mem(Movable where False):
   comptime x = __mlir_type.i8
   comptime B = def (foo: Self.x) thin -> None

comptime def_type_alias = def() thin -> None

@always_inline
def func_with_decorator(): pass

struct TwoParamsStruct[a: Int, b: Int](ImplicitlyCopyable):
    pass

# CHECK-LABEL: lit.fn @"variadic_subscript{{.*}}"<{{.*}}param_list<!Int>, +, idx: !Int,
# CHECK-SAME: a: !lit.struct<#ParameterList <:!AnyType !Int, :param_list<!Int> *"a.values`">> pos_vararg
def variadic_subscript[idx: Int, *a: Int](*b: Int):
    # CHECK: lit.alias.decl *"v0{{.*}}": {{.*}}Int = <#kgen.param_list.get<:param_list<!Int> *"a.values`", 2>>
    comptime v0 = a[2]

    # CHECK: %v1 = lit.var.decl "v1"
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <#kgen.param_list.get<:param_list<!Int> *"a.values`", 3>>
    # CHECK: lit.ref.store [[TMP]], %v1
    var v1 = a[3]
    # CHECK: {{.*}}__getitem__{{.*}}(%b, %{{.*}})
    var v2 = b[idx]


# CHECK-LABEL: lit.fn @"variadic_memory_subscript
# CHECK-SAME: !lit.ref<{{.*}}TwoParamsStruct
# CHECK-SAME:   #kgen.param_list.get<:param_list<!Int> *"a.values`", 0>
# CHECK-SAME:   #kgen.param_list.get<:param_list<!Int> *"a.values`", 1>
def variadic_memory_subscript[*a: Int](*b: TwoParamsStruct[a[0], a[1]]):
    # CHECK: [[B1REF:%.*]] = {{.*}}__getitem__{{.*}}(%b,
    # CHECK: %v0 = lit.var.decl
    # CHECK: lit.memcpy [[B1REF]], %v0

    var v0 = b[1]
    # CHECK: [[B2REF:%.*]] = {{.*}}__getitem__{{.*}}(%b,
    # CHECK: %v1 = lit.var.decl
    # CHECK: lit.memcpy [[B2REF]], %v1
    var v1 = b[2]

def testTransferWarning():
  var a = MemoryOnlyInt()

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(a^^)

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(MemoryOnlyInt()^)


##===----------------------------------------------------------------------===##
# Test nonmaterializable IntLiteral beyond Int bounds.
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl *"bigggNumber{{.*}}#IntLiteral <:!pop.int_literal 115792089237316195423570985008687907853269984665640564039457584007913129639936>> = <{}>
comptime bigggNumber = 2 << 255
def useBigNumber() -> Int:
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 512})>
  var notSoBig = bigggNumber // (2 << 246)
  # Easy min-Index
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> -9223372036854775808})>
  var minInt = -(2<<62)
  return notSoBig

struct IndexList[size: Int](TrivialRegisterPassable):
    @implicit
    def __init__(out self, *elements: Int):
        pass

    def __getitem__(self) -> Int:
      pass

    def __setitem__(mut self, val: Int):
        pass

# Issue 23233 https://github.com/modularml/modular/issues/23233
def setitemParamToDLValue():
  comptime x = 3
  var coords = IndexList[3](0)
  # The main check is just that it's not erroring.
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <{:scalar<index> -3}>
  # CHECK: lit.call {{.*}}IndexList{{.*}}__setitem__{{.*}}[[VAR]]
  coords[] = -x

# https://github.com/modular/mojo/issues/734
def reg_passable_trivial():
  var x : Int = 100
  x = 42
  _ = x^  # expected-warning {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}

  var y : Int = 100
  # expected-warning @+1 {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}
  _ = y^  # Consume RValue / BValue is not, this isn't tracked.

def del_warnings():
  # These copy the value before destroying it, which is pointless.
  var m = MemoryOnlyInt()
  m.__deinit__()  # expected-warning {{explicit call to '__deinit__' destroys a copy of the value; consider removing this call}}
  var r = RegPassable(1)
  r.__deinit__()  # expected-warning {{explicit call to '__deinit__' destroys a copy of the value; consider removing this call}}

  # These is weird/unneeded, but at least it does what it says.
  MemoryOnlyInt().__deinit__()
  RegPassable(1).__deinit__()

##===----------------------------------------------------------------------===##
# Parameter inference
##===----------------------------------------------------------------------===##

# Test that parameter inference can handle this.
def dependent_call_it[dtype: DType](ptr: UnsafePointer[SIMD[dtype, 1], AnyOrigin[mut=True]]):
   dependent_callee(ptr, 0.0)
# This requires substitution to realize that storage.type.type == DType
def dependent_callee[dtype: DType](storage: UnsafePointer[SIMD[dtype, 1], AnyOrigin[mut=True]],
                   pad_value: SIMD[storage.type.dtype, 1]):
   pass

# This requires handling of ParamListAttr in parameter inference.
def variadic_attr_caller(*inputs: Tuple[Int]):
   variadic_attr_callee[Int](*inputs)
def variadic_attr_callee[key_type: ImplicitlyCopyable](*inputs: Tuple[key_type]):
  pass

# Test that parameter inference works with implicit conversions - in this case
# that we can infer the parameters of 'thing_taking_reference' even though x
# needs to be built as a Pointer.
def thing_taking_ref[
  type: AnyType,
  //,
  origin: Origin
](ref [origin] arg: type): pass

def thing_taking_ref2[type: AnyType](ref arg: type): pass

def thing_taking_pointer2[type: AnyType](arg: Pointer[type, _]): pass

# CHECK-LABEL: lit.fn @"test_thing_taking_reference
def test_thing_taking_reference(mut x: String):
  # CHECK-NEXT: lit.call {{.*}}thing_taking_ref{{.*}}(%x)
  thing_taking_ref(x)
  # CHECK-NEXT: lit.call {{.*}}thing_taking_ref2{{.*}}(%x)
  thing_taking_ref2(x)
  # CHECK-NEXT: lit.call {{.*}}@Pointer::@"__init__
  thing_taking_pointer2(Pointer(to=x))

struct StructWithStaticMethods(Movable where False):
   @staticmethod
   def _init_op_state(state: Pointer[Int, _], foo: Int): pass
   def thing(self):
     var x = 42
     Self._init_op_state(Pointer(to=x), x)

def infer_through_alias():
  comptime MyType = MemoryOnlyInt
  _ = MyType(4)


# CHECK-LABEL: lit.fn @"infer_address_space
def infer_address_space[
    mut: __mlir_type.i1,
    //,
    origin: Origin[mut=mut]
](a: Pointer[Int, origin, address_space=AddressSpace(4)]._mlir_lit_ref):
  # Show that we can infer the address space parameter of Pointer from a
  # !lit.ref.

  # The ref is rebound to match the `to:` parameter origin before
  # `Pointer.__init__`.
  # CHECK: kgen.rebind %a
  # CHECK-NEXT: lit.call {{.*}}@Pointer::@"__init__{{.*}}({{%.*}})
  var x = Pointer(to=__get_litref_as_mvalue(a))


# https://linear.app/modularml/issue/MOCO-584/[references]-we-cannot-bind-litref-in-parameter-context
# [References] We cannot bind !lit.ref in parameter context
struct ThingWithMethodReferenceSelf(Movable where False):
    def method(ref a: Self):
      pass

# CHECK-LABEL: lit.fn @"testThingWithMethodReferenceSelf
def testThingWithMethodReferenceSelf[a: ThingWithMethodReferenceSelf]():
    # CHECK-NEXT: lit.alias.decl *"sizzle`": none =
    # CHECK-SAME: <apply(:!lit.generator<("a": !lit.ref<!ThingWithMethodReferenceSelf,
    # CHECK-SAME:     <:scalar<bool> false, :origin<false> #lit.comptime.origin>, store_to_mem(a))>
    comptime sizzle = a.method()

struct HasOverloadedFooMethods(Movable where False):
    def foo(ref self): pass
    def foo(var self): pass

# CHECK-LABEL: lit.fn @"testHasOverloadedFooMethods
def testHasOverloadedFooMethods():
    var foo: HasOverloadedFooMethods
    # CHECK: lit.call {{.*}}@HasOverloadedFooMethods::@"foo{{.*}}(%foo){{.*}}mut *"foo`"> ref) -> !kgen.none>
    foo.foo()
    # CHECK: lit.call {{.*}}@HasOverloadedFooMethods::@"foo{{.*}}(%foo){{.*}}owned_in_mem) -> !kgen.none>
    foo^.foo()
