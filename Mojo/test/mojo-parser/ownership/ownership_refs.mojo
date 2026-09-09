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

# Test more advanced reference cases.

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s

def use_any[*Ts: AnyType](*args: *Ts): pass

# ===----------------------------------------------------------------------=== #
# Parsing of references
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample(ImplicitlyCopyable):
  def __init__(out self): pass
  def __init__(out self, *, deinit move: Self): pass
  def __init__(out self, *, copy: Self): pass
  def __deinit__(deinit self): pass
  def noop(self): pass
  def mutate(mut self): pass

# CHECK-LABEL: lit.fn @"borrow{{.*}}"<{{.*}}>(%a: !lit.ref<!MemExample, imm *"lt._mlir_origin`">
def borrow[lt: ImmOrigin](a: Pointer[MemExample, lt]._mlir_lit_ref):
  pass

# CHECK-LABEL: lit.fn @"mutate{{.*}}"<{{.*}}>(%a: !lit.ref<!MemExample, mut *"lt._mlir_origin`">
def mutate[lt: MutOrigin](a: Pointer[MemExample, lt]._mlir_lit_ref):
  pass

# CHECK-LABEL: lit.fn @"implicit_borrow
def implicit_borrow(a: MemExample):
  pass

# CHECK-LABEL: lit.fn @"implicit_inout
def implicit_inout(mut a: MemExample):
  pass

# CHECK-LABEL: lit.fn @"implicit_owned
def implicit_owned(var a: MemExample):
  pass

# This preserves reference mutability
# CHECK-LABEL: lit.fn @"parametricMut
# CHECK-SAME: (%a: !lit.ref<!MemExample, mut=#lit.struct.extract<:!Bool *"o.mut`", "_mlir_value">, *"o._mlir_origin`1">) ->
# CHECK-SAME: !lit.ref<!MemExample, mut=#lit.struct.extract<:!Bool *"o.mut`", "_mlir_value">, *"o._mlir_origin`1">
def parametricMut[o: Origin](a: Pointer[MemExample, o]._mlir_lit_ref)
   -> Pointer[MemExample, o]._mlir_lit_ref:
  return a

# CHECK-LABEL: lit.fn @"testParametricMut
def testParametricMut(i: MemExample, mut m: MemExample):
  # This infers an immutable reference.
  # CHECK:  lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, imm *"i`">
  _ = parametricMut(__get_mvalue_as_litref(i))

  # This infers a mutable reference.
  # CHECK: lit.call {{.*}}parametricMut{{.*}}!lit.ref<!MemExample, mut *"m`1">
  _ = parametricMut(__get_mvalue_as_litref(m))

##===----------------------------------------------------------------------===##
# Conditional origins
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"testUseConditional
def testUseConditional(cond: __mlir_type.`!kgen.scalar<bool>`):
  # CHECK-NOT: __deinit__

  # CHECK: lit.call {{.*}}__init__{{.*}}(%a)
  var a = MemExample()

  # CHECK: lit.call {{.*}}__init__{{.*}}(%b)
  var b = MemExample()

  # CHECK: %cptr = lit.var.decl "cptr"
  var cptr = Pointer(to=a) if cond else Pointer(to=b)

  # This uses both A and B, so it needs to extend both of their origins.
  cptr[].noop()
  # CHECK: [[TMP:%.*]] = kgen.rebind %cptr
  # CHECK: [[CV:%.*]] = lit.ref.load [[TMP]]
  # CHECK-NEXT: lit.var.lifetime.end %cptr
  # CHECK-NEXT: [[MREF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[CV]])
  # CHECK-NEXT: lit.ref.immut [[MREF]]
  # CHECK-NEXT: lit.call {{.*}}noop
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%b)
  # CHECK-NEXT: lifetime.end %b
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a

# CHECK-LABEL: lit.fn @"testDefConditional
def testDefConditional(cond: __mlir_type.`!kgen.scalar<bool>`):
  # CHECK-NOT: lit.call {{[^)]*}}__deinit__

  var a = MemExample()
  var b = MemExample()

  # CHECK: %cptr = lit.var.decl "cptr"
  var cptr = Pointer(to=a) if cond else Pointer(to=b)


  # Mutating either of these is fine - it doesn't matter which one is mutated,
  # we know that both are live.
  cptr[].mutate()
  # CHECK: [[TMP:%.*]] = kgen.rebind %cptr
  # CHECK: [[CP:%.*]] = lit.ref.load [[TMP]]
  # CHECK-NEXT: [[MREF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[CP]])
  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}([[MREF]])

  # Overwriting one means that we need to immediately destroy the same reference
  # because we cannot know which one is being set.
  cptr[] = MemExample()
  # CHECK: [[TMP:%.*]] = kgen.rebind %cptr
  # CHECK-NEXT: [[CP:%.*]] = lit.ref.load [[TMP]]
  # CHECK-NEXT: [[MREF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[CP]])
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[MREF]])

  # Overwriting is eligible for copy => move optimization as well.
  var shouldBeMovedFrom = MemExample()
  # CHECK: lit.call {{.*}}__init__{{.*}}(%shouldBeMovedFrom)
  cptr[] = shouldBeMovedFrom
  # CHECK: [[TMP:%.*]] = kgen.rebind %cptr
  # CHECK-NEXT: [[CP:%.*]] = lit.ref.load [[TMP]]
  # CHECK-NEXT: lit.var.lifetime.end %cptr
  # CHECK-NEXT: [[MREF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[CP]])
  # CHECK-NEXT: lit.ref.immut
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[MREF]])
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}move"
  # CHECK-NEXT: lifetime.end %shouldBeMovedFrom

  # The mutation above could either of A or B, so we needed to extend both of
  # their origins, but now we can say goodbye.
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%b)
  # CHECK-NEXT: lifetime.end %b

  # A use so the assignment isn't dead.
  a.noop()
  # CHECK-NEXT: [[ATMP:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[ATMP]])
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a

# ===----------------------------------------------------------------------=== #
# Tests of the Pointer type.
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"testUseConditionalReference

def testUseConditionalReference(cond: __mlir_type.`!kgen.scalar<bool>`, immArg: MemExample):
  # CHECK: %a = lit.var.decl {{.*}} : !lit.ref<!MemExample, mut *"a`1">
  # CHECK: lit.call {{.*}}__init__{{.*}}(%a)

  var a = MemExample()

  # CHECK: lit.call @std::@builtin::@stubs::@Pointer::@"__init__{{.*}}(%a)
  var aref = Pointer(to=a)
  # CHECK: lit.alias.decl *"aLifetime{{.*}}": origin<true> = <*"a`1">
  comptime aLifetime =  aref.origin._mlir_origin

  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[AR]])
  aref[].noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[REF]]
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])

  # This is a mutable reference so go head and store through it whynot?
  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[AR]])
  aref[] = MemExample()
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[REF]])
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])

  # The reference being alive doesn't keep the underlying stuff alive, only
  # accesses
  # CHECK-NEXT: %aref2 = lit.var.decl "aref2"
  # CHECK-NEXT: [[AR:%.*]] = lit.ref.load %aref
  # CHECK-NEXT: lifetime.end %aref
  # CHECK-NEXT: lifetime.start %aref2
  # CHECK-NEXT: lit.ref.store [[AR]], %aref2
  # CHECK-NEXT: lifetime.end %aref2
  # expected-warning @+1 {{assignment to 'aref2' was never used}}
  var aref2 = aref

  # Ok, this was the last use of A so it can go away.
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%a)
  # CHECK-NEXT: lifetime.end %a

  # Pointer can bind to immutable things as well, no problem.
  # CHECK-NEXT: [[IMMRV:%.*]] = lit.call tail @std::@builtin::@stubs::@Pointer::@"__init__{{.*}}(%immArg)
  # CHECK-NEXT: %immref = lit.var.decl "immref"
  # CHECK: lit.ref.store [[IMMRV]], %immref
  var immref = Pointer(to=immArg)
  immref[].noop()

# ===----------------------------------------------------------------------=== #
# Test that we can bind self origin.
# ===----------------------------------------------------------------------=== #

# Need a way to get a origin of Self.
# https://github.com/modularml/modular/issues/29069

struct SelfRefTest(ImplicitlyCopyable):
  def __init__(out self): pass
  def __deinit__(deinit self): pass
  # CHECK-LABEL: lit.fn @"method
  # CHECK-SAME: (%self: !lit.ref<!SelfRefTest
  def method(ref self) -> Pointer[Self, origin_of(self)]:
      return Pointer(to=self)

# CHECK-LABEL: lit.fn @"testSelfRef
def testSelfRef(a: SelfRefTest, mut b: SelfRefTest):
  # Bind immutably to a
  # CHECK: = lit.call {{.*}}method{{.*}}<:scalar<bool> false, :origin<false> *"a`">(%a)
  _ = a.method()

  # Bind mutably to b
  # CHECK: = lit.call {{.*}}method{{.*}}<:scalar<bool> true, :origin<true> *"b`1">(%b)
  _ = b.method()


# CHECK-LABEL: lit.fn @"testLifetimeOf1
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> imm_mem) ->
# CHECK-SAME: !lit.struct<#Pointer <{{.*}}:origin<false> *"a`">> {}, :!AddressSpace {_value: !SIMDLength = {0}}>>
def testLifetimeOf1(a: MemExample) -> Pointer[MemExample, origin_of(a)]:
  return Pointer(to=a)

# CHECK-LABEL: lit.fn @"testLifetimeOf2
# CHECK-SAME: (%a: !lit.ref<!MemExample, imm *"a`"> imm_mem) ->
# CHECK-SAME: !lit.ref<!MemExample, imm *"a`">
def testLifetimeOf2(a: MemExample) -> Pointer[MemExample, origin_of(a)]._mlir_lit_ref:

  # CHECK: kgen.return {{.*}} : !lit.ref<!MemExample, imm *"a`">
  return __get_mvalue_as_litref(a)

# CHECK-LABEL: lit.fn @"callByRefResultLifetime
def callByRefResultLifetime(mut x: MemExample, mut y: MemExample, z: MemExample):
  # CHECK: lit.var.decl "l1" var : !lit.ref<{{.*}}(mutcast mut *"x`")
  var l1 = returnOneArgLifetime(x)

  # CHECK: lit.var.decl "l2" var : !lit.ref<{{.*}}#TwoLifetimes <:origin<false> (mutcast mut *"x`"), :origin<false> (mutcast mut *"y`1"),
  var l2 = returnTwoArgLifetimes(x, y)
  # CHECK: %l3 = lit.var.decl "l3" var : !lit.ref<{{.*}}#TwoLifetimes <:origin<false> (mutcast mut *"x`"), :origin<false> (mutcast mut *"x`"),
  var l3 = returnTwoArgLifetimes(x, x)
  # CHECK: %l4 = lit.var.decl "l4" var : !lit.ref<{{.*}}#TwoLifetimes <:origin<false> *"z`2", :origin<false> *"z`2",
  var l4 = returnTwoArgLifetimes(z, z)

  use_any(l1, l2, l3, l4)

def returnOneArgLifetime(a: MemExample)
  -> OneLifetime[origin_of(a)]:
  return OneLifetime[origin_of(a)]()

def returnTwoArgLifetimes(a: MemExample, b: MemExample)
  -> TwoLifetimes[origin_of(a), origin_of(b)]:
  return TwoLifetimes[origin_of(a), origin_of(b)]()

struct OneLifetime[a_origin: ImmOrigin](Movable where False):
  def __init__(out self): pass

struct TwoLifetimes[a_origin: ImmOrigin,
                    b_origin: ImmOrigin](Movable where False):
  def __init__(out self): pass

# Test that we can infer the type of 'T' in the func param invocation.
# CHECK-LABEL: CutDownVariadicPack
struct CutDownVariadicPack[element_trait: type_of(AnyType), //,
                           *element_types: element_trait](Movable where False):

    # CHECK: lit.fn @"each_hack
    def each_hack[i: Int, func: def[T: Self.element_trait] (T) thin -> None](self):
        # Test that we can infer the type of 'T' from the argument.
        # CHECK-NEXT: [[REFVAL:%.*]] = lit.call {{.*}}get_element{{.*}}(%self)
        # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}}([[REFVAL]])
        # CHECK-NEXT: lit.call{{.*}} func,
        # CHECK-SAME: :!kgen.param<:meta<!AnyType> element_trait> #kgen.param_list.get<:param_list<:meta<!AnyType> element_trait>
        # CHECK-SAME: element_types{{.*}}([[REF]])
        func(self.get_element[i]()[])

    def get_element[index: Int](self) -> Pointer[
        Self.element_types[index],
        origin_of(self)]:
       while True: pass

# Test that you can implicitly convert an "any" mutable reference (as is returned
# by Pointer for example) to mortal reference with specified origin.
# CHECK: lit.fn @"test_immortal_to_mortal
def test_immortal_to_mortal(arg: Pointer[Int, _])
    -> Pointer[Int, arg.origin]:
  # CHECK-NEXT: [[ARGREF:%.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}}(%arg)
  # CHECK-NEXT: [[PTRVAL:%.*]] = lit.call {{.*}}Pointer::@"__init__{{.*}}([[ARGREF]])
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}}([[PTRVAL]])
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}@Pointer::@"__init__{{.*}}([[REF]])
  # CHECK-NEXT: kgen.return [[RES]]
  return Pointer[Int, arg.origin](to=Pointer(to=arg[])[])


# CHECK-LABEL: lit.fn @"ref_copyability
def ref_copyability[*element_types: ImplicitlyCopyable & Deinitable](*args: *element_types):
  # CHECK: [[ITEM:%.*]] = lit.call tail @std::@builtin::@stubs::@VariadicPack::@"__getitem_param__
  # CHECK: %_x = lit.var.decl
  # CHECK: lit.call[!lit.generator<[2](*, "copy"{{.*}}#kgen.get_witness<{{.*}}__init__{{.*}}([[ITEM]], %_x)
  var _x = args[4]

  # CHECK-NEXT: lit.call[{{.*}}#kgen.get_witness<{{.*}}__deinit__{{.*}}(%_x)

# Issue #37659: Parameter inference doesn't work with force-immut origins

# FIXME (Patch #48185): need to support implicit conversions to immutable reference.

#def thing_taking_immutable_ref[T: AnyType, value_origin: Origin[]](a: Pointer[T, value_origin]): pass
#def test_passing_mutable_ref(mut i: String):
#    thing_taking_immutable_ref(Pointer(to=i))

# Verify that we can propagate parametric mutability through field accesses.
struct ThingWithFields(Movable where False):
  var field: Int

# CHECK-LABEL: lit.fn @"parametric_mut_mbvalue
def parametric_mut_mbvalue[origin: Origin](a: Pointer[ThingWithFields, origin]) -> Pointer[Int, origin_of(a[].field)]:
  # CHECK: lit.ref.struct.ger
  return Pointer(to=a[].field)

# Pointer directly with inferred params.
struct SomeStructWithReferenceSelfArgument(Movable where False):
    def __init__(out self): pass
    def hello(ref self):
        pass

# CHECK-LABEL: lit.fn @"testMethodRef
def testMethodRef(a: SomeStructWithReferenceSelfArgument):
    # CHECK-NEXT: lit.call {{.*}}@"hello{{.*}}(%a)
    a.hello()



# CHECK-LABEL: lit.fn @"variadic_inout_mems_iter
def variadic_inout_mems_iter(mut *mems: MemExample):
  # Verify the iterator keeps the VariadicList alive.

  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__iter__{{.*}}(%mems)
  # CHECK: %iter = lit.var.decl
  # CHECK-NEXT: lifetime.start %iter
  # CHECK-NEXT: lit.ref.store [[TMP]], %iter
  var iter = mems.__iter__()

  # CHECK-NEXT: %__try_error__ = lit.var.decl
  # CHECK-NEXT: lit.try

  # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl {{.*}} : !lit.ref<!lit.ref<
  # CHECK: lit.call {{.*}}__next__{{.*}}(%iter, {{.*}}, %__call_result_tmp__)

  # Iterator is destroyed as soon as we're done with it.
  # CHECK-NEXT: lifetime.end %iter

  # CHECK: [[ELTREF:%.*]] = lit.load.consume %__call_result_tmp__
  # CHECK-NEXT: lit.var.lifetime.end %__call_result_tmp__

  # Copy the result of __next__ into !lit.ref
  # CHECK-NEXT: %x = lit.var.decl
  # CHECK-NEXT: [[ELTREFIMM:%.*]] = lit.ref.immut [[ELTREF]]
  # CHECK-NEXT: lifetime.start %x
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}copy"
  try:
    var x : MemExample = iter.__next__()

    # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x)
    x.mutate()
  except:
    pass

  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%x)
  # CHECK-NEXT: lifetime.end %x


# CHECK-LABEL: lit.fn @"test_pvalue_ref_formation
def test_pvalue_ref_formation[a: SelfRefTest]():
  # This is invoking a method (accepting a ref) on a pvalue.  This need to
  # materialize into a temporary and use the origin of the temporary, not an
  # immortal origin.

  # CHECK: [[ANONTMP:%.*]] = lit.var.decl "anonymous*" {{.*}}!lit.ref<!SelfRefTest,
  var r = a.method()
  # The result reference should have inferred the origin of the temp
  # CHECK: lit.ref.store {{.*}}, %r : {{.*}}!SelfRefTest, {{.*}}origin<false> (mutcast mut *"anonymous*`

  # This use of the temp should keep it alive.
  # CHECK: [[REFERENCE:%.*]] = lit.ref.load %r
  # CHECK: [[REF:%.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}}([[REFERENCE]])
  # CHECK-NEXT: lit.call {{.*}}method{{.*}}([[REF]])
  _ = r[].method()
  # CHECK-NEXT: lit.call {{.*}}SelfRefTest::@"__deinit__{{.*}}([[ANONTMP]])

# MOCO-1025 - Need hierarchical origins
struct FieldRefPropagation(Movable where False):
  var field1 : Optional[Int]
  var field2 : Int

  def __init__(out self):
     # Should be able to initialize field1 and use it.
     self.field1 = 42
     # Should be able to project it and assign through ref.
     self.field1.value() = 17
     # Then initialize field2
     self.field2 = 1


# Issue #3444 (nightly) Raising init causing use of uninitialized variable
# https://github.com/modular/mojo/issues/3444
struct HasRaisingInit(Copyable):
  def __init__(out self) raises: pass

struct ImmovableRaisingInit(Movable where False):
  def __init__(out self) raises: pass

struct RaisingInitWrapper(Movable where False):
    var field: HasRaisingInit
    var immfield: ImmovableRaisingInit

    def __init__(out self) raises:
      self.field = HasRaisingInit()
      self.immfield = ImmovableRaisingInit()

# CHECK-LABEL: lit.fn @"test_inout_raising_init
def test_inout_raising_init(mut a: HasRaisingInit, mut b: RaisingInitWrapper) raises:
  # These init calls need a temporary instead of direct assignment into the dest
  # to avoid invalidating a value on the error path.
  # CHECK-NEXT: [[TEMP:%.*]] = lit.var.decl
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__init__{{.*}}({{.*}}, [[TEMP]])
  a = HasRaisingInit()
  # EH logic.
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__init__{{.*}}move"

  # CHECK: [[FIELDREF:%.*]] = lit.ref.struct.ger %b[field]
  # CHECK: [[TEMP:%.*]] = lit.var.decl
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__init__{{.*}}({{.*}}, [[TEMP]])
  b.field = HasRaisingInit()
  # EH logic.
  # CHECK: lit.call {{.*}}HasRaisingInit::@"__init__{{.*}}(*, "move":

# CHECK-LABEL: lit.fn @"test_parameter_closure_captures
def test_parameter_closure_captures(var x: MemExample, var y: MemExample):
  # CHECK: lit.fn *"capture
  @__parameter
  def capture():
    _ = x^
    _ = y^

  # CHECK: lit.call[!lit.generator<:{mut *"x`{{.*}}", mut *"y`{{.*}}"}:
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__deinit__{{.*}}(%y)
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__deinit__{{.*}}(%x)
  capture()

def higher_order_function[lts: __mlir_type.`!lit.origin.set`, //, f: def() capturing [lts] -> None]():
  pass

# CHECK-LABEL: lit.fn @"test_higher_order_capture
def test_higher_order_capture(var x: MemExample, var y: MemExample):
  # CHECK: lit.fn *"capture
  @__parameter
  def capture():
    _ = x^
    _ = y^

  # CHECK: lit.call {{.*}}higher_order_function{{.*}} !lit.generator<:{mut *"x`{{.*}}", mut *"y`{{.*}}"}
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__deinit__{{.*}}(%y)
  # CHECK-NEXT: lit.call {{.*}}MemExample::@"__deinit__{{.*}}(%x)
  higher_order_function[capture]()

# CHECK-LABEL: lit.fn @"test_origin_ref_spec
# CHECK-SAME: !lit.ref<!Int, mut *"our_origin._mlir_origin`"> mutref)
def test_origin_ref_spec[our_origin: Origin[mut=True]](ref[our_origin] a: Int):
    pass

# CHECK-LABEL: lit.fn @"another_min
def another_min[mut: Bool, //, ao: Origin[mut=mut], bo: Origin[mut=mut]](ref [ao]a: Int, ref [bo]b: Int) -> ref [a, b] Int:
    if a < b:
        return a
    else: # This failed due to union canonicalization problems.
        return b

struct RefResultStruct(Movable where False):
  var x: Int
  def __init__(out self):  self.x = 1
  def method(self) -> ref [self.x] Int: return self.x

# https://github.com/modular/mojo/issues/3960
# CHECK-LABEL: lit.struct.decl @FieldSensitiveUse
struct FieldSensitiveUse(Movable where False):
    var x: RefResultStruct
    var y: String

    # CHECK: lit.fn @"__init__
    def __init__(out self):
        # CHECK: lit.call {{.*}}RefResultStruct::@"__init__
        self.x = RefResultStruct()
        # CHECK: [[TMP:%.*]] = lit.call {{.*}}RefResultStruct::@"method
        _ = self.x.method()
        # CHECK-NEXT: lit.ownership.use [[TMP]]
        self.y = String()
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[y]
        # CHECK-NEXT: lit.call {{.*}}String::@"__init__{{.*}}([[TMP]])


# MOCO-2077: https://github.com/modular/modular/issues/4705
# CHECK-LABEL: lit.fn @"test_getitem_setitem
def test_getitem_setitem(mut d: TestDict[Int, Int]):
    # This should bind to a mutable reference, not an immutable one.
    # CHECK: %0 = lit.call {{.*}}@TestDict::@"__getitem__
    # CHECK: lit.call {{.*}}@"check_mutability{{.*}}<:!Bool {:scalar<bool> true}, {{.*}}(%0)
    check_mutability(d[])


def check_mutability[
    is_mutable: Bool, //, origin: Origin[mut=is_mutable], T: AnyType
](ref [origin]s: T):
    pass


struct TestDict[K: AnyType, V: Deinitable](Movable where False):
    def __getitem__(ref self) -> ref [self] Self.V:
        while True: pass

    def __setitem__(mut self, var value: Self.V):
        pass

# ===----------------------------------------------------------------------=== #
# Interior origins
# ===----------------------------------------------------------------------=== #

struct MyListInterior[T: Movable](Movable where False):
    var data: UnsafePointer[Self.T, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = UnsafePointer[Self.T, UntrackedOrigin[mut=True]].unsafe_dangling()

    def __deinit__(deinit self):
        pass # Explicit deinit so it isn't considered trivial and elided.

    def mutate(mut self):
        pass

    def __getitem__(
        ref self
    ) -> ref[self.data._get_ref_with_unsafe_interior_origin["element"](self)] Self.T:
      return (self.data)._get_ref_with_unsafe_interior_origin["element"](self)

# CHECK-LABEL: lit.fn @"test_interior0
def test_interior0():
    # CHECK: lit.call {{.*}}MyListInterior::@"__init__
    var list = MyListInterior[Int]()
    # CHECK: lit.call {{.*}}MyListInterior::@"__getitem__
    ref elt = list[]
    # CHECK: lit.call {{.*}}SIMD::@"__iadd__
    elt += 4
    # CHECK: lit.call {{.*}}MyListInterior::@"__deinit__

# ===----------------------------------------------------------------------=== #
# Subtree origins
# ===----------------------------------------------------------------------=== #

trait MyIndexable:
    def __getitem__(ref self, idx: Int) -> ref[origin_of(self).subtree] Int:
        ...

struct MyList(MyIndexable):
    var elements: List[Int]

    # More concrete origin so concrete clients get a tighter origin bound.
    def __getitem__(ref self, idx: Int) -> ref[self.elements[idx]] Int:
        return self.elements[idx]

def test_MyIndexable(mut idxable: Some[MyIndexable]):
    _ = idxable[0]
    idxable[0] += 1

def test_subtree(mut mylist: MyList):
    test_MyIndexable(mylist)

# MOCO-4453: Tests for origin union collapsing.
struct OriginWhereClauseTest:
    var f: Int

    def __init__(out self):
        self.f = 0

    def field_in_subtree(ref self) where origin_of(self).subtree.contains[origin_of(self.f)]:
        pass
    def field_subtree_in_subtree(ref self) where origin_of(self).subtree.contains[origin_of(self.f).subtree]:
        pass

def test_origin_where_clause():
    var s = OriginWhereClauseTest()
    s.field_in_subtree()
    s.field_subtree_in_subtree()
