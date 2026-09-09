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

def use_string(arg: String): pass

def has_default_args(a: Int, b: Int = 1, c: Int = 2):
    pass


# CHECK-LABEL: lit.fn @"test_kw_arg_passing
def test_kw_arg_passing(x: Int, y: Int, z: Int):
    # CHECK: %[[C2:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %[[C2]])
    has_default_args(x, b=y)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(x, b=y, c=z)

    # CHECK: %[[C1:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %[[C1]], %z)
    has_default_args(x, c=z)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(x, c=z, b=y)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(a=x, c=z, b=y)

    # CHECK: call {{.*}}@"has_default_args{{.*}}"(%x, %y, %z)
    has_default_args(c=z, b=y, a=x)


# CHECK-LABEL: lit.fn @"test_kw_arg_passing_indirect
def test_kw_arg_passing_indirect[callee: def(a: Int, b: Int=1, c: Int=2) thin -> None](x: Int, y: Int, z: Int):
    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: lit.call tail[{{.*}}](%x, %[[C1]], %z)
    callee(x, c=z)

    # CHECK-NEXT: lit.call tail[{{.*}}](%x, %y, %z)
    callee(c=z, b=y, a=x)

def has_default_params[a: Int, b: Int = 1, c: Int = 2]():
    pass


# CHECK-LABEL: lit.fn @"test_kw_param_passing
def test_kw_param_passing[x: Int, y: Int, z: Int]():
    # CHECK: lit.call {{.*}}@"has_default_params{{.*}}"<:!Int x, :!Int y, :!Int {:scalar<index> 2}>
    has_default_params[x, b=y]()

    # CHECK: lit.call {{.*}}@"has_default_params{{.*}}"<:!Int x, :!Int y, :!Int z>
    has_default_params[x, b=y, c=z]()

    # CHECK: lit.call {{.*}}@"has_default_params{{.*}}"<:!Int x, :!Int {:scalar<index> 1}, :!Int z>
    has_default_params[x, c=z]()

    # CHECK: lit.call {{.*}}@"has_default_params{{.*}}"<:!Int x, :!Int y, :!Int z>
    has_default_params[x, c=z, b=y]()

    # CHECK: lit.call {{.*}}@"has_default_params{{.*}}"<:!Int x, :!Int y, :!Int z>
    has_default_params[a=x, c=z, b=y]()

    # CHECK: lit.call {{.*}}@"has_default_params{{.*}}"<:!Int x, :!Int y, :!Int z>
    has_default_params[c=z, b=y, a=x]()


# CHECK-LABEL: lit.fn @"test_kw_param_passing_indirect
def test_kw_param_passing_indirect[x: Int, y: Int, z: Int,
                                  callee: def[a: Int, b: Int=1, c: Int=2]() thin -> None]():

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, :!Int x, :!Int {:scalar<index> 1}, :!Int z)]()
    callee[x, c=z]()

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, :!Int x, :!Int y, :!Int z)]()
    callee[c=z, b=y, a=x]()


@fieldwise_init
struct MyCallable(Movable where False):
    def __call__(self, m: Int, n: Int = 2):
        pass


# CHECK-LABEL: lit.fn @"test_callable_object
def test_callable_object(x: Int, y: Int):
    # CHECK: %[[CALLABLE:.*]] = lit.var.decl {{.*}}: !lit.ref<!MyCallable
    var callable = MyCallable()

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %x, %[[C2]])
    callable(x)

    # CHECK-DAG: %[[IMMREF:.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-NEXT: call {{.*}}@MyCallable::@"__call__{{.*}}(%[[IMMREF]], %y, %x)
    callable(n=x, m=y)


def takes_kw_only_args(a: Int, b: Int = 1, *, c: Int, d: Int = 2):
    pass


# CHECK-LABEL: lit.fn @"test_kw_only_args
def test_kw_only_args(x: Int):
    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %[[C1]], %x, %[[C2]])
    takes_kw_only_args(x, c=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %[[C1]], %x, %[[C2]])
    takes_kw_only_args(c=x, a=x)

    # CHECK: %[[C2:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %x, %x, %[[C2]])
    takes_kw_only_args(x, c=x, b=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %[[C1]], %x, %x)
    takes_kw_only_args(x, d=x, c=x)

    # CHECK: lit.call {{.*}}@"takes_kw_only_args{{.*}}"(%x, %x, %x, %x)
    takes_kw_only_args(d=x, b=x, c=x, a=x)


# CHECK-LABEL: lit.fn @"test_kw_only_indirect
def test_kw_only_indirect[callee: def(a: Int, b: Int = 1, *, c: Int, d: Int = 2) thin -> None](x: Int):

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-DAG: %[[C2:.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: lit.call tail[{{.*}}](%x, %[[C1]], %x, %[[C2]])
    callee(x, c=x)

    # CHECK-DAG: %[[C1:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: lit.call tail[{{.*}}](%x, %[[C1]], %x, %x)
    callee(x, d=x, c=x)


def takes_kw_only_params[a: Int, b: Int = 1, *, c: Int, d: Int = 2]():
    pass


# CHECK-LABEL: lit.fn @"test_kw_only_params
def test_kw_only_params[x: Int]():
    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<:!Int x, :!Int {:scalar<index> 1}, :!Int x, :!Int {:scalar<index> 2}>()
    takes_kw_only_params[x, c=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<:!Int x, :!Int {:scalar<index> 1}, :!Int x, :!Int {:scalar<index> 2}>()
    takes_kw_only_params[c=x, a=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<:!Int x, :!Int x, :!Int x, :!Int {:scalar<index> 2}>()
    takes_kw_only_params[x, c=x, b=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<:!Int x, :!Int {:scalar<index> 1}, :!Int x, :!Int x>()
    takes_kw_only_params[x, d=x, c=x]()

    # CHECK: call {{.*}}takes_kw_only_params{{.*}}"<:!Int x, :!Int x, :!Int x, :!Int x>()
    takes_kw_only_params[d=x, b=x, c=x, a=x]()


# CHECK-LABEL: lit.fn @"test_kw_only_params_indirect
def test_kw_only_params_indirect[x: Int, callee: def[a: Int, b: Int = 1, *, c: Int, d: Int = 2]() thin -> None]():

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, :!Int x, :!Int {:scalar<index> 1}, :!Int x, :!Int {:scalar<index> 2})]()
    callee[x, c=x]()

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, :!Int x, :!Int {:scalar<index> 1}, :!Int x, :!Int x)]()
    callee[x, d=x, c=x]()


def takes_variadic_and_kw_only_args(
    a: Int, b: Int, *args: Int, c: Int, d: Int = 0
):
    pass


# CHECK-LABEL: lit.fn @"test_variadic_and_kw_only_args
def test_variadic_and_kw_only_args(x: Int):
    # CHECK: [[VARIADIC:%.*]] = kgen.param.constant: !lit.ref<array<0, !lit.ref<!Int, imm {}>>, imm {}> = <#interp.pointer<0>>
    # CHECK-NEXT: [[T1:%.*]] = lit.call {{.*}}VariadicList::@"__init__{{.*}}([[VARIADIC]])
    # CHECK-NEXT: [[TMPVD:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[T1]], [[TMPVD]]
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.immut [[TMPVD]]
    # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK-NEXT: lit.call {{.*}}@"takes_variadic_and_kw_only_args{{.*}}"{{.*}}(%x, %x, [[T2]], %x, [[ZERO]])
    takes_variadic_and_kw_only_args(x, x, c=x)

    # CHECK: [[VARIADIC:%.*]] = kgen.param.constant: !lit.ref<array<0, !lit.ref<!Int, imm {}>>, imm {}> = <#interp.pointer<0>>
    # CHECK-NEXT: [[T1:%.*]] = lit.call {{.*}}VariadicList::@"__init__{{.*}}([[VARIADIC]])
    # CHECK-NEXT: [[TMPVD:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[T1]], [[TMPVD]]
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.immut [[TMPVD]]
    # CHECK-NEXT: lit.call {{.*}}@"takes_variadic_and_kw_only_args{{.*}}"{{.*}}(%x, %x, [[T2]], %x, %x)
    takes_variadic_and_kw_only_args(x, x, d=x, c=x)

    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [{{.*}}]
    # CHECK: [[T1:%.*]] = lit.call {{.*}}VariadicList::@"__init__
    # CHECK-NEXT: [[TMPVD:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[T1]], [[TMPVD]]
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.immut [[TMPVD]]
    # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK-NEXT: lit.call {{.*}}@"takes_variadic_and_kw_only_args{{.*}}"{{.*}}(%x, %x, [[T2]], %x, [[ZERO]])
    takes_variadic_and_kw_only_args(x, x, x, x, c=x)


def takes_variadic_and_kw_only_params[
    a: Int, b: Int, *args: Int, c: Int, d: Int = 0
]():
    pass


# CHECK-LABEL: lit.fn @"test_variadic_and_kw_only_params
def test_variadic_and_kw_only_params[x: Int]():
    # CHECK: lit.call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<:param_list<!Int> [], :!Int x, :!Int x, {{.*}}, :!Int x, :!Int {:scalar<index> 0}>()
    takes_variadic_and_kw_only_params[x, x, c=x]()

    # CHECK: lit.call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<:param_list<!Int> [], :!Int x, :!Int x, {{.*}}, :!Int x, :!Int x>()
    takes_variadic_and_kw_only_params[x, x, d=x, c=x]()

    # CHECK: lit.call {{.*}}takes_variadic_and_kw_only_param{{.*}}"<:param_list<!Int> [x, x], :!Int x, :!Int x, {{.*}}, :!Int x, :!Int {:scalar<index> 0}>()
    takes_variadic_and_kw_only_params[x, x, x, x, c=x]()


# CHECK-LABEL: lit.fn @"test_variadic_and_kw_only_params_indirect
def test_variadic_and_kw_only_params_indirect[x: Int,
    callee: def [a: Int, b: Int, *args: Int, c: Int, d: Int = 0]() thin -> None]():

    # CHECK: lit.call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, :param_list<!Int> [], :!Int x, :!Int x, {{.*}}, :!Int x, :!Int {:scalar<index> 0})]()
    callee[x, x, c=x]()

    # CHECK: call{{.*}}bind_params(:!lit.generator<{{.*}}> callee, :param_list<!Int> [x, x], :!Int x, :!Int x, {{.*}}, :!Int x, :!Int {:scalar<index> 0})]()
    callee[x, x, x, x, c=x]()


## Complex address space support

# Passing non-default address space through Self in an initializer.


# CHECK-LABEL: lit.fn @"initialize_in_addrspace
def initialize_in_addrspace(
    ptr: UnsafePointer[ExampleRegPassable, AnyOrigin[mut=True], address_space=AddressSpace(1)]
):

    # Get !lit.ref in addr space #1
    # CHECK-NEXT: [[PTRREF:%.*]] = lit.call{{.*}}@Pointer::@"__getitem__{{.*}}(%ptr)

    # CHECK-NEXT: [[REGVAL:%.*]] = lit.call {{.*}}@ExampleRegPassable::@"__init__{{.*}}()

    # Use lit.ref.store to move into addrspace 1
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[PTRREF]] : <!ExampleRegPassable, mut #lit.any.origin, 1>
    ptr[] = ExampleRegPassable()


struct SomeRefItemStruct(Movable where False):
    def __getitem__(self) -> ref [self] Int:
        pass


# CHECK-LABEL: lit.fn @"test_param_refitem
def test_param_refitem[a: SomeRefItemStruct]():
    # CHECK-NEXT: !alias_Int1 = <load_from_mem(:!lit.ref<:meta<!Int> #alias_Int, imm #lit.comptime.origin> apply(:{{.*}}SomeRefItemStruct::@"__getitem__
    comptime x = a[]


# Passing non-default address space through mut arg, must use temporary.
# CHECK-LABEL: lit.fn @"mutate_in_addrspace
def mutate_in_addrspace(
    a: ExampleRegPassable,
    ptr: UnsafePointer[ExampleRegPassable, AnyOrigin[mut=True], address_space=AddressSpace(1)],
):
    # Get !lit.ref in addr space #1
    # CHECK-NEXT: [[PTRREF:%.*]] = lit.call {{.*}}@Pointer::@"__getitem__{{.*}}(%ptr)

    # Use a temporary to get an MLValue in the default address space.
    # CHECK-NEXT: [[REGVAL:%.*]] = lit.ref.load [[PTRREF]] : <!ExampleRegPassable, mut #lit.any.origin, 1>
    # CHECK-NEXT: %anonymous2A = lit.var.decl "anonymous
    # CHECK-NEXT: lit.ref.store [[REGVAL]], %anonymous2A
    # CHECK-NEXT: lit.call {{.*}}@ExampleRegPassable::@"mutateArg{{.*}}(%a, %anonymous2A)

    # Use lit.load/store to move back into addrspace 1
    # CHECK-NEXT: [[REGVAL:%.*]] = lit.load.consume %anonymous2A
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[PTRREF]] : <!ExampleRegPassable, mut #lit.any.origin, 1>
    a.mutateArg(ptr[])


struct ExampleRegPassable(TrivialRegisterPassable):
    def __init__(out self):
        pass

    def mutateArg(self, mut other: Self):
        pass


## Partial Binding of Function Symbols With Implicit Parameters


struct Matrix[rows: Int, cols: Int](Movable where False):
    pass


def matmul_unrolled[I: Int](mut C: Matrix):
    pass


@always_inline
def test_matrix_equal[
    func: def (mut: Matrix) thin -> None
](mut C: Matrix) raises -> Bool:
    func(C)
    return True


# CHECK-LABEL: lit.fn @"partialBind
def partialBind(mut C: Matrix[1, 2]) raises:
    # CHECK-NEXT: %exp = lit.var.decl "exp
    # CHECK-NEXT: lit.call {{.*}}::@"test_matrix_equal{{.*}}"[mut *"C`{{.*}}", mut *"__error__`{{.*}}", mut *"exp`{{.*}}"]
    # CHECK-SAME: <:!lit.generator<<?, ".rows`2x": !Int, ".cols`2x1": !Int>[1](!lit.ref<!lit.struct<#Matrix <:!Int *(0,0), :!Int *(0,1)>>, mut *[0,0]> mut, |) -> !kgen.none>
    # CHECK-SAME: rebind(:!lit.generator<<?, "C.rows`": !Int, "C.cols`1": !Int>[1]("C": !lit.ref<!lit.struct<#Matrix <:!Int *(0,0), :!Int *(0,1)>>, mut *[0,0]> mut) -> !kgen.none>
    # CHECK-SAME: @{{.*}}::@"matmul_unrolled{{.*}}"<:!Int {:scalar<index> 0}, :!Int ?, :!Int ?>), :!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>(%C, %__error__, %exp)
    var exp = test_matrix_equal[matmul_unrolled[0]](C)


# MOCO-692: [mojo-lang][ownership] Implicit conversion failure
# CHECK-LABEL: lit.fn @"test_implicit_conversion_bvalue
def test_implicit_conversion_bvalue():
    # CHECK-NEXT: %foo = lit.var.decl
    # CHECK-NEXT: Struct1::@"__init__
    var foo = Struct1()
    # CHECK-NEXT: lit.ownership.use %foo
    # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}Struct2::@"__init__
    # CHECK-NEXT: lit.ref.immut
    # CHECK-NEXT: lit.call {{.*}}take_struct2
    take_struct2(foo^)


struct Struct1(Movable where False):
    def __init__(out self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass


struct Struct2(Movable where False):
    @implicit
    def __init__(out self, var foo: Struct1):
        pass


def take_struct2(bar: Struct2):
    pass


def pack_it[*Ts: AnyType](*args: *Ts) -> String:
    return String()


def also_broken(r: Pointer[String, _]) -> String:
    return r[]


# MOCO-858: isSafeToUseValueDestForDirectResult doesn't handle aliasing through references
# CHECK-LABEL: lit.fn @"test_byref_slot_with_references
def test_byref_slot_with_references():
    var f = String()

    # CHECK: [[RESULTTMP:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lit.call {{.*}}pack_it{{.*}}({{.*}},  [[RESULTTMP]])
    f = pack_it(f)
    # CHECK-NEXT: lit.call {{.*}}String::@"__init__{{.*}}"{{.*}}([[RESULTTMP]], %f){{.*}}*, "move"

    # CHECK: [[RESULTTMP:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lit.call {{.*}}also_broken{{.*}}({{.*}},  [[RESULTTMP]])
    f = also_broken(Pointer(to=f))
    # CHECK-NEXT: lit.call {{.*}}String::@"__init__{{.*}}"{{.*}}([[RESULTTMP]], %f){{.*}}*, "move"

    # CHECK: [[RESULTTMP:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lit.call {{.*}}also_broken{{.*}}({{.*}},  [[RESULTTMP]])
    f = also_broken(Pointer(to=f))
    # CHECK-NEXT: lit.call {{.*}}String::@"__init__{{.*}}"{{.*}}([[RESULTTMP]], %f){{.*}}*, "move"


# CHECK-LABEL: lit.fn @"test_byref_slot_closure_capture
def test_byref_slot_closure_capture(var x: String):
    # CHECK: lit.fn *"capture
    @__parameter
    def capture() -> String:
        return x

    # CHECK: %__call_result_tmp__
    # CHECK-NEXT: lit.call[{{.*}}: *"capture{{.*}}(%__call_result_tmp__)
    x = capture()
    # CHECK-NEXT: lit.call {{.*}}@String::@"__init__{{.*}}"{{.*}}(%__call_result_tmp__, %x){{.*}}*, "move"


def test_int_ref(ref x: Int) -> ref [x] Int:
    return x


# CHECK-LABEL: lit.fn @"complex_ref_box_emission
def complex_ref_box_emission[p: Int](a: Int):
    # Parameter ref just needs a box.
    _ = test_int_ref(p)
    # CHECK: [[VAR:%.*]] = lit.var.decl {{.*}}!lit.ref<!Int,
    # CHECK-NEXT: kgen.param.constant: !Int = <p>
    # CHECK-NEXT: lit.ref.store {{.*}}, [[VAR]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[VAR]]
    # CHECK-NEXT: lit.call {{.*}}test_int_ref{{.*}}([[TMP]])

    # Needs a conversion from IntegerLiteral to Int
    _ = test_int_ref(4)
    # CHECK: [[VAR:%.*]] = lit.var.decl {{.*}}!lit.ref<!Int,
    # CHECK-NEXT: kgen.param.constant: !Int = <{:scalar<index> 4}>
    # CHECK-NEXT: lit.ref.store {{.*}}, [[VAR]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[VAR]]
    # CHECK-NEXT: lit.call {{.*}}test_int_ref{{.*}}([[TMP]])

    # RValues infer as immutable, just like you can't pass them to mut.
    _ = test_int_ref(Int())
    # CHECK: [[REGVAL:%.*]] = lit.call {{.*}}SIMD::@"__init__{{.*}}()
    # CHECK-NEXT: [[VAR:%.*]] = lit.var.decl {{.*}}!lit.ref<!Int,
    # CHECK-NEXT: lit.ref.store [[REGVAL]], [[VAR]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[VAR]]
    # CHECK-NEXT: lit.call {{.*}}test_int_ref{{.*}}([[TMP]])

    # TODO: Should work fine; needs generalized writeback.
    # _ = test_int_ref(a)
    # _ = test_int_ref(a+a)

# MOCO-1440 - Weird conditional conformance mismatch
struct ThingWithParam[X: Int](Movable where False):
  @implicit
  def __init__(out self: ThingWithParam[42], other: Bool): pass

def test_cond_conformance(exclude: Bool):
    comptime local_alias = 42
    var ptr : UnsafePointer[ThingWithParam[local_alias], AnyOrigin[mut=True]]
    ptr[] = exclude


# MOCO-1442: Unnecessary copies being generated from owned values in constructors
@fieldwise_init  # This is copyable, but we don't want to.
struct Heavy(ImplicitlyCopyable):
  pass

# This is intended to be a lightweight view of Heavy.
struct ViewOfHeavy(Movable where False):
  @implicit
  def __init__(out self, h: Heavy): pass

def takeOwnedValue(var view: ViewOfHeavy): pass

# CHECK-LABEL: lit.fn @"testUnneededCopy
def testUnneededCopy(heavy: Heavy):
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl
  # CHECK-NEXT: lit.call {{.*}}ViewOfHeavy::@"__init__{{.*}}(%heavy, [[TMP]])
  # CHECK-NEXT: lit.call {{.*}}takeOwnedValue
  takeOwnedValue(heavy)
  # CHECK-NEXT: kgen.param.constant: none


# Check that field sensitivity is properly field sensitive.
struct NonCopyable(Movable where False): pass
def take_and_return(a: NonCopyable) -> NonCopyable: pass
# CHECK-LABEL: lit.struct.decl @TestFieldSensitiveResultSlot
struct TestFieldSensitiveResultSlot(Movable where False):
    var a: NonCopyable
    var b: NonCopyable

    # CHECK: lit.fn @"__init__
    def __init__(out self):
        # CHECK-NEXT: [[B:%.*]] = lit.ref.struct.ger %self[b]
        # CHECK-NEXT: [[A:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT: [[AI:%.*]] = lit.ref.immut [[A]]
        # CHECK-NEXT: lit.call {{.*}}take_and_return{{.*}}([[AI]], [[B]])
        # Should be in-place without a copy/move
        self.b = take_and_return(self.a)

# Check that imm origin binding works with partially applied functions (which
# get bound to a function pointer then called indirectly.
struct SomeStructWithRefMethod(Movable where False):
    def take_ref(ref self) -> SomeStructWithRefMethod: pass

# CHECK-LABEL: lit.fn @"testSomeStructWithRefMethod
def testSomeStructWithRefMethod[val: SomeStructWithRefMethod]():
    comptime f = SomeStructWithRefMethod.take_ref
    # CHECK: lit.alias.decl *"b`1":
    # CHECK-SAME: <:scalar<bool> false, :origin<false> #lit.comptime.origin>), store_to_mem(val))>
    comptime b = f(val)



# COM: Avoid copies when unnecessary

@fieldwise_init
struct MyDictEntry[K: KeyElement & Deinitable, V: Copyable & Deinitable](Copyable):
    var key: Self.K
    var value: Self.V


struct MyDict[K: KeyElement & Deinitable, V: Copyable & Deinitable](Copyable):
    var _entries: List[MyDictEntry[Self.K, Self.V]]

    def __getitem__(
        ref self, key: Self.K
    ) raises -> ref [self._entries[0].value] Self.V:
        ref entry = self._entries[0]
        return entry.value

    def __setitem__(mut self, var key: Self.K, var value: Self.V):
        pass

@fieldwise_init
struct Value(ImplicitlyCopyable):
    var max_width: Int
    def mutate(mut self): pass

# CHECK-LABEL: lit.fn @"entry
def entry(mut value: MyDict[String, Value], name: String) raises:
    # CHECK: lit.call {{.*}}::@MyDict::@"__getitem__
    # CHECK-NEXT: [[V1:%.*]] = lit.load.consume
    # CHECK-NEXT: [[V2:%.*]] = lit.ref.struct.ger [[V1]]
    # CHECK-NEXT: [[V3:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 12})>
    # CHECK-NEXT: lit.ref.store [[V3]], [[V2]]
    value[name].max_width = 12


# getitem on mutable list returns a mutable ref.
struct MyMutGetItemCollection[T: AnyType](Movable where False):
    var state: Value
    def clear(mut self): pass
    def __init__(out self): pass
    def __getitem__(ref self, idx: Int) -> ref [self] Self.T: pass
    def __setitem__(mut self, idx: Int, var value: Self.T): pass

# CHECK-LABEL: lit.fn @"subscript_assignment_inplace
def subscript_assignment_inplace(mut list: MyMutGetItemCollection[MyMutGetItemCollection[Int]]):
    # CHECK: [[V0:%.*]] = lit.call {{.*}}::@MyMutGetItemCollection::@"__getitem__
    # CHECK-NEXT: lit.call {{.*}}::@MyMutGetItemCollection::@"clear{{.*}}"[{{.*}}]<{{.*}}>([[V0]])
    list[0].clear()

    # Make sure this isn't copying 'state'.
    # CHECK: [[T1:%.*]] = lit.call {{.*}}MyMutGetItemCollection::@"__getitem__
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.struct.ger [[T1]][state]
    # CHECK-NEXT: lit.call {{.*}}Value::@"mutate{{.*}}([[T2]])
    list[0].state.mutate()

    # Mutating the element directly should call the setitem, not use a reference
    # returned by a mutating getter.
    # CHECK: [[VALUETMP:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK: lit.call {{.*}}MyMutGetItemCollection::@"__init__{{.*}}([[VALUETMP]])
    # CHECK: lit.call {{.*}}MyMutGetItemCollection::@"__setitem__{{.*}}(%list, {{.*}}, [[VALUETMP]])
    list[0] = MyMutGetItemCollection[Int]()

    # Mutating the inner part of a computed LValue should use the reference
    # returned by the getitem, because the getitem is guaranteed to have been
    # called either way.
    # CHECK: [[VALUETMP:%.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!Value,
    # CHECK: lit.call {{.*}}Value::@"__init__
    # CHECK: [[T1:%.*]] = lit.call {{.*}}MyMutGetItemCollection::@"__getitem__
    # CHECK-NEXT: [[T2:%.*]] = lit.ref.struct.ger [[T1]][state]
    # CHECK-NEXT: lit.call {{.*}}Value::@"__init__{{.*}}"{{.*}}([[VALUETMP]], [[T2]]){{.*}}*, "move"
    list[0].state = Value(1)

# getitem on mutable list returns a immutable ref so setitem is needed.
struct MyImmutGetItemCollection[T: AnyType](Movable where False):
    def __getitem__(self, idx: Int) -> ref [self] Self.T: pass
    def __setitem__(mut self, idx: Int, var value: Self.T): pass

# CHECK-LABEL: lit.fn @"subscript_assignment_writeback
def subscript_assignment_writeback(mut list: MyImmutGetItemCollection[Value]):
    # This MUST perform a copy of the value, because getitem returns an
    # immutable ref.

    # CHECK: lit.call {{.*}}::@MyImmutGetItemCollection::@"__getitem__
    # CHECK: lit.call {{.*}}Value::@"mutate
    # CHECK: lit.call {{.*}}::@MyImmutGetItemCollection::@"__setitem__
    list[0].mutate()


# CHECK-LABEL: lit.fn @"check_tail_call
def check_tail_call(str_arg: String, var var_str_arg: String):
    var local_str = String()

    # This touches the local stack temporary so can't be a tail call.
    # CHECK: lit.call @{{.*}}@"use_string{{.*}}local_str
    use_string(local_str)

    # These touch memory that isn't in this frame, so they can be.
    # CHECK: lit.call tail @{{.*}}@"use_string{{.*}}var_str_arg
    use_string(var_str_arg)
    # CHECK: lit.call tail @{{.*}}@"use_string{{.*}}(%str_arg)
    use_string(str_arg)


# These take values that have to be in memory.  Verify that we're forming the
# temp boxes correctly even though we're inferring parameters with them.
def test_mem_temp_1(ref x: Int, *, y: Int) -> ref[x] Int:
    return x

def test_mem_temp_2(*args: Int, ref x: Int) -> ref[x] Int:
    return x

# CHECK-LABEL: lit.fn @"test_mem_temp_caller
def test_mem_temp_caller():
    # CHECK: lit.call {{.*}}test_mem_temp_1{{.*}}: !lit.generator<("x": !lit.ref<!Int, [[TMPORIGIN:.*]]> ref, *, "y": !Int) refresult -> !lit.ref<:meta<!Int> #alias_Int, [[TMPORIGIN]]>>
    _ = test_mem_temp_1(y=4, x=12)
    # CHECK: lit.call {{.*}}test_mem_temp_2{{.*}}"x": !lit.ref<!Int, [[TMPORIGIN:.*]]> ref) refresult -> !lit.ref<:meta<!Int> #alias_Int, [[TMPORIGIN]]>>
    _ = test_mem_temp_2(x=5)


def test_var_varargs_inference():
    var x = String()
    var y = String()
    # The list ctor has a 'var' varargs, forcing a copy of x/y.  The copy result
    # has a different origin than x/y.
    var z = [x, y]

def nonmat_callee(x: Int): pass
def nonmat_callee[T: AnyType](x: T): pass
# CHECK-LABEL: lit.fn @"test_nonmat_callee
def test_nonmat_callee():
    # This should call the concrete nonmat_callee, not the generic one.
    # CHECK: lit.call {{.*}}@"nonmat_callee(::SIMD[DType.int, 1])"
    nonmat_callee(1)

# CHECK-LABEL: lit.fn @"test_singleton_fnptr_params
def test_singleton_fnptr_params():
  var someInt : Int

  # CHECK: [[FP1:%.*]] = lit.ref.load %fp1
  var fp1: def[a: MutOrigin](ref [a] x: Int) thin -> None
  # CHECK-NEXT: [[FP1BOUND:%.*]] = lit.bind_params [[FP1]] :
  # CHECK-SAME: :!lit.struct<#Origin <:!Bool {{{.*}} true}, :origin<true> *"someInt`">>
  # CHECK-NEXT: [[SOMEINT:%.*]] = kgen.rebind %someInt {{.*}} to !lit.ref<!Int, mut *"someInt`">
  # CHECK-NEXT: lit.call_indirect [[FP1BOUND]]([[SOMEINT]])
  fp1(someInt)

  # Need to bind the origin of the variadic list.
  var fp2: def(*Int) thin -> None
  # CHECK: [[FP2:%.*]] = lit.ref.load %fp2
  # CHECK: [[FP2BOUND:%.*]] = lit.bind_params [[FP2]] :
  # CHECK: lit.call_indirect [[FP2BOUND]]
  fp2(someInt, someInt)


@fieldwise_init
struct TestMemType:
    pass

def no_exclusivity_violation[mut: Bool, o: Origin[mut=mut], //](ref [o] x: TestMemType, imm y : TestMemType):
    pass


# CHECK-LABEL: lit.fn @"test()"
def test():
    var x = TestMemType()
    # Binding 'mut' to False makes the 'ref' argument an immutable access, so
    # aliasing it with the 'imm' argument is not an exclusivity violation.
    # CHECK: [[X1:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: [[X2:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: lit.call {{.*}}@"no_exclusivity_violation{{.*}}[muttoimm *"x`"]<:!Bool {:scalar<bool> false}{{.*}}([[X1]], [[X2]])
    no_exclusivity_violation[mut = False](x, x)
