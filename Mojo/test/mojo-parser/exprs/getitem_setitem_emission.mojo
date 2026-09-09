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


struct WeirdArray(Movable where False):
    def __getitem__(self, x: Int) -> Int:
        return x

    def __getitem__(self, x: Int, y: Int) -> Int:
        return x

    def __getitem__(self, x: Int, y: Int, z: Int) -> Int:
        return x

    def __getitem__(self, x: SIMDLength, *ints: Int) -> Int:
        return 1

    def __setitem__(self, x: Int, y: Int, value: Int):
        pass

    def __getitem__(self, s: Slice) -> Int:
        return 2


# CHECK-LABEL: lit.fn @"test_getitem
def test_getitem(a: WeirdArray, idx: Int, f: SIMDLength):
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
    _ = a[idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx)
    _ = a[idx, idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx, %idx)
    _ = a[idx, idx, idx]

    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [{{.*}}]
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %f, {{.*}})
    _ = a[f, idx, idx, idx, idx]


def test_getitem_kw(a: WeirdArray, idx: Int, idx2: Int, idx3: Int):
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
    _ = a[x=idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2)
    _ = a[y=idx2, x=idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
    _ = a[z=idx3, x=idx, y=idx2]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
    _ = a[idx, z=idx3, y=idx2]


# CHECK-LABEL: lit.fn @"test_setitem
def test_setitem[x: Int](a: WeirdArray, idx: Int):
    # CHECK: %[[X:.*]] = kgen.param.constant: !Int = <x>
    # CHECK: lit.call {{.*}}__setitem__{{.*}}(%a, %idx, %idx, %[[X]])
    a[idx, idx] = x


# CHECK-LABEL: lit.fn @"test_getitem_slice
def test_getitem_slice(a: WeirdArray, i: Int, j: Int, k: Int):
    # CHECK: [[SLICE:%.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value
    # CHECK-SAME: :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value>(%none{{.*}}, %none{{.*}}, %none{{.*}}) :
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, [[SLICE]])
    _ = a[:]

    # CHECK: [[SLICE:%.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value,
    # CHECK-SAME: :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value>{{.*}}(%none{{.*}}, %none{{.*}}, %none{{.*}}) :
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, [[SLICE]])
    _ = a[::]

    # CHECK: [[SLICE:%.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int,
    # CHECK-SAME: :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value>(%i, %j, %none{{.*}}) :
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, [[SLICE]])
    _ = a[i:j]

    # CHECK: [[SLICE:%.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable #type_value,
    # CHECK-SAME: :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int>(%none{{.*}}, %i, %j, {{.*}}) :
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, [[SLICE]])
    _ = a[:i:j]

    # CHECK: [[SLICE:%.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int,
    # CHECK-SAME: :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int>(%i, %j, %k, {{.*}}) :
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, [[SLICE]])
    _ = a[i:j:k]


struct IndexArray(Movable where False):
    def __getitem__(mut self, x: Int) -> Int:
        pass

    def __setitem__(mut self, x: Int, value: Int):
        pass


struct IndexArrayArray(Movable where False):
    def __getitem__(mut self, x: Int) -> IndexArray:
        pass

    def __setitem__(mut self, x: Int, var value: IndexArray):
        pass


def takes_inout_int(mut a: Int):
    pass


# CHECK-LABEL: lit.fn @"test_writeback1
def test_writeback1[x: Int, y: Int](mut a: IndexArray, mut b: IndexArrayArray):
    # CHECK: %[[V0:.*]] = kgen.param.constant: !Int = <x>
    # CHECK-NEXT: %[[V1:.*]] = lit.call {{.*}}__getitem__{{.*}}(%a, %[[V0]])
    # CHECK-NEXT: %[[LT:.*]] = lit.var.decl "anonymous*" synth
    # CHECK-NEXT: lit.ref.store %[[V1]], %[[LT]]
    # CHECK-NEXT: %[[LTR:.*]] = kgen.rebind %[[LT]] {{.*}} to !lit.ref<!Int
    # CHECK-NEXT: lit.call {{.*}}takes_inout_int{{.*}}(%[[LTR]])
    # CHECK-NEXT: %[[LTR2:.*]] = kgen.rebind %[[LT]] {{.*}} to !lit.ref<!Int
    # CHECK-NEXT: %[[V3:.*]] = kgen.param.constant: !Int = <x>
    # CHECK-NEXT: %[[V4:.*]] = lit.ref.load %[[LTR2]]
    # CHECK-NEXT: lit.call {{.*}}__setitem__{{.*}}(%a, %[[V3]], %[[V4]])
    takes_inout_int(a[x])


# CHECK-LABEL: lit.fn @"test_writeback2
def test_writeback2[x: Int, y: Int](mut a: IndexArray, mut b: IndexArrayArray):
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant: !Int = <x>
    # CHECK-NEXT: %[[LT2:.*]] = lit.var.decl {{.*}}!IndexArray
    # CHECK-NEXT: %[[V4:.*]] = {{.*}}__getitem__{{.*}}(%b, %[[C1]], %[[LT2]])
    # CHECK-NEXT: %[[C2:.*]] = kgen.param.constant: !Int = <y>
    # CHECK-NEXT: %[[V5:.*]] = lit.call {{.*}}__getitem__{{.*}}(%[[LT2]], %[[C2]])

    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant: !Int = <x>
    # CHECK-NEXT: lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], %[[LT2]])
    # CHECK-NEXT: %[[LT1:.*]] = lit.var.decl {{.*}}!lit.ref<:meta<!Int> #alias_Int
    # CHECK-NEXT: lit.ref.store %[[V5]], %[[LT1]]
    # CHECK-NEXT: %[[LT1R:.*]] = kgen.rebind %[[LT1]] {{.*}} to !lit.ref<!Int
    # CHECK-NEXT: lit.call {{.*}}takes_inout_int{{.*}}(%[[LT1R]])
    # CHECK-NEXT: %[[LT1R2:.*]] = kgen.rebind %[[LT1]] {{.*}} to !lit.ref<!Int

    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant: !Int = <x>
    # CHECK-NEXT: %[[LT3:.*]] = lit.var.decl {{.*}}!IndexArray
    # CHECK-NEXT: lit.call {{.*}}__getitem__{{.*}}(%b, %[[C1]], %[[LT3]])
    # CHECK-NEXT: %[[C2:.*]] = kgen.param.constant: !Int = <y>
    # CHECK-NEXT: %[[V9:.*]] = lit.ref.load %[[LT1R2]]
    # CHECK-NEXT: lit.call {{.*}}__setitem__{{.*}}(%[[LT3]], %[[C2]], %[[V9]])
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant: !Int = <x>
    # CHECK-NEXT: %[[V11:.*]] = lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], %[[LT3]])
    takes_inout_int(b[x][y])


struct RegWeirdArray(RegisterPassable):
    def __getitem__(self, idx: Int) -> Int:
        return idx

    def __setitem__(self, idx: Int, value: Int):
        pass


# CHECK-LABEL: lit.fn @"test_dlvalue_to_pvalue
def test_dlvalue_to_pvalue[arr: RegWeirdArray, y: Int]():
    # CHECK-NEXT: lit.alias.decl *"x{{.*}}": !alias_Int1 = <apply({{.*}}@RegWeirdArray::@"__getitem__{{.*}}"), store_to_mem(arr), y)>
    comptime x = arr[y]


struct XYZ(Movable where False):
    def __getattr_param__[name: StringLiteral](self) -> Int:
        comptime if name == "x":
            return 4
        elif name == "y":
            return 6
        else:
            comptime assert name == "z", "can only index with x, y, or z"
            return 8


struct ParamIndex(Movable where False):
    def __getitem_param__[a: Int, b: Int](self) -> Int:
        return 42


# CHECK-LABEL: lit.fn @"test_param_indexing
def test_param_indexing(a: XYZ, b: ParamIndex) -> Int:
    # Issue #35662: Support parameter input to getattr
    # CHECK: lit.call {{.*}}__getattr_param__{{.*}}!lit.struct<#StringLiteral <:string "x">> {}>(%a)
    _ = a.x
    # CHECK: lit.call {{.*}}__getattr_param__{{.*}}!lit.struct<#StringLiteral <:string "y">> {}>(%a)
    _ = a.y
    # CHECK: lit.call {{.*}}__getitem_param__{{.*}}<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 4}>(%b)
    _ = b[2, 4]


struct TestCompTime[wa: WeirdArray, value: Int = wa[4]](Movable where False):
    def test1[a: Int](self):
        var b: TestCompTime[Self.wa]

    def test2(self):
        self.test1[Self.wa[1]]()
        self.test1[Self.value]()


# ===----------------------------------------------------------------------=== #
# Keyword arguments in setters


@fieldwise_init
struct VariadicIndexList(Movable where False):
    def __getitem__(mut self, *indices: Int) -> Int:
        pass

    def __setitem__(mut self, *indices: Int, val: Int):
        pass


# CHECK-LABEL: lit.fn @"testVariadicIndexList
# MOCO-696: Support variadic length keys in __setitem__
def testVariadicIndexList(mut foo: VariadicIndexList, i: Int, the_value: Int):
    # Getter is straight-forward.
    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [{{.*}}]
    # CHECK: lit.call {{.*}}VariadicIndexList::@"__getitem__{{.*}}(%foo, {{.*}})
    _ = foo[i, i]

    # Setter needs to pass the new value as 'val', not in the variadics.
    # CHECK: lit.var.decl "__passed_varargs__"
    # CHECK-NEXT: {{%.*}} = pop.array.create [{{.*}}]
    # CHECK: lit.call {{.*}}VariadicIndexList::@"__setitem__{{.*}}(%foo, {{.*}}, {{.*}})
    foo[i, i, i, i] = the_value


# MOCO-1244:


struct RefResultInOverloaded(Movable where False):
    var x: String

    def __getitem__(self) raises -> ref[self.x] String:
        return self.x

    def __setitem__(mut self, var x: String):
        pass


# CHECK-LABEL: lit.fn @"testRefResultInOverloaded
def testRefResultInOverloaded(
    mut rrio: RefResultInOverloaded, var str: String
) raises:
    # CHECK: lit.call {{.*}}__getitem__
    # CHECK: lit.call {{.*}}unsafe_ptr
    _ = rrio[].unsafe_ptr()
    # CHECK-NOT: __init__{{.*}}"copy"
    # CHECK: lit.call {{.*}}__setitem__
    rrio[] = str^


# ===----------------------------------------------------------------------=== #
# DLV Subscript -> Ref binding resolution
# ===----------------------------------------------------------------------=== #


struct MinimalDict(Movable where False):
    var state: Int

    def __getitem__(self, key: Int) raises -> ref[self.state] Int:
        return self.state

    def __setitem__(mut self, key: Int, value: Int):
        self.state = value


def take_ref(ref x: Int):
    pass


# CHECK-LABEL: lit.fn @"test_ref_subscript_binding
def test_ref_subscript_binding(mut d: MinimalDict) raises -> ref[d[0]] Int:
    # Check that a ref returned by the getitem is directly passed.

    # CHECK-NEXT: [[DIMM:%.*]] = lit.ref.immut %d
    # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lit.call {{.*}}@MinimalDict::@"__getitem__{{.*}}([[DIMM]], [[ZERO]], %__error__, %__call_result_tmp__)
    # CHECK-NEXT: [[REF:%.*]] = lit.load.consume %__call_result_tmp__
    # CHECK-NEXT: [[REFR:%.*]] = kgen.rebind [[REF]] {{.*}} to !lit.ref<!Int
    # CHECK-NEXT: lit.call {{.*}}take_ref{{.*}}([[REFR]])
    take_ref(d[0])

    ref some_ref = d[0]

    return d[0]


# ===----------------------------------------------------------------------=== #
# By-Ref Result on __getattr__handled correctly
# ===----------------------------------------------------------------------=== #


trait FromInt:
    def __init__(out self, *, from_int: Int):
        ...


struct MyInt(FromInt, Movable where False):
    def __init__(out self, *, from_int: Int):
        pass


@fieldwise_init
struct Test[T: FromInt](ImplicitlyCopyable):
    # CHECK:       lit.fn @"__getattr_param__
    # CHECK-SAME:  byref_result
    def __getattr_param__[dim: StringLiteral](self) -> Self.T:
        return Self.T(from_int=42)


comptime t = Test[MyInt]()
# CHECK: lit.alias.decl *"tx{{.*}}": !MyInt = <{{.*}}@"__getattr_param__
comptime tx = t.x
