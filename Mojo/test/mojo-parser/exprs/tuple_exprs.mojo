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

##===----------------------------------------------------------------------===##
# RValue tests
##===----------------------------------------------------------------------===##


def use[T: AnyType](a: T):
    pass


# CHECK-LABEL: lit.fn @"tuples_rv
def tuples_rv(a: Int, b: FloatDyn):
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl{{.*}}Tuple <:param_list<!AnyType_Movable> []
    # CHECK: lit.call {{.*}}@Tuple::@"__init__({{.*}}([[TMPVAR]])
    use(())
    # CHECK: lit.call {{.*}}tuple_exprs::@"use

    # CHECK: [[AREF:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.ref.store %a, [[AREF]]
    # CHECK-NEXT: [[BREF:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.ref.store %b, [[BREF]]
    # CHECK-NEXT: [[AIMM:%.*]] = lit.ref.immut [[AREF]] : <!Int, mut [[ALT:.*]]>
    # CHECK-NEXT: [[BIMM:%.*]] = lit.ref.immut [[BREF]] : <!FloatDyn, mut [[BLT:.*]]>
    # CHECK-NEXT: [[AREBOUND:%.*]] = lit.ref.upcast [[AIMM]] : <!Int, muttoimm [[ALT]]> -> <!Int, imm {(mutcast mut [[ALT]]), (mutcast mut [[BLT]])}>
    # CHECK-NEXT: [[BREBOUND:%.*]] = lit.ref.upcast [[BIMM]] : <!FloatDyn, muttoimm [[BLT]]> -> <!FloatDyn, imm {(mutcast mut [[ALT]]), (mutcast mut [[BLT]])}>
    # CHECK-NEXT: = lit.ref.pack.create([[AREBOUND]], [[BREBOUND]])
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}Tuple
    # CHECK: lit.call {{.*}}@Tuple::@"__init__{{.*}}({{.*}}, [[TMPVAR]])
    use((a, b))
    # CHECK: lit.call {{.*}}tuple_exprs::@"use

    # CHECK: = lit.ref.pack.create({{%[0-9]+}}, {{%[0-9]+}})
    # CHECK: %t3 = lit.var.decl "t3"
    # CHECK: lit.call {{.*}}@Tuple::@"__init__{{.*}}({{.*}}, %t3)
    var t3 = a, b
    use(t3)
    # CHECK: lit.call {{.*}}tuple_exprs::@"use

    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}#Tuple
    # CHECK: lit.call {{.*}}@Tuple::@"__init__{{.*}}, [[TMPVAR]])
    use((a,))
    # CHECK: lit.call {{.*}}tuple_exprs::@"use

    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}#Tuple
    # CHECK: lit.call {{.*}}@Tuple::@"__init__{{.*}}, [[TMPVAR]])
    use((a,))
    # CHECK: lit.call {{.*}}tuple_exprs::@"use

    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: %t2 = lit.var.decl "t2"
    # CHECK: [[TUP2:%.*]] = lit.call {{.*}}@Tuple::@"__init__{{.*}}({{.*}}, %t2)
    var t2 = (a,)
    use(t2)
    # CHECK: lit.call {{.*}}tuple_exprs::@"use


##===----------------------------------------------------------------------===##
# LValue tests
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"tuples_lv
def tuples_lv(i0: Int, f0: FloatDyn):
    var i1 = 1
    var i2 = 2

    # CHECK: %iTup = lit.var.decl "iTup"
    var iTup: Tuple[Int, Int]

    # Tuple Rvalue
    # CHECK: [[TUP:%.*]] = lit.call {{.*}}@Tuple::@"__init__{{.*}}({{.*}}, %iTup)
    iTup = (i1, i2)

    # Tuple LValue
    # CHECK: [[IMMTUP:%.*]] = lit.ref.immut %iTup
    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem_param__{{.*}}([[IMMTUP]])
    # CHECK: [[ELTREB:%.*]] = kgen.rebind [[ELT]]
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELTREB]]
    # CHECK: lit.ref.store [[ELTV]], %i1

    # CHECK: [[IMMTUP:%.*]] = lit.ref.immut %iTup
    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem_param__{{.*}}([[IMMTUP]])
    # CHECK: [[ELTREB:%.*]] = kgen.rebind [[ELT]]
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELTREB]]
    # CHECK: lit.ref.store [[ELTV]], %i2
    (i1, i2) = iTup

    # Check that the swap idiom is correct, this requires producing a copy of the
    # whole RValue on the right before extracting from it.

    # CHECK:  = lit.ref.pack.create
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}#Tuple
    # CHECK: [[TUPRV:%.*]] = lit.call {{.*}}__init__{{.*}}({{.*}}, [[TMPVAR]])

    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem_param__{{.*}}>(
    # CHECK: [[ELTREB:%.*]] = kgen.rebind [[ELT]]
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELTREB]]
    # CHECK: lit.ref.store [[ELTV]], %i1

    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem_param__{{.*}}>(
    # CHECK: [[ELTREB:%.*]] = kgen.rebind [[ELT]]
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELTREB]]
    # CHECK: lit.ref.store [[ELTV]], %i2
    (i1, i2) = (i2, i1)

    # CHECK: [[ELT:%.*]] = lit.call {{.*}}__getitem_param__{{.*}}(%iTup)
    # CHECK-NEXT: [[I1REB:%.*]] = kgen.rebind %i1
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load [[I1REB]]
    # CHECK-NEXT: lit.ref.store [[TMP]], [[ELT]]
    iTup[1] = i1

    var f1: FloatDyn
    # Mixed element types should work.  Don't need check lines though.
    (i1, f1) = (i0, f0)


##===----------------------------------------------------------------------===##
# Memory-only element tests
##===----------------------------------------------------------------------===##


trait CollectionType(ImplicitlyCopyable):
    pass


struct Container[T: CollectionType & Deinitable](Movable where False):
    var x: Self.T

    def __setitem__(mut self, i: Int, var value: Self.T):
        self.x = value

    def __getitem__(self, i: Int) -> Self.T:
        return self.x


# CHECK-LABEL: lit.fn @"swap_container_fields
def swap_container_fields(mut v: Container[_]):
    v[0], v[1] = v[1], v[0]


##===----------------------------------------------------------------------===##
# Tuple Types
##===----------------------------------------------------------------------===##

# FIXME: Empty tuple `Tuple[]` cannot be spelled.


# CHECK-LABEL: lit.fn @"returnTup0
# CHECK-SAME: %__result__: !lit.ref<{{.*}}#Tuple <:param_list<!AnyType_Movable> []
def returnTup0() -> Tuple[]:
    # CHECK: lit.call {{.*}}@Tuple::@"__init__{{.*}}(%__result__)
    return ()


# CHECK-LABEL: lit.fn @"returnTup0a
# CHECK-SAME: %__result__: !lit.ref<{{.*}}#Tuple <:param_list<!AnyType_Movable> []
def returnTup0a() -> ():
    # CHECK: lit.call {{.*}}@Tuple::@"__init__{{.*}}(%__result__)
    return ()


# CHECK-LABEL: lit.fn @"returnTup1
# CHECK-SAME: %__result__: !lit.ref<{{.*}}#Tuple <:param_list<!AnyType_Movable> [!Int]
def returnTup1() -> Tuple[Int]:
    # CHECK: %0 = kgen.param.constant: !Int
    # CHECK:   = lit.ref.pack.create({{.*}}) : !lit.ref.pack<:param_list<!AnyType_Movable> [!Int],
    # CHECK:  = lit.call{{.*}}__init__
    return (Int(4),)


# CHECK-LABEL: lit.fn @"returnTup1
# CHECK-SAME: %__result__: !lit.ref<{{.*}}#Tuple <:param_list<!AnyType_Movable> [!Int]
def returnTup1a() -> Tuple[Int]:
    return (Int(4),)


def returnTup1b() -> Tuple[Int]:
    return (Int(4),)


# CHECK-LABEL: lit.fn @"returnTup2
# CHECK-SAME:  %__result__: !lit.ref<{{.*}}#Tuple <:param_list<!AnyType_Movable> [!Int, !FloatDyn]
def returnTup2() -> Tuple[Int, FloatDyn]:
    # CHECK:  = kgen.param.constant: !Int = <{:scalar<index> 4}>
    # CHECK:  = kgen.param.constant: !FloatDyn = <{{.*}}{:scalar<f64> "2"}
    # CHECK: lit.ref.pack.create({{.*}}) : !lit.ref.pack<:param_list<!AnyType_Movable> [!Int, !FloatDyn]
    return (Int(4), 2.0)


# CHECK-LABEL: lit.fn @"returnTup2a
# CHECK-SAME: %__result__: !lit.ref<{{.*}}#Tuple <:param_list<!AnyType_Movable> [!Int, !FloatDyn]
def returnTup2a() -> Tuple[Int, FloatDyn]:
    # CHECK: lit.ref.pack.create({{.*}}) : !lit.ref.pack<:param_list<!AnyType_Movable> [!Int, !FloatDyn]
    return (Int(4), 2.0)


# CHECK-LABEL: lit.fn @"returnTup2b
def returnTup2b() -> Tuple[Int, FloatDyn]:
    return Int(4), 2.0


# CHECK-LABEL: lit.fn @"takesSugarTuple{{.*}}<T: !AnyType_Copyable_ImplicitlyCopyable_Movable>
# CHECK-SAME: #Tuple <:param_list<!AnyType_Movable> [upcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable T), upcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable T)]
def takesSugarTuple[T: ImplicitlyCopyable](elements: Tuple[T, T]):
    pass


# CHECK-LABEL: lit.fn @"index_homogenous_tuple
def index_homogenous_tuple[idx: Int]():
    var tup = (1, 2, 3, 4)
    # CHECK: [[ELTPTR:%.*]] = lit.call {{.*}}Tuple::@"__getitem_param__{{.*}}:!Int {:scalar<index> 1}{{.*}}(%tup)
    # CHECK-NEXT: %test1 = lit.var.decl "test1"
    # CHECK-NEXT: [[ELTREB:%.*]] = kgen.rebind [[ELTPTR]]
    # CHECK-NEXT: [[INTVAL:%.*]] = lit.ref.load [[ELTREB]]
    # CHECK-NEXT: lit.ref.store [[INTVAL]], %test1
    var test1: Int = tup[1]

    # TODO(MOCO-4505): temporarily disabled for closure migration, re-enable
    # later.
    #
    # C_HECK: [[ELTPTR:%.*]] = lit.call {{.*}}Tuple::@"__getitem_param__{{.*}}:!Int idx{{.*}}(%tup)
    # C_HECK-NEXT: %test2 = lit.var.decl "test2"
    # C_HECK-NEXT: [[ELTREB:%.*]] = kgen.rebind [[ELTPTR]]
    # C_HECK-NEXT: [[INTVAL:%.*]] = lit.ref.load [[ELTREB]]
    # C_HECK-NEXT: lit.ref.store [[INTVAL]], %test2
    # var test2: Int = rebind[Int](tup[idx])
