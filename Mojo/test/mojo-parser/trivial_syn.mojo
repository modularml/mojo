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


struct X_T(ImplicitlyCopyable):
    pass


struct Y_T(ImplicitlyCopyable):
    pass


struct X_N(ImplicitlyCopyable):
    def __deinit__(deinit self):
        pass


struct X_T_U(ImplicitlyCopyable):
    def __deinit__(deinit self):
        pass

    # User marked __del__as trivial
    comptime __del__is_trivial: Bool = True


# CHECK-LABEL: lit.struct.decl @C
#  CHECK-SAME: <X: [[X_TYPE:!.*]], Y: [[Y_TYPE:!.*]]>
struct C[X: ImplicitlyCopyable & Deinitable, Y: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable):
    var x: Self.X
    var y: Self.Y

    # CHECK-LABEL:  kgen.conformance @{{.*}}::@AnyType {
    # CHECK-NEXT:   }

    # CHECK-LABEL:  kgen.conformance @{{.*}}::@Copyable {
    # CHECK-NEXT:      kgen.witness "__init__{{.*}}(*, "copy":{{.*}}"
    # CHECK-NEXT:      kgen.witness "copy{{.*}}"
    # CHECK: kgen.witness "__copy_ctor_is_trivial" : !Bool = sugar_builtin(apply({{.*}})

    # CHECK-LABEL:  kgen.conformance @{{.*}}::@Deinitable {
    # CHECK-NEXT:    kgen.witness "__deinit__{{.*}}"
    # CHECK: kgen.witness "__del__is_trivial" : !Bool = sugar_builtin(apply({{.*}})

    # CHECK-LABEL:  kgen.conformance @{{.*}}::@ImplicitlyCopyable {
    # CHECK-NEXT:   }

    # CHECK-LABEL:  kgen.conformance @{{.*}}::@Movable {
    # CHECK-NEXT:    kgen.witness "__init__{{.*}}(*, "move":{{.*}}"
    # CHECK: kgen.witness "__move_ctor_is_trivial" : !Bool = sugar_builtin(apply({{.*}})


# CHECK-LABEL: lit.struct.decl @StructMLIRTypeOnly
struct StructMLIRTypeOnly(ImplicitlyCopyable):
    var x: __mlir_type.index
    var y: __mlir_type.index

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:scalar<bool> true}>
    # CHECK-DAG: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:scalar<bool> true}>
    # CHECK-DAG: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:scalar<bool> true}>


# MOCO-2396:
# CHECK-LABEL: lit.struct.decl @NotTrivial
struct NotTrivial(Copyable):
    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    def __deinit__(deinit self):
        pass

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:scalar<bool> false}>
    # CHECK-DAG: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:scalar<bool> false}>
    # CHECK-DAG: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:scalar<bool> false}>


# CHECK-LABEL: lit.struct.decl @Wrapper
struct Wrapper(Copyable):
    var value: NotTrivial

    # Should be parser-folded.

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:scalar<bool> false}>
    # CHECK-DAG: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:scalar<bool> false}>
    # CHECK-DAG: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:scalar<bool> false}>


# CHECK-LABEL: lit.struct.decl @TrivialFieldGen
# CHECK: lit.alias.decl __del__is_trivial: !Bool = <#kgen.get_witness<:!AnyType_Deinitable_Movable T, @{{.*}}::@Deinitable, "__del__is_trivial">>
# CHECK: lit.alias.decl __move_ctor_is_trivial: !Bool = <#kgen.get_witness<:!AnyType_Deinitable_Movable T, @{{.*}}::@Movable, "__move_ctor_is_trivial">>
struct TrivialFieldGen[T: Movable & Deinitable](Movable):
    var z: Self.T
    var y: Int
    var q: Self.T


# CHECK-LABEL: lit.struct.decl @TestTrivialRegisterPassable
# CHECK: lit.alias.decl __del__is_trivial: !Bool = <{:scalar<bool> true}>
# CHECK: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:scalar<bool> true}>
# CHECK: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:scalar<bool> true}>
struct TestTrivialRegisterPassable[T: TrivialRegisterPassable](Copyable):
    var _value: Self.T


# MOCO-3862: overriding a base trait's defaulted `__init__` while conforming to
# a builtin `Copyable`/`Movable`/`Deinitable` trait must not crash
# trivial-special-function synthesis.
trait DefaultsInit:
    def __init__(out self, x: Int):
        _ = x


# CHECK-LABEL: lit.struct.decl @OverridesDefaultInit
struct OverridesDefaultInit(DefaultsInit, Copyable):
    def __init__(out self, x: Int):
        pass
