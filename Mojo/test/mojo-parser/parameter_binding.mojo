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

# RUN: %parse-mojo-isolated -split-input-file %s | FileCheck %s


@fieldwise_init
struct ParamType[a: Int](Movable where False):
    pass


struct SomeStruct[a: Int, b: Int, c: Int = 2](Movable where False):
    @staticmethod
    def foo(arg: ParamType[Self.b]) -> Int:
        return Self.b


# This install default c = 2 and create `SomeStruct[1, ?, 2]`
# CHECK: lit.alias.decl *"c0`{{.*}}": meta<!lit.struct<#SomeStruct <:!Int {:scalar<index> 1}, :!Int ?, :!Int {:scalar<index> 2}>
comptime c0 = SomeStruct[1, _]

# This is SomeStruct[1, ?, ?]

# CHECK: lit.alias.decl *"c1`{{.*}}": meta<!lit.struct<#SomeStruct <:!Int {:scalar<index> 1}, :!Int ?, :!Int ?>
comptime c1 = SomeStruct[1, _, _]


# This is the same as SomeStruct[1, 3, 2], this is because
# SomeStruct[1, _] binds a and c (since [] always produces the MOST concrete type), the second [] binds b to 3.
# CHECK: lit.alias.decl *"c2`{{.*}}": meta<!lit.struct<#SomeStruct <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 3}, :!Int {:scalar<index> 2}>
comptime c2 = SomeStruct[1, _][3]


def foo[a: Int, b: Int, c: Int, d: Int](x: SomeStruct[a, b, c]):
    pass


def test(x: SomeStruct[1, 2, 3]):
    # Make sure we handle call binding correctly without requiring:
    # foo[d=1, ...] or foo[_, _, _, d=1]

    # CHECK: lit.call tail @parameter_binding::@"foo[::SIMD[DType.int, 1],::SIMD[DType.int, 1],::SIMD[DType.int, 1],::SIMD[DType.int, 1]]
    foo[d=1](x)

    # Although they are all valid syntax ofc.

    # CHECK: lit.call tail @parameter_binding::@"foo[::SIMD[DType.int, 1],::SIMD[DType.int, 1],::SIMD[DType.int, 1],::SIMD[DType.int, 1]]
    foo[d=1, ...](x)

    # CHECK: lit.call tail @parameter_binding::@"foo[::SIMD[DType.int, 1],::SIMD[DType.int, 1],::SIMD[DType.int, 1],::SIMD[DType.int, 1]]
    foo[_, _, _, d=1](x)

    # CHECK: lit.call @parameter_binding::@SomeStruct::@"foo({{.*}})"{{.*}}<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 4}, :!Int {:scalar<index> 2}>
    var _ = SomeStruct[1].foo(ParamType[4]())


def foo[T: def[a: Int, b: Int](ParamType[b]) thin -> Int](param: ParamType[1]):
    # CHECK: lit.call tail{{.*}}bind_params(:{{.*}} T, :!Int {:scalar<index> 2}, :!Int {:scalar<index> 1})]
    T[2](param)


# We allow default values on inferred parameters.
# CHECK: lit.struct.decl @MySpan<mut: !Bool = {:scalar<bool> false}
struct MySpan[
    mut: Bool = False,
    //,
    origin: Origin[mut=mut],
]():
    pass


# // -----


def defaulted_slice_param[param: StringSlice = "world"]() -> String:
    return "Hello " + param


def use_defaulted_slice_param():
    # Binding the default installs the static-origin `StringSpan` for `param`.
    # CHECK: lit.call @parameter_binding::@"defaulted_slice_param[{{.*}}]()"
    # CHECK-SAME: <:!Bool {:scalar<bool> false},
    # CHECK-SAME: :!lit.struct<#StringSpan {{.*}}#alias_ImmStaticOrigin>>
    var _ = defaulted_slice_param()


# // -----

comptime KE = Movable & Deinitable


struct Entry[K: KE](Movable where False):
    var key: Self.K

    def __init__(out self, var key: Self.K):
        self.key = key^


def use[K: KE & Copyable](e: Entry[K]):
    # Make sure we can look through sugar rebind due to `KE`.
    # CHECK: lit.call{{.*}}#kgen.get_witness<:!{{.*}} K, @{{.*}}Copyable, "copy($0)">
    var _k = (e.key).copy()
