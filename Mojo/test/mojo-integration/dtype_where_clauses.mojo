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

# This test intentionally uses the real stdlib to test DType method evaluation
# in where clauses. It should remain as an integration test.

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s \
# RUN:   | kgen-opt --kgen-print-inline-type-values | FileCheck %s


@always_inline("builtin")
def dtype_where_clause_eq_ne[d: DType]() -> Int where d == .int32:
    return 42


@always_inline("builtin")
def dtype_where_clause_eq_ne[d: DType]() -> Int where d != .int32:
    return 0


@always_inline("builtin")
def dtype_where_clause_is_signed[d: DType]() -> Int where d.is_signed():
    return 1


@always_inline("builtin")
def dtype_where_clause_is_signed[d: DType]() -> Int where not d.is_signed():
    return 0


@always_inline("builtin")
def dtype_where_clause_is_unsigned[d: DType]() -> Int where d.is_unsigned():
    return 1


@always_inline("builtin")
def dtype_where_clause_is_unsigned[d: DType]() -> Int where not d.is_unsigned():
    return 0


@always_inline("builtin")
def dtype_where_clause_is_integral[d: DType]() -> Int where d.is_integral():
    return 1


@always_inline("builtin")
def dtype_where_clause_is_integral[d: DType]() -> Int where not d.is_integral():
    return 0


@always_inline("builtin")
def dtype_where_clause_is_floating_point[
    d: DType
]() -> Int where d.is_floating_point():
    return 1


@always_inline("builtin")
def dtype_where_clause_is_floating_point[
    d: DType
]() -> Int where not d.is_floating_point():
    return 0


@always_inline("builtin")
def dtype_where_clause_is_half_float[d: DType]() -> Int where d.is_half_float():
    return 1


@always_inline("builtin")
def dtype_where_clause_is_half_float[
    d: DType
]() -> Int where not d.is_half_float():
    return 0


@always_inline("builtin")
def dtype_where_clause_is_float8[d: DType]() -> Int where d.is_float8():
    return 1


@always_inline("builtin")
def dtype_where_clause_is_float8[d: DType]() -> Int where not d.is_float8():
    return 0


@always_inline("builtin")
def dtype_where_clause_is_numeric[d: DType]() -> Int where d.is_numeric():
    return 1


@always_inline("builtin")
def dtype_where_clause_is_numeric[d: DType]() -> Int where not d.is_numeric():
    return 0


# CHECK: lit.fn @"foo
def foo():
    # CHECK: lit.alias.decl *"x`": ![[INT_TYPE:.*]] = {{.*}}{:scalar<index> 42}
    comptime x = dtype_where_clause_eq_ne[.int32]()
    # CHECK: lit.alias.decl *"y`1": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime y = dtype_where_clause_eq_ne[.int64]()

    # CHECK: lit.alias.decl *"c`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime c = dtype_where_clause_is_signed[.int32]()
    # CHECK: lit.alias.decl *"d`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime d = dtype_where_clause_is_signed[.uint32]()
    # CHECK: lit.alias.decl *"e`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime e = dtype_where_clause_is_signed[.float64]()

    # CHECK: lit.alias.decl *"f`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime f = dtype_where_clause_is_unsigned[.uint32]()
    # CHECK: lit.alias.decl *"g`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime g = dtype_where_clause_is_unsigned[.int32]()
    # CHECK: lit.alias.decl *"h`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime h = dtype_where_clause_is_unsigned[.float64]()

    # CHECK: lit.alias.decl *"i`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime i = dtype_where_clause_is_integral[.uint32]()
    # CHECK: lit.alias.decl *"j`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime j = dtype_where_clause_is_integral[.int32]()
    # CHECK: lit.alias.decl *"k`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime k = dtype_where_clause_is_integral[.float64]()

    # CHECK: lit.alias.decl *"l`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime l = dtype_where_clause_is_floating_point[.int32]()
    # CHECK: lit.alias.decl *"m`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime m = dtype_where_clause_is_floating_point[.float32]()

    # CHECK: lit.alias.decl *"n`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime n = dtype_where_clause_is_half_float[.int32]()
    # CHECK: lit.alias.decl *"o`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime o = dtype_where_clause_is_half_float[.float32]()
    # CHECK: lit.alias.decl *"p`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime p = dtype_where_clause_is_half_float[.float16]()
    # CHECK: lit.alias.decl *"q`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime q = dtype_where_clause_is_half_float[.bfloat16]()

    # CHECK: lit.alias.decl *"r0`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime r0 = dtype_where_clause_is_float8[.float8_e3m4]()
    # CHECK: lit.alias.decl *"r1`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime r1 = dtype_where_clause_is_float8[.float8_e5m2]()
    # CHECK: lit.alias.decl *"r2`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime r2 = dtype_where_clause_is_float8[.float32]()

    # CHECK: lit.alias.decl *"s0`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime s0 = dtype_where_clause_is_numeric[.int32]()
    # CHECK: lit.alias.decl *"s1`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 1}
    comptime s1 = dtype_where_clause_is_numeric[.float32]()
    # CHECK: lit.alias.decl *"s2`{{.*}}": ![[INT_TYPE]] = {{.*}}{:scalar<index> 0}
    comptime s2 = dtype_where_clause_is_numeric[.bool]()
