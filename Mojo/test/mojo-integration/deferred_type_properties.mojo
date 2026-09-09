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

# RUN: %mojo %s | FileCheck %s
# RUN: kgen -emit=llvm %s | FileCheck %s --check-prefix=IR

# `nsw`/`nuw` are property-stored, so their presence in the LLVM IR proves
# `_properties` (whether `__mlir_attr` or `__mlir_deferred_attr`) survived
# the `kgen.deferred` round trip.

from std.collections.string.string_span import _get_kgen_string


@no_inline
def add_with_nsw[
    width: Int
](
    a: __mlir_deferred_type[`i`, +width.__mlir_index__()],
    b: __mlir_deferred_type[`i`, +width.__mlir_index__()],
) -> __mlir_deferred_type[`i`, +width.__mlir_index__()]:
    # IR: add nsw i32
    return __mlir_op.`llvm.add`[
        _type=__mlir_deferred_type[`i`, +width.__mlir_index__()],
        _properties=__mlir_attr.`{overflowFlags = #llvm.overflow<nsw>}`,
    ](a, b)


@always_inline("nodebug")
def _overflow_kind_str[signed: Bool]() -> StaticString:
    comptime if signed:
        return "nsw"
    else:
        return "nuw"


@no_inline
def add_with_deferred_props[
    width: Int, signed: Bool
](
    a: __mlir_deferred_type[`i`, +width.__mlir_index__()],
    b: __mlir_deferred_type[`i`, +width.__mlir_index__()],
) -> __mlir_deferred_type[`i`, +width.__mlir_index__()]:
    # IR: add nuw i32
    return __mlir_op.`llvm.add`[
        _type=__mlir_deferred_type[`i`, +width.__mlir_index__()],
        _properties=__mlir_deferred_attr[
            `{overflowFlags = #llvm.overflow<`,
            +_get_kgen_string[_overflow_kind_str[signed]()](),
            `>}`,
        ],
    ](a, b)


def main():
    # CHECK: ok
    comptime w: Int = 32
    var a = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i32](
        Int32(5)._mlir_value
    )
    var b = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i32](
        Int32(7)._mlir_value
    )
    _ = add_with_nsw[w](a, b)
    _ = add_with_deferred_props[w, False](a, b)
    print("ok")
