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

# Test @align decorator with parametric alignment values.


# Test basic parametric alignment using struct parameter
# CHECK-LABEL: lit.struct.decl @AlignedBuffer
# CHECK-SAME: minAlignment = #kgen.cast_to_builtin<#lit.struct.extract<:!Int alignment, "_mlir_value"> : !kgen.scalar<index>> : index
@align(alignment)
struct AlignedBuffer[alignment: Int](Movable where False):
    var data: Int


# Test parametric alignment with TrivialRegisterPassable
# CHECK-LABEL: lit.struct.decl @AlignedTrivialParam
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: minAlignment = #kgen.cast_to_builtin<#lit.struct.extract<:!Int n, "_mlir_value"> : !kgen.scalar<index>> : index
@align(n)
struct AlignedTrivialParam[n: Int](TrivialRegisterPassable):
    var value: __mlir_type.index


# Test parametric alignment combined with multiple parameters
# CHECK-LABEL: lit.struct.decl @MultiParam
# CHECK-SAME: minAlignment = #kgen.cast_to_builtin<#lit.struct.extract<:!Int align_val, "_mlir_value"> : !kgen.scalar<index>> : index
@align(align_val)
struct MultiParam[T: __mlir_type.`!kgen.type`, align_val: Int](Movable where False):
    var value: Self.T

    def __deinit__(deinit self):
        pass


# Test parametric alignment with an expression (n * 2)
# CHECK-LABEL: lit.struct.decl @AlignedExpr
# CHECK-SAME: minAlignment = #kgen.cast_to_builtin<#kgen.param.expr<mul, #lit.struct.extract<:!Int n, "_mlir_value"> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>> : !kgen.scalar<index>> : index
@align(n * 2)
struct AlignedExpr[n: Int](Movable where False):
    var data: Int
