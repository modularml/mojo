# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""This module includes the inlined_assembly function."""

from sys.intrinsics import _mlirtype_is_eq

# ===----------------------------------------------------------------------===#
# 0-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r",
    has_side_effect: Bool = True,
]() -> result_type:
    """Generates assembly via inline for instructions with 0 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ]()
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ]()
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ]()
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ]()


# ===----------------------------------------------------------------------===#
# 1-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r",
    has_side_effect: Bool = True,
](arg0: arg0_type) -> result_type:
    """Generates assembly via inline for instructions with 1 arg."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0)


# ===----------------------------------------------------------------------===#
# 2-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r",
    has_side_effect: Bool = True,
](arg0: arg0_type, arg1: arg1_type) -> result_type:
    """Generates assembly via inline for instructions with 2 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1)


# ===----------------------------------------------------------------------===#
# 3-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r",
    has_side_effect: Bool = True,
](arg0: arg0_type, arg1: arg1_type, arg2: arg2_type) -> result_type:
    """Generates assembly via inline for instructions with 3 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2)


# ===----------------------------------------------------------------------===#
# 4-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    arg3_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type, arg1: arg1_type, arg2: arg2_type, arg3: arg3_type
) -> result_type:
    """Generates assembly via inline for instructions with 4 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3)


# ===----------------------------------------------------------------------===#
# 5-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    arg3_type: AnyTrivialRegType,
    arg4_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
) -> result_type:
    """Generates assembly via inline for instructions with 5 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4)


# ===----------------------------------------------------------------------===#
# 6-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    arg3_type: AnyTrivialRegType,
    arg4_type: AnyTrivialRegType,
    arg5_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
) -> result_type:
    """Generates assembly via inline for instructions with 6 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5)


# ===----------------------------------------------------------------------===#
# 7-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    arg3_type: AnyTrivialRegType,
    arg4_type: AnyTrivialRegType,
    arg5_type: AnyTrivialRegType,
    arg6_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
    arg6: arg6_type,
) -> result_type:
    """Generates assembly via inline for instructions with 7 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)


# ===----------------------------------------------------------------------===#
# 8-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    arg3_type: AnyTrivialRegType,
    arg4_type: AnyTrivialRegType,
    arg5_type: AnyTrivialRegType,
    arg6_type: AnyTrivialRegType,
    arg7_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
    arg6: arg6_type,
    arg7: arg7_type,
) -> result_type:
    """Generates assembly via inline for instructions with 8 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)


# ===----------------------------------------------------------------------===#
# 9-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    arg3_type: AnyTrivialRegType,
    arg4_type: AnyTrivialRegType,
    arg5_type: AnyTrivialRegType,
    arg6_type: AnyTrivialRegType,
    arg7_type: AnyTrivialRegType,
    arg8_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r,r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
    arg6: arg6_type,
    arg7: arg7_type,
    arg8: arg8_type,
) -> result_type:
    """Generates assembly via inline for instructions with 9 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg3, arg5, arg6, arg7, arg8)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)


# ===----------------------------------------------------------------------===#
# 10-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringLiteral,
    result_type: AnyTrivialRegType,
    arg0_type: AnyTrivialRegType,
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
    arg3_type: AnyTrivialRegType,
    arg4_type: AnyTrivialRegType,
    arg5_type: AnyTrivialRegType,
    arg6_type: AnyTrivialRegType,
    arg7_type: AnyTrivialRegType,
    arg8_type: AnyTrivialRegType,
    arg9_type: AnyTrivialRegType,
    /,
    *,
    constraints: StringLiteral = "r,r,r,r,r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
    arg6: arg6_type,
    arg7: arg7_type,
    arg8: arg8_type,
    arg9: arg9_type,
) -> result_type:
    """Generates assembly via inline for instructions with 10 args."""

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
