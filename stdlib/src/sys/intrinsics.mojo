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
"""Defines intrinsics.

You can import these APIs from the `complex` package. For example:

```mojo
from sys import PrefetchLocality
```
"""

from sys import sizeof

from memory import AddressSpace, DTypePointer

# ===----------------------------------------------------------------------===#
# llvm_intrinsic
# ===----------------------------------------------------------------------===#

# FIXME: Need tuple unpacking to write a single function definition.


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral, type: AnyTrivialRegType, has_side_effect: Bool = True
]() -> type:
    """Calls an LLVM intrinsic with no arguments.

    Calls an LLVM intrinsic with the name intrin and return type type.

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Returns:
      The result of calling the llvm intrinsic with no arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=None,
            ]()
            return rebind[type](None)

        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ]()
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=type,
            ]()
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ]()


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    has_side_effect: Bool = True,
](arg0: T0) -> type:
    """Calls an LLVM intrinsic with one argument.

    Calls the intrinsic with the name intrin and return type type on argument
    arg0.

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Args:
      arg0: The argument to call the LLVM intrinsic with. The type of arg0
        must be T0.

    Returns:
      The result of calling the llvm intrinsic with arg0 as an argument.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    has_side_effect: Bool = True,
](arg0: T0, arg1: T1) -> type:
    """Calls an LLVM intrinsic with two arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
    arguments arg0 and arg1.

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of
        arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of
        arg1 must be T1.

    Returns:
      The result of calling the llvm intrinsic with arg0 and arg1 as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1)
        return rebind[type](None)
    else:
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0, arg1)

        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    has_side_effect: Bool = True,
](arg0: T0, arg1: T1, arg2: T2) -> type:
    """Calls an LLVM intrinsic with three arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
    arguments arg0, arg1 and arg2.

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of
        arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of
        arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of
        arg2 must be T2.

    Returns:
      The result of calling the llvm intrinsic with arg0, arg1 and arg2 as
      arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0, arg1, arg2)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    has_side_effect: Bool = True,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3) -> type:
    """Calls an LLVM intrinsic with four arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
    arguments arg0, arg1, arg2 and arg3.

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      T3: The type of the fourth argument to the intrinsic (arg3).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of
        arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of
        arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of
        arg2 must be T2.
      arg3: The fourth argument to call the LLVM intrinsic with. The type of
        arg3 must be T3.

    Returns:
      The result of calling the llvm intrinsic with arg0, arg1, arg2 and arg3
      as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2, arg3)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0, arg1, arg2, arg3)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    has_side_effect: Bool = True,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3, arg4: T4) -> type:
    """Calls an LLVM intrinsic with five arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
      arguments arg0, arg1, arg2, arg3 and arg4.

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      T3: The type of the fourth argument to the intrinsic (arg3).
      T4: The type of the fifth argument to the intrinsic (arg4).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.


    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of arg2 must be T2.
      arg3: The fourth argument to call the LLVM intrinsic with. The type of arg3 must be T3.
      arg4: The fifth argument to call the LLVM intrinsic with. The type of arg4 must be T4.

    Returns:
      The result of calling the llvm intrinsic with arg0...arg4 as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2, arg3, arg4)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0, arg1, arg2, arg3, arg4)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    has_side_effect: Bool = True,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3, arg4: T4, arg5: T5) -> type:
    """Calls an LLVM intrinsic with six arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
      arguments arg0, arg1, ..., arg5

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      T3: The type of the fourth argument to the intrinsic (arg3).
      T4: The type of the fifth argument to the intrinsic (arg4).
      T5: The type of the sixth argument to the intrinsic (arg5).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.


    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of arg2 must be T2.
      arg3: The fourth argument to call the LLVM intrinsic with. The type of arg3 must be T3.
      arg4: The fifth argument to call the LLVM intrinsic with. The type of arg4 must be T4.
      arg5: The sixth argument to call the LLVM intrinsic with. The type of arg5 must be T5.

    Returns:
      The result of calling the llvm intrinsic with arg0...arg5 as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2, arg3, arg4, arg5)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=type,
            ](arg0, arg1, arg2, arg3, arg4, arg5)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    has_side_effect: Bool = True,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3, arg4: T4, arg5: T5, arg6: T6) -> type:
    """Calls an LLVM intrinsic with seven arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
      arguments arg0, arg1, ..., arg6

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      T3: The type of the fourth argument to the intrinsic (arg3).
      T4: The type of the fifth argument to the intrinsic (arg4).
      T5: The type of the sixth argument to the intrinsic (arg5).
      T6: The type of the seventh argument to the intrinsic (arg6).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.


    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of arg2 must be T2.
      arg3: The fourth argument to call the LLVM intrinsic with. The type of arg3 must be T3.
      arg4: The fifth argument to call the LLVM intrinsic with. The type of arg4 must be T4.
      arg5: The sixth argument to call the LLVM intrinsic with. The type of arg5 must be T5.
      arg6: The seventh argument to call the LLVM intrinsic with. The type of arg6 must be T6.

    Returns:
      The result of calling the llvm intrinsic with arg0...arg6 as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    has_side_effect: Bool = True,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
) -> type:
    """Calls an LLVM intrinsic with eight arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
      arguments arg0, arg1, ..., arg7

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      T3: The type of the fourth argument to the intrinsic (arg3).
      T4: The type of the fifth argument to the intrinsic (arg4).
      T5: The type of the sixth argument to the intrinsic (arg5).
      T6: The type of the seventh argument to the intrinsic (arg6).
      T7: The type of the eighth argument to the intrinsic (arg7).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of arg2 must be T2.
      arg3: The fourth argument to call the LLVM intrinsic with. The type of arg3 must be T3.
      arg4: The fifth argument to call the LLVM intrinsic with. The type of arg4 must be T4.
      arg5: The sixth argument to call the LLVM intrinsic with. The type of arg5 must be T5.
      arg6: The seventh argument to call the LLVM intrinsic with. The type of arg6 must be T6.
      arg7: The eighth argument to call the LLVM intrinsic with. The type of arg7 must be T7.

    Returns:
      The result of calling the llvm intrinsic with arg0...arg7 as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    has_side_effect: Bool = True,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
) -> type:
    """Calls an LLVM intrinsic with nine arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
      arguments arg0, arg1, ..., arg8

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      T3: The type of the fourth argument to the intrinsic (arg3).
      T4: The type of the fifth argument to the intrinsic (arg4).
      T5: The type of the sixth argument to the intrinsic (arg5).
      T6: The type of the seventh argument to the intrinsic (arg6).
      T7: The type of the eighth argument to the intrinsic (arg7).
      T8: The type of the ninth argument to the intrinsic (arg8).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of arg2 must be T2.
      arg3: The fourth argument to call the LLVM intrinsic with. The type of arg3 must be T3.
      arg4: The fifth argument to call the LLVM intrinsic with. The type of arg4 must be T4.
      arg5: The sixth argument to call the LLVM intrinsic with. The type of arg5 must be T5.
      arg6: The seventh argument to call the LLVM intrinsic with. The type of arg6 must be T6.
      arg7: The eighth argument to call the LLVM intrinsic with. The type of arg7 must be T7.
      arg8: The ninth argument to call the LLVM intrinsic with. The type of arg8 must be T8.

    Returns:
      The result of calling the llvm intrinsic with arg0...arg8 as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=type,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    T9: AnyTrivialRegType,
    has_side_effect: Bool = True,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
    arg9: T9,
) -> type:
    """Calls an LLVM intrinsic with ten arguments.

    Calls the LLVM intrinsic with the name intrin and return type type on
      arguments arg0, arg1, ..., arg10

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      T0: The type of the first argument to the intrinsic (arg0).
      T1: The type of the second argument to the intrinsic (arg1).
      T2: The type of the third argument to the intrinsic (arg2).
      T3: The type of the fourth argument to the intrinsic (arg3).
      T4: The type of the fifth argument to the intrinsic (arg4).
      T5: The type of the sixth argument to the intrinsic (arg5).
      T6: The type of the seventh argument to the intrinsic (arg6).
      T7: The type of the eighth argument to the intrinsic (arg7).
      T8: The type of the ninth argument to the intrinsic (arg8).
      T9: The type of the tenth argument to the intrinsic (arg9).
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.


    Args:
      arg0: The first argument to call the LLVM intrinsic with. The type of arg0 must be T0.
      arg1: The second argument to call the LLVM intrinsic with. The type of arg1 must be T1.
      arg2: The third argument to call the LLVM intrinsic with. The type of arg2 must be T2.
      arg3: The fourth argument to call the LLVM intrinsic with. The type of arg3 must be T3.
      arg4: The fifth argument to call the LLVM intrinsic with. The type of arg4 must be T4.
      arg5: The sixth argument to call the LLVM intrinsic with. The type of arg5 must be T5.
      arg6: The seventh argument to call the LLVM intrinsic with. The type of arg6 must be T6.
      arg7: The eighth argument to call the LLVM intrinsic with. The type of arg7 must be T7.
      arg8: The ninth argument to call the LLVM intrinsic with. The type of arg8 must be T8.
      arg9: The tenth argument to call the LLVM intrinsic with. The type of arg9 must be T9.

    Returns:
      The result of calling the llvm intrinsic with arg0...arg9 as arguments.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=None
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
            return rebind[type](None)
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=None,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
        return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value, _type=type
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = intrin.value,
            _type=type,
            hasSideEffects = __mlir_attr.false,
        ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)


# ===----------------------------------------------------------------------===#
# _gather
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn gather[
    type: DType, size: Int
](
    base: SIMD[DType.address, size],
    mask: SIMD[DType.bool, size],
    passthrough: SIMD[type, size],
    alignment: Int = 0,
) -> SIMD[type, size]:
    """Reads scalar values from a SIMD vector, and gathers them into one vector.

    The gather function reads scalar values from a SIMD vector of memory
    locations and gathers them into one vector. The memory locations are
    provided in the vector of pointers `base` as addresses. The memory is
    accessed according to the provided mask. The mask holds a bit for each
    vector lane, and is used to prevent memory accesses to the masked-off
    lanes. The masked-off lanes in the result vector are taken from the
    corresponding lanes of the `passthrough` operand.

    In general, for some vector of pointers `base`, mask `mask`, and passthrough
    `pass` a call of the form:

    ```python
    gather(base, mask, pass)
    ```

    is equivalent to the following sequence of scalar loads in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      result[i] = mask[i] ? *base[i] : passthrough[i];
    ```

    Parameters:
      type: DType of the return SIMD buffer.
      size: Size of the return SIMD buffer.

    Args:
      base: The vector containing memory addresses that gather will access.
      mask: A binary vector which prevents memory access to certain lanes of
        the base vector.
      passthrough: In the result vector, the masked-off lanes are replaced
        with the passthrough vector.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value.

    Returns:
      A SIMD[type, size] containing the result of the gather operation.
    """

    @parameter
    if size == 1:
        return DTypePointer[type](base[0]).load() if mask else passthrough[0]
    return llvm_intrinsic[
        "llvm.masked.gather",
        __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`],
    ](
        base,
        Int32(alignment),
        mask,
        passthrough,
    )


# ===----------------------------------------------------------------------===#
# _scatter
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn scatter[
    type: DType, size: Int
](
    value: SIMD[type, size],
    base: SIMD[DType.address, size],
    mask: SIMD[DType.bool, size],
    alignment: Int = 0,
):
    """Takes scalar values from a SIMD vector and `scatters` them into a
    vector of pointers.

    The scatter operation stores scalar values from a SIMD vector of memory
    locations and scatters them into a vector of pointers. The memory locations
    are provided in the vector of pointers `base` as addresses. The memory is
    stored according to the provided mask. The mask holds a bit for each vector
    lane, and is used to prevent memory accesses to the masked-off lanes.

    The `value` operand is a vector value to be written to memory. The `base`
    operand is a vector of pointers, pointing to where the value elements
    should be stored. It has the same underlying type as the value operand. The
    `mask` operand, mask, is a vector of boolean values. The types of the
    `mask` and the `value` operand must have the same number of vector
    elements.

    Scatter with overlapping addresses is guaranteed to be ordered from
    least-significant to most-significant element.

    In general, for some vector %value, vector of pointers %base, and mask
    %mask instructions of the form:

    ```mlir
    %0 = pop.simd.scatter %value, %base[%mask] : !pop.simd<N, type>
    ```

    is equivalent to the following sequence of scalar loads in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      if (mask[i])
        base[i] = value[i];
    ```

    Parameters:
      type: DType of `value`, the result SIMD buffer.
      size: Size of `value`, the result SIMD buffer.

    Args:
      value: The vector that will contain the result of the scatter operation.
      base: The vector containing memory addresses that scatter will access.
      mask: A binary vector which prevents memory access to certain lanes of
        the base vector.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value.
    """

    @parameter
    if size == 1:
        if mask:
            var ptr = DTypePointer[type](base[0])
            ptr.store(value[0])
        return
    llvm_intrinsic["llvm.masked.scatter", NoneType](
        value,
        base,
        Int32(alignment),
        mask,
    )


# ===----------------------------------------------------------------------===#
# prefetch
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct PrefetchLocality:
    """The prefetch locality.

    The locality, rw, and cache type correspond to LLVM prefetch intrinsic's
    inputs (see
    [LLVM prefetch locality](https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic))
    """

    var value: Int32
    """The prefetch locality to use. It should be a value in [0, 3]."""
    alias NONE = PrefetchLocality(0)
    """No locality."""
    alias LOW = PrefetchLocality(1)
    """Low locality."""
    alias MEDIUM = PrefetchLocality(2)
    """Medium locality."""
    alias HIGH = PrefetchLocality(3)
    """Extremely local locality (keep in cache)."""

    @always_inline("nodebug")
    fn __init__(value: Int) -> PrefetchLocality:
        """Constructs a prefetch locality option.

        Args:
            value: An integer value representing the locality. Should be a value
                   in the range `[0, 3]`.

        Returns:
            The prefetch locality constructed.
        """
        return PrefetchLocality {value: value}


@register_passable("trivial")
struct PrefetchRW:
    """Prefetch read or write."""

    var value: Int32
    """The read-write prefetch. It should be in [0, 1]."""
    alias READ = PrefetchRW(0)
    """Read prefetch."""
    alias WRITE = PrefetchRW(1)
    """Write prefetch."""

    @always_inline("nodebug")
    fn __init__(value: Int) -> PrefetchRW:
        """Constructs a prefetch read-write option.

        Args:
            value: An integer value representing the prefetch read-write option
                   to be used. Should be a value in the range `[0, 1]`.

        Returns:
            The prefetch read-write option constructed.
        """
        return PrefetchRW {value: value}


# LLVM prefetch cache type
@register_passable("trivial")
struct PrefetchCache:
    """Prefetch cache type."""

    var value: Int32
    """The cache prefetch. It should be in [0, 1]."""
    alias INSTRUCTION = PrefetchCache(0)
    """The instruction prefetching option."""
    alias DATA = PrefetchCache(1)
    """The data prefetching option."""

    @always_inline("nodebug")
    fn __init__(value: Int) -> PrefetchCache:
        """Constructs a prefetch option.

        Args:
            value: An integer value representing the prefetch cache option to be
                   used. Should be a value in the range `[0, 1]`.

        Returns:
            The prefetch cache type that was constructed.
        """
        return PrefetchCache {value: value}


@register_passable("trivial")
struct PrefetchOptions:
    """Collection of configuration parameters for a prefetch intrinsic call.

    The op configuration follows similar interface as LLVM intrinsic prefetch
    op, with a "locality" attribute that specifies the level of temporal locality
    in the application, that is, how soon would the same data be visited again.
    Possible locality values are: `NONE`, `LOW`, `MEDIUM`, and `HIGH`.

    The op also takes a "cache tag" attribute giving hints on how the
    prefetched data will be used. Possible tags are: `ReadICache`, `ReadDCache`
    and `WriteDCache`.

    Note: the actual behavior of the prefetch op and concrete interpretation of
    these attributes are target-dependent.
    """

    var rw: PrefetchRW
    """Indicates prefetching for read or write."""
    var locality: PrefetchLocality
    """Indicates locality level."""
    var cache: PrefetchCache
    """Indicates i-cache or d-cache prefetching."""

    @always_inline("nodebug")
    fn __init__(inout self):
        """Constructs an instance of PrefetchOptions with default params."""
        self.rw = PrefetchRW.READ
        self.locality = PrefetchLocality.HIGH
        self.cache = PrefetchCache.DATA

    @always_inline("nodebug")
    fn for_read(self) -> Self:
        """
        Sets the prefetch purpose to read.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.rw = PrefetchRW.READ
        return updated

    @always_inline("nodebug")
    fn for_write(self) -> Self:
        """
        Sets the prefetch purpose to write.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.rw = PrefetchRW.WRITE
        return updated

    @always_inline("nodebug")
    fn no_locality(self) -> Self:
        """
        Sets the prefetch locality to none.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.NONE
        return updated

    @always_inline("nodebug")
    fn low_locality(self) -> Self:
        """
        Sets the prefetch locality to low.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.LOW
        return updated

    @always_inline("nodebug")
    fn medium_locality(self) -> Self:
        """
        Sets the prefetch locality to medium.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.MEDIUM
        return updated

    @always_inline("nodebug")
    fn high_locality(self) -> Self:
        """
        Sets the prefetch locality to high.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.HIGH
        return updated

    @always_inline("nodebug")
    fn to_data_cache(self) -> Self:
        """
        Sets the prefetch target to data cache.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.cache = PrefetchCache.DATA
        return updated

    @always_inline("nodebug")
    fn to_instruction_cache(self) -> Self:
        """
        Sets the prefetch target to instruction cache.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.cache = PrefetchCache.INSTRUCTION
        return updated


@always_inline("nodebug")
fn prefetch[
    params: PrefetchOptions, type: DType, address_space: AddressSpace
](addr: DTypePointer[type, address_space]):
    """Prefetches an instruction or data into cache before it is used.

    The prefetch function provides prefetching hints for the target
    to prefetch instruction or data into cache before they are used.

    Parameters:
      params: Configuration options for the prefect intrinsic.
      type: The DType of value stored in addr.
      address_space: The address space of the pointer.

    Args:
      addr: The data pointer to prefetch.
    """
    return llvm_intrinsic["llvm.prefetch", NoneType](
        addr.bitcast[DType.invalid.value](),
        params.rw,
        params.locality,
        params.cache,
    )


# ===----------------------------------------------------------------------===#
# masked load
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn masked_load[
    size: Int
](
    addr: DTypePointer,
    mask: SIMD[DType.bool, size],
    passthrough: SIMD[addr.type, size],
    alignment: Int = 1,
) -> SIMD[addr.type, size]:
    """Loads data from memory and return it, replacing masked lanes with values
    from the passthrough vector.

    Parameters:
      size: Size of the return SIMD buffer.

    Args:
      addr: The base pointer for the load.
      mask: A binary vector which prevents memory access to certain lanes of
        the memory stored at addr.
      passthrough: In the result vector, the masked-off lanes are replaced
        with the passthrough vector.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value. Default is 1.

    Returns:
      The loaded memory stored in a vector of type SIMD[type, size].
    """

    @parameter
    if size == 1:
        return addr.load() if mask else passthrough[0]

    return llvm_intrinsic["llvm.masked.load", SIMD[addr.type, size]](
        addr.bitcast[DType.invalid.value]().address,
        Int32(alignment),
        mask,
        passthrough,
    )


# ===----------------------------------------------------------------------===#
# masked store
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn masked_store[
    size: Int
](
    value: SIMD,
    addr: DTypePointer[value.type],
    mask: SIMD[DType.bool, size],
    alignment: Int = 1,
):
    """Stores a value at a memory location, skipping masked lanes.

    Parameters:
      size: Size of `value`, the data to store.

    Args:
      value: The vector containing data to store.
      addr: A vector of memory location to store data at.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
      alignment: The alignment of the destination locations. Must be 0 or a
        power of two constant integer value.
    """

    @parameter
    if size == 1:
        if mask:
            addr.store(value[0])
        return

    llvm_intrinsic["llvm.masked.store", NoneType](
        value,
        addr.bitcast[DType.invalid.value]().address,
        Int32(alignment),
        mask,
    )


# ===----------------------------------------------------------------------===#
# compressed store
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn compressed_store[
    type: DType, size: Int
](
    value: SIMD[type, size],
    addr: DTypePointer[type],
    mask: SIMD[DType.bool, size],
):
    """Compresses the lanes of `value`, skipping `mask` lanes, and stores
    at `addr`.

    Parameters:
      type: DType of `value`, the value to store.
      size: Size of `value`, the value to store.

    Args:
      value: The vector containing data to store.
      addr: The memory location to store the compressed data.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
    """

    @parameter
    if size == 1:
        if mask:
            addr.store(value[0])
        return

    llvm_intrinsic["llvm.masked.compressstore", NoneType](
        value,
        addr.bitcast[DType.invalid.value]().address,
        mask,
    )


# ===----------------------------------------------------------------------===#
# strided load
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn strided_load[
    type: DType,
    simd_width: Int,
    /,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    addr: DTypePointer[type, address_space],
    stride: Int,
    mask: SIMD[DType.bool, simd_width],
) -> SIMD[type, simd_width]:
    """Loads values from addr according to a specific stride.

    Parameters:
      type: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.
      address_space: The address space of the memory location.

    Args:
      addr: The memory location to load data from.
      stride: How many lanes to skip before loading again.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.

    Returns:
      A vector containing the loaded data.
    """

    @parameter
    if simd_width == 1:
        return addr.load() if mask else Scalar[type]()

    alias IndexTy = SIMD[DType.index, simd_width]
    var iota = llvm_intrinsic[
        "llvm.experimental.stepvector", IndexTy, has_side_effect=False
    ]()
    var offset = IndexTy(int(addr)) + IndexTy(stride) * iota * IndexTy(
        sizeof[type]()
    )
    var passthrough = SIMD[type, simd_width]()
    return gather[type, simd_width](
        offset.cast[DType.address](), mask, passthrough
    )


@always_inline("nodebug")
fn strided_load[
    type: DType,
    simd_width: Int,
    /,
    address_space: AddressSpace = AddressSpace.GENERIC,
](addr: DTypePointer[type, address_space], stride: Int) -> SIMD[
    type, simd_width
]:
    """Loads values from addr according to a specific stride.

    Parameters:
      type: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.
      address_space: The address space of the memory location.

    Args:
      addr: The memory location to load data from.
      stride: How many lanes to skip before loading again.

    Returns:
      A vector containing the loaded data.
    """

    @parameter
    if simd_width == 1:
        return addr.load()

    return strided_load[type, simd_width](addr, stride, True)


# ===----------------------------------------------------------------------===#
# strided store
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn strided_store[
    type: DType,
    simd_width: Int,
    /,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    value: SIMD[type, simd_width],
    addr: DTypePointer[type, address_space],
    stride: Int,
    mask: SIMD[DType.bool, simd_width],
):
    """Loads values from addr according to a specific stride.

    Parameters:
      type: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.
      address_space: The address space of the memory location.

    Args:
      value: The values to store.
      addr: The location to store values at.
      stride: How many lanes to skip before storing again.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
    """

    @parameter
    if simd_width == 1:
        if mask:
            addr.store(value[0])
        return

    alias IndexTy = SIMD[DType.index, simd_width]
    var iota = llvm_intrinsic[
        "llvm.experimental.stepvector", IndexTy, has_side_effect=False
    ]()
    var offset = IndexTy(int(addr)) + IndexTy(stride) * iota * IndexTy(
        sizeof[type]()
    )

    scatter[type, simd_width](value, offset.cast[DType.address](), mask)


@always_inline("nodebug")
fn strided_store[
    type: DType,
    simd_width: Int,
    /,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    value: SIMD[type, simd_width],
    addr: DTypePointer[type, address_space],
    stride: Int,
):
    """Loads values from addr according to a specific stride.

    Parameters:
      type: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.
      address_space: The address space of the memory location.

    Args:
      value: The values to store.
      addr: The location to store values at.
      stride: How many lanes to skip before storing again.
    """

    @parameter
    if simd_width == 1:
        addr.store(value[0])
        return

    strided_store[type, simd_width](value, addr, stride, True)


# ===-------------------------------------------------------------------===#
# _mlirtype_is_eq
# ===-------------------------------------------------------------------===#


fn _mlirtype_is_eq[t1: AnyTrivialRegType, t2: AnyTrivialRegType]() -> Bool:
    """Compares the two type for equality.

    Parameters:
        t1: The LHS of the type comparison.
        t2: The RHS of the type comparison.

    Returns:
        Returns True if t1 and t2 are the same type and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        `#kgen.parameterizedtype.constant<`,
        t1,
        `> : !kgen.type`,
        `,`,
        `#kgen.parameterizedtype.constant<`,
        t2,
        `> : !kgen.type`,
        `> : i1`,
    ]


fn _type_is_eq[t1: AnyType, t2: AnyType]() -> Bool:
    """Compares the two type for equality.

    Parameters:
        t1: The LHS of the type comparison.
        t2: The RHS of the type comparison.

    Returns:
        Returns True if t1 and t2 are the same type and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        `#kgen.parameterizedtype.constant<`,
        +t1,
        `> : !kgen.type`,
        `,`,
        `#kgen.parameterizedtype.constant<`,
        +t2,
        `> : !kgen.type`,
        `> : i1`,
    ]


# ===----------------------------------------------------------------------=== #
# Transitional type used for llvm_intrinsic
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _RegisterPackType[*a: AnyTrivialRegType]:
    var storage: __mlir_type[`!kgen.pack<`, a, `>`]

    @always_inline("nodebug")
    fn __getitem__[i: Int](self) -> a[i.value]:
        """Get the element.

        Parameters:
            i: The element index.

        Returns:
            The tuple element at the requested index.
        """
        return __mlir_op.`kgen.pack.extract`[index = i.value](self.storage)
