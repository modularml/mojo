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
"""Provides inline assembly support for low-level hardware control.

You can import these APIs from the `sys` package. For example:

```mojo
from std.sys import inlined_assembly
```
"""

from std.collections.string.string_span import _get_kgen_string


@always_inline("nodebug")
def inlined_assembly[
    asm: StaticString,
    result_type: TrivialRegisterPassable,
    *types: AnyType,
    constraints: StaticString,
    has_side_effect: Bool = True,
](*args: *types) -> result_type:
    """Generates inline assembly code with the given constraints and arguments.

    This function allows embedding raw assembly instructions directly into Mojo
    code, providing fine-grained control over hardware operations. It uses
    LLVM-style inline assembly syntax and constraint strings.

    The assembly string uses `$0`, `$1`, etc. to reference operands. Output
    operands (including the return value) are numbered first, followed by input
    operands.

    Example:

    ```mojo
    from std.sys import inlined_assembly

    # Convert bfloat16 to float32 on NVIDIA GPU using PTX assembly.
    # "$0" is the output (float32), "$1" is the input (int16 bitcast of bf16).
    var my_bf16_as_int16 = Int16(0x3F80)  # Example bf16 bit pattern
    var result = inlined_assembly[
        "cvt.f32.bf16 $0, $1;",
        Float32,
        constraints="=f,h",
        has_side_effect=False,
    ](my_bf16_as_int16)

    # Execute a no-op sleep instruction on AMD GPU (no return value).
    inlined_assembly[
        "s_sleep 0",
        NoneType,
        constraints="",
        has_side_effect=True,
    ]()
    ```

    Parameters:
        asm: The assembly instruction string. Use `$0`, `$1`, etc. to reference
            operands, where `$0` is the output (if any) and subsequent numbers
            are inputs in order.
        result_type: The return type of the assembly operation. Use `NoneType`
            for assembly that produces no result.
        types: The types of the input arguments.
        constraints: LLVM-style constraint string specifying register allocation
            and operand placement. The output constraint comes first (prefixed
            with `=`), followed by input constraints separated by commas.
            Available constraints are target-specific; refer to LLVM's inline
            assembly documentation and your target's backend for valid options.
        has_side_effect: If `True` (default), the assembly is treated as having
            side effects and won't be optimized away or reordered. Set to
            `False` for pure computations to enable compiler optimizations.

    Args:
        args: The input arguments to pass to the assembly instruction.

    Returns:
        The result of the assembly operation, or `None` if `result_type` is
        `NoneType`."""
    var loaded_pack = args.get_loaded_kgen_pack()

    comptime asm_kgen_string = _get_kgen_string[asm]()
    comptime constraints_kgen_string = _get_kgen_string[constraints]()

    comptime if result_type == NoneType:
        __mlir_op.`pop.inline_asm`[
            _type=None,
            assembly=asm_kgen_string,
            constraints=constraints_kgen_string,
            hasSideEffects=has_side_effect.__mlir_i1__(),
            isStackAligned=False.__mlir_i1__(),
        ](loaded_pack)
        return rebind[result_type](None)
    else:
        return __mlir_op.`pop.inline_asm`[
            _type=result_type,
            assembly=asm_kgen_string,
            constraints=constraints_kgen_string,
            hasSideEffects=has_side_effect.__mlir_i1__(),
            isStackAligned=False.__mlir_i1__(),
        ](loaded_pack)
