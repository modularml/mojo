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
"""Provides the backend implementation for matmuls."""


from std.collections import OptionalReg
from std.collections.string.string_span import get_static_string
from std.math import align_up, ceildiv
from std.sys.info import align_of, simd_width_of

from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_valid_target
from layout import (
    Layout,
    LayoutTensor,
    RowMajorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
)
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList

from . import cpu
from ..gemv import gemv
from ..utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .gpu import _matmul_gpu


@always_inline
def matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
    # Kept last so existing positional-parameter callers (e.g. the mo.matmul
    # op) are unaffected; set it by keyword.
    use_tf32: Bool = True,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[address_space=.GENERIC, ...],
    b: TileTensor[address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Primary TileTensor matmul implementation. Routes GPU directly, delegates
    CPU path to cpu.matmul.

    `use_tf32=False` (GPU only) requires IEEE-fp32 multiplies for fp32
    inputs instead of TF32 tensor-core truncation; see `_matmul_gpu`. The
    CPU path is always IEEE fp32.

    Parameters:
        transpose_a: Transpose `a` before the matmul (defaults to `False`);
            currently unsupported.
        transpose_b: Transpose `b` before the matmul (defaults to `False`).
        b_packed: `b` is already in a CPU-friendly packed layout (CPU only;
            defaults to `False`).
        elementwise_lambda_fn: Epilogue lambda applied to each stored tile of
            `c` (defaults to `None`).
        elementwise_compute_lambda_fn: Compute lambda transforming each result
            tile before store; on CPU it is wrapped as an epilogue (defaults
            to `None`).
        saturated_vnni: Use the saturating VNNI variant on x86 (CPU only;
            defaults to `False`).
        _trace_description: Extra string folded into the op trace label
            (defaults to an empty string).
        target: Target platform string such as `"cpu"` or a GPU target
            (defaults to `"cpu"`).
        use_tf32: On GPU, allow TF32 tensor-core truncation for fp32 inputs
            (defaults to `True`).

    Args:
        c: Rank-2 output `TileTensor` accumulating the matmul result.
        a: Rank-2 LHS `TileTensor` of the matmul.
        b: Rank-2 RHS `TileTensor` of the matmul.
        ctx: Optional `DeviceContext` required for GPU targets and used for
            CPU tracing (defaults to `None`).
    """
    comptime assert c.rank == 2, "c must be rank 2"
    comptime assert a.rank == 2, "a must be rank 2"
    comptime assert b.rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must have a non-nested layout"
    comptime assert a.flat_rank == 2, "a must have a non-nested layout"
    comptime assert b.flat_rank == 2, "b must have a non-nested layout"

    comptime if not is_cpu[target]():
        # GPU path: call _matmul_gpu directly with tracing. CPU-only params
        # (b_packed, saturated_vnni) are intentionally not forwarded here.
        comptime assert not transpose_a, "transpose_a not yet supported"
        assert Bool(ctx), "expected DeviceContext for GPU target"

        if Int(c.dim[0]()) == 0 or Int(c.dim[1]()) == 0:
            return

        @always_inline
        def description_fn() {imm} -> String:
            var shape = GemmShape.get[transpose_b](c, a, b)
            # fmt: off
            return String(
                "(",
                target,
                ";", trace_arg("A", IndexList[2](shape.M, shape.K), a.dtype),
                ";", trace_arg("B", IndexList[2](shape.K, shape.N), b.dtype),
                ";", trace_arg("C", IndexList[2](shape.M, shape.N), c.dtype),
                ";transpose_a=", transpose_a,
                ";transpose_b=", transpose_b,
                ")"
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            get_static_string[
                "matmul",
                _trace_description if _trace_description else "",
            ](),
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=OptionalReg(Int(ctx.value().id())),
        ):
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                use_tf32=use_tf32,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            ](c, a, b, ctx.value())
    else:
        # CPU path: handle tracing and compute lambda wrapping, then
        # delegate to TileTensor cpu.matmul overload.
        comptime assert is_valid_target[target](), "unsupported target"
        comptime assert not transpose_a, "transpose_a not yet supported"

        if Int(c.dim[0]()) == 0 or Int(c.dim[1]()) == 0:
            return

        @always_inline
        def cpu_description_fn() {imm} -> String:
            var shape = GemmShape.get[transpose_b](c, a, b)
            # fmt: off
            return String(
                "(",
                target,
                ";", trace_arg("A", IndexList[2](shape.M, shape.K), a.dtype),
                ";", trace_arg("B", IndexList[2](shape.K, shape.N), b.dtype),
                ";", trace_arg("C", IndexList[2](shape.M, shape.N), c.dtype),
                ";transpose_a=", transpose_a,
                ";transpose_b=", transpose_b,
                ";b_packed=", b_packed,
                ")"
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            get_static_string[
                "matmul",
                _trace_description if _trace_description else "",
            ](),
            Trace[TraceLevel.OP]._get_detail_str(cpu_description_fn),
            task_id=OptionalReg(Int(ctx.value().id())) if ctx else None,
        ):
            var kernel_type_m = (
                a.static_shape[0] if a.static_shape[0] > -1 else 0
            )

            # The CPU version of matmul doesn't support compute lambda.
            # Wrap it around an epilogue lambda instead.
            @__parameter
            @always_inline
            def compute_lambda_wrapper[
                _type: DType, _width: SIMDLength, *, alignment: Int = 1
            ](coords: IndexList[2], val: SIMD[_type, _width]):
                comptime if elementwise_compute_lambda_fn:
                    comptime compute_lambda = elementwise_compute_lambda_fn.value()
                    var output = compute_lambda(coords, val)
                    c.store_linear[alignment=alignment](
                        coords, rebind[SIMD[c.dtype, _width]](output)
                    )

            comptime elementwise_lambda_wrapper = Optional[
                elementwise_epilogue_type
            ](
                compute_lambda_wrapper
            ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

            cpu.matmul[
                transpose_b=transpose_b,
                b_packed=b_packed,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                saturated_vnni=saturated_vnni,
            ](c, a, b, kernel_type_m, ctx=ctx)
