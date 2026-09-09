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
"""Dispatch for fused GEMM+SwiGLU on SM100.

Accepts a ``FusedSwiGLUMatmulConfig`` and launches the SwiGLU kernel. The
caller is responsible for choosing the config (MMA shape, pipeline stages,
etc.).
"""

from std.math import align_up
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import PDLLevel
from layout import Coord, Idx, DefaultEngine, TileTensor, row_major
from std.collections import OptionalReg

from std.utils.index import Index

from internal_utils import Table

from .config import (
    FusedSwiGLUMatmulConfig,
    swiglu_matmul_config,
    choose_swiglu_config,
    build_sm100_swiglu_heuristic_configs,
)
from .matmul_swiglu import _blackwell_matmul_swiglu
from .tuning_configs import TuningConfigSwiGLU, _get_tuning_list_swiglu_bf16


def matmul_swiglu_dispatch_sm100[
    config: FusedSwiGLUMatmulConfig[_, _, _, True],
    pdl_level: PDLLevel = PDLLevel(0),
](
    c_out: TileTensor[
        mut=True, .bfloat16, Engine=DefaultEngine[element_width=1], ...
    ],
    a: TileTensor[mut=False, .bfloat16, ...],
    b: TileTensor[mut=False, .bfloat16, ...],
    ctx: DeviceContext,
    bias_ptr: OptionalReg[UnsafePointer[BFloat16, ImmutAnyOrigin]] = None,
) raises:
    """Dispatch fused GEMM+SwiGLU to SM100 kernel with given config.

    Parameters:
        config: Fully specialized ``FusedSwiGLUMatmulConfig`` carrying the
              MMA shape, cluster shape, swizzle modes, pipeline stage counts,
              ``AB_swapped`` orientation, and ``use_bias`` flag used to
              specialize the kernel.
        pdl_level: Programmatic Dependency Launch level controlling
              inter-grid overlap on the launch (defaults to ``PDLLevel.OFF``).

    Args:
        c_out: [M, H] BF16 output tensor (H = N/2).
        a: [M, K] BF16 input activations.
        b: [N, K] BF16 pre-permuted weights (N = 2H, transposed).
              When ``config.AB_swapped`` is False, ``b`` is permuted on its
              N axis so adjacent (gate, up) pairs are at rows (2i, 2i+1).
              When ``config.AB_swapped`` is True, ``b`` is permuted on its
              N axis with stride-8 row blocks (see kernel docs for the
              exact formula).
        ctx: Device context for kernel launch.
        bias_ptr: Optional [N=2H] BF16 bias (interleaved gate/up pairs).
              Ignored when ``config.use_bias`` is False.
    """
    # When config.use_bias=False, c_out._storage is a valid dummy (bias never
    # accessed).
    var bias_ptr_c = rebind[
        UnsafePointer[Scalar[config.c_type], ImmutAnyOrigin]
    ](c_out._storage)
    comptime if config.use_bias:
        bias_ptr_c = rebind[
            UnsafePointer[Scalar[config.c_type], ImmutAnyOrigin]
        ](bias_ptr.value())
    var bias_tile = TileTensor(bias_ptr_c, row_major(Coord(Int(b.dim[0]()))))
    comptime if config.AB_swapped:
        # Swap inputs so the kernel sees A = W (b) and B = X (a). The
        # inner implementation reads kernel-frame M from a_device.dim[0]
        # and kernel-frame N from b_device.dim[0].
        _blackwell_matmul_swiglu[
            transpose_b=True,
            config=config,
            pdl_level=pdl_level,
        ](c_out, b, a, ctx, OptionalReg(bias_tile))
    else:
        _blackwell_matmul_swiglu[
            transpose_b=True,
            config=config,
            pdl_level=pdl_level,
        ](c_out, a, b, ctx, OptionalReg(bias_tile))


def matmul_swiglu_dispatch_sm100_bf16[
    pdl_level: PDLLevel = PDLLevel(0),
    has_bias: Bool = False,
](
    c_out: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
    a: TileTensor[...],
    b: TileTensor[...],
    ctx: DeviceContext,
    bias_ptr: OptionalReg[UnsafePointer[BFloat16, ImmutAnyOrigin]] = None,
) raises:
    """Auto-dispatch fused GEMM+SwiGLU on SM100 using shape-based tuning table.

    Selects a ``FusedSwiGLUMatmulConfig`` from ``_get_tuning_list_swiglu_bf16``
    by matching the static (N, K) from the weight matrix ``b``, then checking
    the runtime M. Falls back to a safety-net config for untuned shapes.

    N is read from ``b.static_shape[0]`` (the full pre-SwiGLU width), not
    from ``c_out.static_shape[1]`` (which holds H = N/2).

    Parameters:
        pdl_level: Programmatic Dependency Launch level controlling
              inter-grid overlap on the launch (defaults to ``PDLLevel.OFF``).
        has_bias: Whether the kernel consumes ``bias_ptr`` as a bias vector
              (defaults to False). When False, ``bias_ptr`` is ignored and
              the selected config is built with ``use_bias=False``.

    Args:
        c_out: [M, H] BF16 output tensor (H = N/2).
        a: [M, K] BF16 input activations.
        b: [N, K] BF16 weights (N = 2H, pre-permuted, transposed).
        ctx: Device context for kernel launch.
        bias_ptr: Optional [N=2H] BF16 bias (interleaved gate/up pairs).
              Only used when ``has_bias`` is True.
    """
    var m = Int(c_out.dim[0]())
    comptime dtype = DType.bfloat16
    comptime static_N = b.static_shape[0]
    comptime static_K = b.static_shape[1]
    # When has_bias=False, c_out._storage is a valid dummy (bias never accessed).
    var bias_base = rebind[UnsafePointer[BFloat16, ImmutAnyOrigin]](
        c_out._storage
    )
    comptime if has_bias:
        bias_base = bias_ptr.value()
    var bias_tile = TileTensor(bias_base, row_major(Coord(Idx[static_N])))
    comptime assert static_N % 2 == 0, "N must be even for SwiGLU gate/up split"

    comptime tuning_table = Table(
        _get_tuning_list_swiglu_bf16(), "swiglu_bf16_tuning"
    )

    comptime nk_configs = tuning_table.find(
        rule=lambda (x: TuningConfigSwiGLU) -> Bool: x.N == static_N
        and x.K == static_K
    )

    comptime for tc in nk_configs:
        if m >= tc.M and m < tc.M_end:
            comptime config = FusedSwiGLUMatmulConfig[
                dtype, dtype, dtype, True
            ](
                mma_shape=tc.mma_shape,
                cluster_shape=tc.cluster_shape,
                block_swizzle_size=tc.block_swizzle_size,
                raster_order=tc.rasterize_order,
                cta_group=tc.cta_group,
                AB_swapped=tc.swapAB,
                k_group_size=tc.k_group_size,
                num_accum_pipeline_stages=tc.num_accum_pipeline_stages,
                num_clc_pipeline_stages=tc.num_clc_pipeline_stages,
                use_bias=has_bias,
                register_swiglu=tc.register_swiglu,
            )
            comptime if tc.swapAB:
                _blackwell_matmul_swiglu[
                    transpose_b=True,
                    config=config,
                    pdl_level=pdl_level,
                ](c_out, b, a, ctx, OptionalReg(bias_tile))
            else:
                _blackwell_matmul_swiglu[
                    transpose_b=True,
                    config=config,
                    pdl_level=pdl_level,
                ](c_out, a, b, ctx, OptionalReg(bias_tile))
            return

    # Heuristic fallback for untuned (N, K) shapes.
    # Mirrors the normal matmul: build the compile-time set of all configs
    # reachable by the heuristic, run it at runtime for the actual M, then
    # match the runtime value against a compile-time constant so the kernel
    # is still a distinct specialization per config.
    comptime heuristic_configs = build_sm100_swiglu_heuristic_configs[
        dtype, dtype, dtype, static_N, static_K, has_bias=has_bias
    ]()
    var aligned_m = align_up(m, 64) if m >= 256 else m
    var config_runtime = choose_swiglu_config[
        dtype, dtype, dtype, True, has_bias
    ](aligned_m, static_N, static_K)

    comptime for config in heuristic_configs:
        if config_runtime == config:
            comptime if config.AB_swapped:
                _blackwell_matmul_swiglu[
                    transpose_b=True,
                    config=config,
                    pdl_level=pdl_level,
                ](c_out, b, a, ctx, OptionalReg(bias_tile))
            else:
                _blackwell_matmul_swiglu[
                    transpose_b=True,
                    config=config,
                    pdl_level=pdl_level,
                ](c_out, a, b, ctx, OptionalReg(bias_tile))
            return

    # Last-resort hardcoded config if heuristic set is exhausted (e.g. M > 8192).
    comptime default_config = swiglu_matmul_config[dtype, dtype, dtype, True](
        mma_shape=Index(256, 224, 16),
        cluster_shape=Index(2, 1, 1),
        block_swizzle_size=0,
        cta_group=2,
        AB_swapped=False,
        use_bias=has_bias,
    )
    _blackwell_matmul_swiglu[
        transpose_b=True,
        config=default_config,
        pdl_level=pdl_level,
    ](c_out, a, b, ctx, OptionalReg(bias_tile))
