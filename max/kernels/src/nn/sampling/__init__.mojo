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
"""Token-sampling kernels: penalties, top-k/top-p masking and sampling.

`topk_fi` holds the single-block kernels, which serve every target;
`topk_fi_cluster` holds thread-block-cluster variants of the hot spec-decode
kernels for NVIDIA SM90+ devices. The dispatchers in this file pick between
them by device, so callers import from this package and never choose a
launch shape themselves.
"""

from max.gpu.host import DeviceContext
from std.sys.info import _has_sm_9x_or_newer
from layout import (
    ComptimeInt,
    Coord,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
)
from layout.tile_layout import Layout

from .sampling import (
    apply_penalties_to_logits,
    update_frequency_data,
)
from .topk_fi import (
    topk_mask_logits,
    topk_sampling_from_prob,
    topk_softmax_sample,
)
from .topk_fi import topk_topp_masked_probs as _topk_topp_masked_probs_single
from .topk_fi import (
    topk_topp_sampling_from_prob as _topk_topp_sampling_from_prob_single,
)
from .topk_fi_cluster import (
    topk_topp_masked_probs_cluster,
    topk_topp_sampling_from_prob_cluster,
)


def topk_topp_masked_probs[
    dtype: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    ProbsLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
    TopKArrEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPArrEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    logits: TileTensor[mut=False, dtype, ...],
    probs: TileTensor[.float32, ProbsLayoutType, MutAnyOrigin],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    top_k_arr: Optional[
        TileTensor[
            .int64, TopKArrLayoutType, ImmutAnyOrigin, Engine=TopKArrEngine
        ]
    ] = None,
    top_p_arr: Optional[
        TileTensor[
            .float32, TopPArrLayoutType, ImmutAnyOrigin, Engine=TopPArrEngine
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
) raises:
    """Computes per-row top-k/top-p masked softmax.

    Dispatches by device: NVIDIA SM90+ takes the cluster-capable launcher,
    every other target takes the single-block one. The branch is a comptime
    one, so a target without thread-block clusters never instantiates the
    cluster kernels.

    See `topk_fi.topk_topp_masked_probs` for the parameters, arguments and
    output contract; both sides share them.
    """
    comptime if _has_sm_9x_or_newer():
        topk_topp_masked_probs_cluster[
            dtype,
            block_size,
            TopKArrLayoutType,
            TopPArrLayoutType,
            TemperatureLayoutType,
            ProbsLayoutType,
            TopKArrEngine,
            TopPArrEngine,
            TemperatureEngine,
        ](
            ctx,
            logits,
            probs,
            top_k_val,
            top_p_val,
            top_k_arr,
            top_p_arr,
            temperature,
        )
    else:
        _topk_topp_masked_probs_single[
            dtype,
            block_size,
            TopKArrLayoutType,
            TopPArrLayoutType,
            TemperatureLayoutType,
            ProbsLayoutType,
            TopKArrEngine,
            TopPArrEngine,
            TemperatureEngine,
        ](
            ctx,
            logits,
            probs,
            top_k_val,
            top_p_val,
            top_k_arr,
            top_p_arr,
            temperature,
        )


def topk_topp_sampling_from_prob[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    from_logits: Bool = False,
    emit_dist: Bool = False,
    dist_dtype: DType = .float32,
    DistLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    IndicesLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    SeedLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    MinPLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopKArrEngine: TensorEngine = DefaultEngine[element_width=1],
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPArrEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    MinPEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    probs: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, out_idx_type, ...],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    deterministic: Bool = False,
    rng_seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
    rng_offset: UInt64 = 0,
    indices: Optional[
        TileTensor[
            out_idx_type,
            IndicesLayoutType,
            ImmutAnyOrigin,
            Engine=IndicesEngine,
        ]
    ] = None,
    top_k_arr: Optional[
        TileTensor[
            out_idx_type,
            TopKArrLayoutType,
            ImmutAnyOrigin,
            Engine=TopKArrEngine,
        ]
    ] = None,
    top_p_arr: Optional[
        TileTensor[
            .float32,
            TopPArrLayoutType,
            ImmutAnyOrigin,
            Engine=TopPArrEngine,
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    min_p: Optional[
        TileTensor[.float32, MinPLayoutType, ImmutAnyOrigin, Engine=MinPEngine]
    ] = None,
    out_dist: Optional[
        TileTensor[dist_dtype, DistLayoutType, MutAnyOrigin]
    ] = None,
) raises:
    """Joint top-k + top-p sampling from probability distribution.

    Dispatches by device: with `emit_dist` on NVIDIA SM90+, the
    cluster-capable launcher builds the emitted distribution across a
    thread-block cluster; everything else takes the single-block
    launcher. The branch is a comptime one, so a target without
    thread-block clusters never instantiates the cluster kernels.

    See `topk_fi.topk_topp_sampling_from_prob` for the parameters,
    arguments and output contract; both sides share them.
    """
    comptime if emit_dist and from_logits and _has_sm_9x_or_newer():
        topk_topp_sampling_from_prob_cluster[
            dtype,
            out_idx_type,
            block_size=block_size,
            from_logits=from_logits,
            emit_dist=emit_dist,
            dist_dtype=dist_dtype,
        ](
            ctx,
            probs,
            output,
            top_k_val,
            top_p_val,
            deterministic,
            rng_seed,
            rng_offset,
            indices,
            top_k_arr,
            top_p_arr,
            temperature,
            min_p,
            out_dist,
        )
    else:
        _topk_topp_sampling_from_prob_single[
            dtype,
            out_idx_type,
            block_size=block_size,
            from_logits=from_logits,
            emit_dist=emit_dist,
            dist_dtype=dist_dtype,
        ](
            ctx,
            probs,
            output,
            top_k_val,
            top_p_val,
            deterministic,
            rng_seed,
            rng_offset,
            indices,
            top_k_arr,
            top_p_arr,
            temperature,
            min_p,
            out_dist,
        )
