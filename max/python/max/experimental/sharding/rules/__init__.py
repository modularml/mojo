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

"""Per-op placement rules.

Each op has a ``{op}_rule(...)`` function that lists the valid sharding choices
for that op's inputs and output as a set of rows. Each row says: if the inputs
have these placements, the output has this placement. The dispatcher reads one
op's rules at a time, hands them to the active solver, and gets back one chosen
row per mesh axis.

A rule takes one :class:`~max.experimental.sharding.TensorLayout` per tensor
input (plus the op's non-tensor args), builds a list of ``AxisAssignment`` rows
in plain Python, and returns ``build_action_set(...)``. There is no decorator,
DSL, or registration step. Override an op's rule by reassigning
``op.rule = my_rule``.

This module is internal: callers interact with sharding through placements and
solvers, not by calling rule functions directly.
"""

from .buffer import (
    buffer_store_rule,
    buffer_store_slice_rule,
)
from .control_flow import cond_rule, while_loop_rule
from .conv import (
    conv2d_rule,
    conv2d_transpose_rule,
    conv3d_rule,
)
from .elementwise import (
    binary_rule,
    linear_binary_rule,
    linear_unary_rule,
    ternary_rule,
    unary_rule,
)
from .matmul import matmul_rule, outer_rule
from .misc import (
    as_interleaved_complex_rule,
    band_part_rule,
    dequantize_rule,
    fold_rule,
    irfft_rule,
    masked_scatter_rule,
    qmatmul_rule,
    reject_distributed_rule,
    resize_bicubic_rule,
    resize_linear_rule,
    resize_nearest_rule,
    resize_rule,
    scatter_nd_add_rule,
    scatter_nd_rule,
)
from .norm import layer_norm_rule, rms_norm_rule
from .pooling import linear_pool_rule, pool_rule
from .reduction import (
    linear_reduce_rule,
    mean_rule,
    reduce_rule,
    softmax_rule,
)
from .shape import (
    argsort_rule,
    broadcast_to_rule,
    chunk_rule,
    flatten_rule,
    gather_nd_rule,
    gather_rule,
    nonzero_rule,
    pad_rule,
    passthrough_rule,
    permute_rule,
    rebind_rule,
    repeat_interleave_rule,
    reshape_rule,
    scatter_add_rule,
    scatter_rule,
    slice_tensor_rule,
    split_rule,
    squeeze_rule,
    stack_rule,
    tile_rule,
    top_k_rule,
    transpose_rule,
    unsqueeze_rule,
)
from .shape import concat_rule as same_placement_multi_input_rule

__all__ = [
    "argsort_rule",
    "as_interleaved_complex_rule",
    "band_part_rule",
    "binary_rule",
    "broadcast_to_rule",
    "buffer_store_rule",
    "buffer_store_slice_rule",
    "chunk_rule",
    "cond_rule",
    "conv2d_rule",
    "conv2d_transpose_rule",
    "conv3d_rule",
    "dequantize_rule",
    "flatten_rule",
    "fold_rule",
    "gather_nd_rule",
    "gather_rule",
    "irfft_rule",
    "layer_norm_rule",
    "linear_binary_rule",
    "linear_pool_rule",
    "linear_reduce_rule",
    "linear_unary_rule",
    "masked_scatter_rule",
    "matmul_rule",
    "mean_rule",
    "nonzero_rule",
    "outer_rule",
    "pad_rule",
    "passthrough_rule",
    "permute_rule",
    "pool_rule",
    "qmatmul_rule",
    "reduce_rule",
    "reject_distributed_rule",
    "repeat_interleave_rule",
    "reshape_rule",
    "resize_bicubic_rule",
    "resize_linear_rule",
    "resize_nearest_rule",
    "resize_rule",
    "rms_norm_rule",
    "same_placement_multi_input_rule",
    "scatter_add_rule",
    "scatter_nd_add_rule",
    "scatter_nd_rule",
    "scatter_rule",
    "slice_tensor_rule",
    "softmax_rule",
    "split_rule",
    "squeeze_rule",
    "stack_rule",
    "ternary_rule",
    "tile_rule",
    "top_k_rule",
    "transpose_rule",
    "unary_rule",
    "unsqueeze_rule",
    "while_loop_rule",
]
