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
"""Neural network modules for MAX.

Graph-based API:

.. code-block:: python

    from max.nn import Linear, AttentionWithRope
    from max.nn.kv_cache import KVCacheParams

Eager tensor API:

.. code-block:: python

    from max.experimental.nn import Module, Linear, Embedding
"""

from .attention import (
    AttentionWithRope,
    DistributedAttentionImpl,
    GGUFQAttentionWithRope,
    GPTQAttentionWithRope,
    LatentAttentionWithRope,
    MultiheadAttention,
    RaggedAttention,
    TensorParallelAttentionWithRope,
    TensorParallelLatentAttentionWithRope,
)
from .clamp import clamp
from .comm import Allreduce, Signals
from .conv import Conv1D, Conv2d, Conv3D
from .conv_transpose import ConvTranspose1d, WeightNormConvTranspose1d
from .data_parallelism import split_batch, split_batch_replicated
from .embedding import Embedding, VocabParallelEmbedding
from .identity import Identity
from .kv_cache import (
    KVCacheInputs,
    KVCacheMetrics,
    KVCacheParams,
    build_max_lengths_tensors,
)
from .layer import Layer, LayerList, Module, Shardable
from .linear import MLP, ColumnParallelLinear, FusedMLP, GPTQLinear, Linear
from .moe import MoE, MoEGate, MoEQuantized, forward_moe_sharded_layers
from .norm import ConstantLayerNorm, GroupNorm, LayerNorm, RMSNorm
from .quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from .rotary_embedding import (
    DynamicRotaryEmbedding,
    LinearScalingParams,
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
    LongRoPEScalingParams,
    RotaryEmbedding,
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from .sampling import (
    MinPSampler,
    RejectionSampler,
    RejectionSamplerWithResiduals,
)
from .sequential import Sequential
from .stacked_linear import StackedLinear
from .transformer import (
    DistributedTransformer,
    DistributedTransformerBlock,
    ReturnHiddenStates,
    ReturnLogits,
    Transformer,
    TransformerBlock,
)

__all__ = [
    "MLP",
    "Allreduce",
    "AttentionWithRope",
    "ColumnParallelLinear",
    "ConstantLayerNorm",
    "Conv1D",
    "Conv2d",
    "Conv3D",
    "ConvTranspose1d",
    "DistributedAttentionImpl",
    "DistributedTransformer",
    "DistributedTransformerBlock",
    "DynamicRotaryEmbedding",
    "Embedding",
    "FusedMLP",
    "GGUFQAttentionWithRope",
    "GPTQAttentionWithRope",
    "GPTQLinear",
    "GroupNorm",
    "Identity",
    "InputScaleSpec",
    "KVCacheInputs",
    "KVCacheMetrics",
    "KVCacheParams",
    "LatentAttentionWithRope",
    "Layer",
    "LayerList",
    "LayerNorm",
    "Linear",
    "LinearScalingParams",
    "Llama3RopeScalingParams",
    "Llama3RotaryEmbedding",
    "LongRoPERotaryEmbedding",
    "LongRoPEScalingParams",
    "MinPSampler",
    "MoE",
    "MoEGate",
    "MoEQuantized",
    "Module",
    "MultiheadAttention",
    "QuantConfig",
    "QuantFormat",
    "RMSNorm",
    "RaggedAttention",
    "RejectionSampler",
    "RejectionSamplerWithResiduals",
    "ReturnHiddenStates",
    "ReturnLogits",
    "RotaryEmbedding",
    "ScaleGranularity",
    "ScaleOrigin",
    "Sequential",
    "Shardable",
    "Signals",
    "StackedLinear",
    "TensorParallelAttentionWithRope",
    "TensorParallelLatentAttentionWithRope",
    "Transformer",
    "TransformerBlock",
    "VocabParallelEmbedding",
    "WeightNormConvTranspose1d",
    "WeightScaleSpec",
    "YarnRotaryEmbedding",
    "YarnScalingParams",
    "build_max_lengths_tensors",
    "clamp",
    "forward_moe_sharded_layers",
    "split_batch",
    "split_batch_replicated",
]
