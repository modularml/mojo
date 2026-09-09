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
"""Nemotron-H architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

# Nemotron-3's chat template is Qwen-format: it pre-fills ``<think>\n`` in the
# generation prompt (implicit reasoning open, explicit ``</think>`` close),
# backfills ``<think></think>`` into prior assistant turns, and renders tool
# calls as ``<tool_call>/<function=...>/<parameter=...>`` blocks. Built-in
# architectures load lazily, so import the qwen3_5 parser modules here to
# register the parsers named below.
from ..qwen3_5.reasoning import (
    Qwen3_5ReasoningParser,  # noqa: F401  registers "qwen3_5"
)
from ..qwen3_5.tool_parser import (
    Qwen3_5ToolParser,  # noqa: F401  registers "qwen3_5"
)
from .model import NemotronHModel
from .model_config import NemotronHConfig
from .tokenizer import NemotronHTokenizer
from .weight_adapters import convert_nemotron_h_state_dict

nemotron_h_arch = SupportedArchitecture(
    name="NemotronHForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=NemotronHConfig.DEFAULT_ENCODING,
    # modelopt per-tensor static FP8 on the (non-excluded) mamba in/out_proj and
    # MLP up/down projections; attention, conv1d, norms, lm_head stay bf16.
    supported_encodings=NemotronHConfig.SUPPORTED_ENCODINGS,
    pipeline_model=NemotronHModel,
    tokenizer=NemotronHTokenizer,
    context_type=TextContext,
    # NoPE: attention adds no rotary embedding (position flows through the SSM).
    weight_adapters={
        WeightsFormat.safetensors: convert_nemotron_h_state_dict,
    },
    # SSM recurrent state is not reconstructable from a token prefix, so prefix
    # caching must be disabled.
    required_arguments={"enable_prefix_caching": False},
    config=NemotronHConfig,
    multi_gpu_supported=False,
    # Reasoning opens implicitly (template pre-fills ``<think>``) and closes
    # at ``</think>``; without a default parser the CoT and the raw
    # ``</think>`` delimiter leak into OpenAI ``message.content``. The
    # qwen3_5 parsers match Nemotron-3's Qwen-format template exactly.
    reasoning_parser="qwen3_5",
    tool_parser="qwen3_5",
    # SSM conv and state pools are pre-allocated, fixed-address, full-pool
    # buffers.  The in-place slot-indexed kernels (causal_conv1d_varlen_fwd,
    # mamba2_ssd_chunk_scan_varlen_fwd_inplace) mutate them directly on the
    # GPU via slot_idx — no host-device sync required.  ``inplace_copy_from``
    # short-circuits when src-is-self (same Buffer object) so pool contents
    # survive graph-capture replay unchanged.  NemotronHModel implements
    # SupportsSSMStateWarmup so the overlap pipeline releases warmup slots
    # after each (batch_size, cache_length) probe, preventing pool exhaustion.
    supports_device_graph_capture=True,
)
