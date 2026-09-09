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

from __future__ import annotations

from unittest.mock import MagicMock, NonCallableMock

from max.driver import DeviceSpec
from max.pipelines.architectures.deepseekV3.memory_planner import (
    DeepseekV3MemoryPlanner,
)
from max.pipelines.kv_cache.memory_planner import ModelConfigWithKVCache
from max.pipelines.lib import (
    PipelineConfig,
    PipelineRole,
    SupportedEncoding,
)

MAX_SEND_TOKENS_PER_RANK = 128


def _make_planner() -> DeepseekV3MemoryPlanner:
    """Create a DeepseekV3MemoryPlanner with a minimal mock KV config."""
    config = MagicMock(spec=ModelConfigWithKVCache)
    config.get_kv_params.return_value = MagicMock()
    return DeepseekV3MemoryPlanner(config)


NUM_RANKS = 8
GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE = 8 * 1024**3


def mock_pipeline_config(
    pipeline_role: PipelineRole,
    quantization_encoding: SupportedEncoding = "float8_e4m3fn",
) -> NonCallableMock:
    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = MagicMock()
    pipeline_config.runtime = MagicMock()
    pipeline_config.model.quantization_encoding = quantization_encoding
    pipeline_config.model.kv_cache.kv_cache_format = None
    pipeline_config.model.data_parallel_degree = NUM_RANKS
    pipeline_config.model.device_specs = [
        NonCallableMock(spec=DeviceSpec) for _ in range(NUM_RANKS)
    ]

    # Pipeline config attributes
    pipeline_config.runtime.pipeline_role = pipeline_role
    pipeline_config.model.max_length = 1024 * 1024  # ~million tokens
    pipeline_config.runtime.max_batch_total_tokens = None
    pipeline_config.runtime.ep_size = NUM_RANKS
    pipeline_config.runtime.max_batch_input_tokens = MAX_SEND_TOKENS_PER_RANK
    pipeline_config.runtime.device_graph_capture = False
    pipeline_config.runtime.ep_use_allreduce = False
    pipeline_config.speculative = None

    return pipeline_config


def mock_huggingface_config() -> MagicMock:
    huggingface_config = MagicMock()

    # HuggingFace config attributes
    huggingface_config.num_attention_heads = 128
    huggingface_config.qk_nope_head_dim = 128
    huggingface_config.n_routed_experts = 256
    huggingface_config.moe_intermediate_size = 2048
    huggingface_config.hidden_size = 7168

    # Additional attributes for estimate_weights_size
    huggingface_config.num_hidden_layers = 61
    huggingface_config.first_k_dense_replace = 1
    huggingface_config.num_nextn_predict_layers = 1
    huggingface_config.vocab_size = 129280
    huggingface_config.n_shared_experts = 1
    huggingface_config.num_experts_per_tok = 8

    return huggingface_config


def test_deepseekv3_memory_estimation() -> None:
    planner = _make_planner()
    pipeline_config = mock_pipeline_config("decode_only")
    huggingface_config = mock_huggingface_config()
    assert huggingface_config is not None

    memory_estimated = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )

    max_recv_tokens_per_rank = (
        MAX_SEND_TOKENS_PER_RANK * huggingface_config.n_routed_experts
    )
    moe_min_memory = (
        max_recv_tokens_per_rank * huggingface_config.moe_intermediate_size * 1
    )  # Float8
    moe_min_memory += (
        max_recv_tokens_per_rank * huggingface_config.hidden_size * 2
    )  # BFloat16
    moe_min_memory *= NUM_RANKS

    assert memory_estimated > moe_min_memory


def test_deepseekv3_memory_estimation_exact() -> None:
    planner = _make_planner()
    huggingface_config = mock_huggingface_config()
    assert huggingface_config is not None

    # For DecodeOnly, we only need to consider moe_activation_memory
    pipeline_config = mock_pipeline_config("decode_only")
    mem = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 5225054208

    # For PrefillAndDecode, we also need to consider mla_activation_memory
    pipeline_config = mock_pipeline_config("prefill_and_decode")
    mem = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 551759642624

    # Also check model with different quantization encoding
    pipeline_config = mock_pipeline_config("decode_only", "float4_e2m1fnx2")
    mem = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 4399759360


def test_deepseekv3_memory_estimation_adds_graph_capture_headroom() -> None:
    planner = _make_planner()
    huggingface_config = mock_huggingface_config()
    assert huggingface_config is not None

    pipeline_config = mock_pipeline_config("decode_only")
    baseline = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )

    pipeline_config.runtime.device_graph_capture = True
    with_headroom = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )

    expected_headroom = GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE * NUM_RANKS
    assert with_headroom == baseline + expected_headroom


def mock_weights_pipeline_config(
    n_gpus: int, ep_size: int, dp_degree: int
) -> NonCallableMock:
    """Create a mock pipeline config for estimate_weights_size tests."""
    huggingface_config = mock_huggingface_config()
    assert huggingface_config is not None

    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = MagicMock()
    pipeline_config.runtime = MagicMock()
    pipeline_config.model.quantization_encoding = "float8_e4m3fn"
    pipeline_config.model.data_parallel_degree = dp_degree
    pipeline_config.model.device_specs = [
        NonCallableMock(spec=DeviceSpec) for _ in range(n_gpus)
    ]
    pipeline_config.model.huggingface_config = huggingface_config
    # Use a large enough weights size to account for the algorithm's subtractions.
    # DeepSeek-V3 has ~671B parameters, ~700GB at FP8.
    pipeline_config.model.weights_size.return_value = 700 * 1024**3
    pipeline_config.runtime.ep_size = ep_size

    return pipeline_config


def compute_routing_experts_size() -> int:
    """Compute the routing_experts_size from mock_huggingface_config values.

    This matches the calculation in estimate_weights_size:
    routing_experts_size = n_sparse_layers * n_routed_experts * expert_size
    where expert_size = moe_intermediate_size * hidden_size * 3 * dtype
    """
    hf_config = mock_huggingface_config()
    dtype = 1  # float8_e4m3fn size in bytes
    n_sparse_layers = (
        hf_config.num_hidden_layers - hf_config.first_k_dense_replace
    )
    expert_size = (
        hf_config.moe_intermediate_size * hf_config.hidden_size * 3 * dtype
    )
    return n_sparse_layers * hf_config.n_routed_experts * expert_size


def test_deepseekv3_estimate_weights_size_no_expert_parallelism() -> None:
    """Test estimate_weights_size with ep_size=1 and multiple devices.

    This is a regression test for a bug where ep_size=1 with multiple GPUs
    would cause a ZeroDivisionError (n_nodes = 1 // 8 = 0).
    """
    planner = _make_planner()
    # EP=1 (no expert parallelism), 8 GPUs, DP=1
    pipeline_config = mock_weights_pipeline_config(
        n_gpus=8, ep_size=1, dp_degree=1
    )

    # This should not raise ZeroDivisionError
    mem = planner.estimate_weights_size(pipeline_config)
    assert mem > 0


def test_deepseekv3_estimate_weights_size_dp_ep_exact() -> None:
    planner = _make_planner()
    # EP=8, 8 GPUs, DP=8
    pipeline_config = mock_weights_pipeline_config(
        n_gpus=8, ep_size=8, dp_degree=8
    )

    # The result is quite large because the mock weights size is larger
    # than the actual weights size.
    mem = planner.estimate_weights_size(pipeline_config)
    assert mem == 1124551261664


def test_deepseekv3_estimate_weights_size_tp_ep_exact() -> None:
    planner = _make_planner()
    # EP=8, 8 GPUs, TP attention (DP=1)
    pipeline_config = mock_weights_pipeline_config(
        n_gpus=8, ep_size=8, dp_degree=1
    )

    mem = planner.estimate_weights_size(pipeline_config)
    assert mem == 754209345468


def test_deepseekv3_estimate_weights_size_routing_experts_scaling() -> None:
    """Verify routing experts memory scales correctly with EP configurations.

    Currently, ep_size must be either 1 (no EP) or a multiple of n_gpus_per_node
    (EP across full nodes). Mixed EP/TP strategies are not yet supported.

    For supported configurations:
    - EP=1: routing_experts_memory = routing_experts_size (full copy)
    - EP=n_gpus*n_nodes: routing_experts_memory = routing_experts_size / n_nodes
    """
    planner = _make_planner()
    routing_experts_size = compute_routing_experts_size()
    n_gpus = 8

    # Convert to int since the memory estimation involves some float arithmetic.
    # EP=1: no expert parallelism
    mem_ep1 = int(
        planner.estimate_weights_size(
            mock_weights_pipeline_config(n_gpus=n_gpus, ep_size=1, dp_degree=1)
        )
    )
    # EP=8: single node with full EP (n_nodes=1)
    mem_ep8 = int(
        planner.estimate_weights_size(
            mock_weights_pipeline_config(n_gpus=n_gpus, ep_size=8, dp_degree=1)
        )
    )
    # EP=16: two nodes (n_nodes=2)
    mem_ep16 = int(
        planner.estimate_weights_size(
            mock_weights_pipeline_config(n_gpus=n_gpus, ep_size=16, dp_degree=1)
        )
    )

    # Verify the routing experts contribution:
    # EP=1: full routing_experts_size (no split)
    # EP=8 (n_nodes=1): routing_experts_size / 1 = routing_experts_size
    # EP=16 (n_nodes=2): routing_experts_size / 2

    # EP=1 vs EP=8: EP=1 has full routing_experts_size, EP=8 has routing_experts_size
    assert mem_ep1 == mem_ep8  # Both have full routing_experts_size (n_nodes=1)

    # EP=8 vs EP=16: EP=16 splits across 2 nodes, so routing_experts_size / 2
    assert mem_ep8 - mem_ep16 == routing_experts_size // 2
