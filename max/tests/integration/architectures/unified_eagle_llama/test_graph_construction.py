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
from max.dtype import DType
from max.graph import DeviceRef, Graph
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.unified_eagle_llama3.model_config import (
    UnifiedEagleLlama3Config,
)
from max.pipelines.architectures.unified_eagle_llama3.unified_eagle_llama3 import (
    UnifiedEagleLlama3,
)
from max.pipelines.lib.config import SpeculativeConfig


def create_dummy_llama3_config(layers: int) -> Llama3Config:
    return Llama3Config(
        hidden_size=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=layers,
        rope_theta=1234.0,
        rope_scaling_params=None,
        max_seq_len=2048,
        intermediate_size=256,
        interleaved_rope_weights=True,
        vocab_size=128256,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=4,
            head_dim=2,
            num_layers=layers,
            devices=[DeviceRef.GPU(0)],
            data_parallel_degree=1,
            speculative_method="eagle",
            num_draft_tokens=1,
        ),
        attention_multiplier=1.0,
        embedding_multiplier=2.0,
        residual_multiplier=3.0,
        rms_norm_eps=4.0,
        clip_qkv=6.0,
        norm_method="rms_norm",
        devices=[DeviceRef.GPU(0)],
    )


def create_dummy_eagle_llama3_config(
    enable_structured_output: bool = False,
) -> UnifiedEagleLlama3Config:
    return UnifiedEagleLlama3Config(
        target=create_dummy_llama3_config(layers=8),
        draft=create_dummy_llama3_config(layers=1),
        speculative_config=SpeculativeConfig(num_speculative_tokens=1),
        enable_structured_output=enable_structured_output,
    )


def test_graph_construction() -> None:
    config = create_dummy_eagle_llama3_config()
    model = UnifiedEagleLlama3(config)

    state_dict = model.state_dict()

    # State dict must not be empty.
    assert state_dict, "State dict must not be empty"
    # Weights must be namespaced under "target." or "draft." prefixes.
    assert all(
        weight.startswith(("target.", "draft.")) for weight in state_dict
    )
    assert "draft.layers.0.mlp.up_proj.weight" in state_dict
    assert "target.layers.7.mlp.up_proj.weight" in state_dict

    # Shared weights (embed_tokens, lm_head) should appear under target.
    assert "target.embed_tokens.weight" in state_dict
    assert "target.lm_head.weight" in state_dict

    # Verify input types include draft_tokens and draft_cache_lengths.
    input_types = model.input_types()
    # Expected: tokens, input_row_offsets, return_n_logits,
    #           + target KV (7 fields) + draft KV (7 fields) + draft_tokens,
    #           + rng seed, + sampling params (temperature, top_k, max_k, top_p, min_top_p)
    assert len(input_types) == 24, (
        f"Expected 24 input types, got {len(input_types)}"
    )

    # Smoke test that graph construction (not compilation) works
    with Graph(
        "unified_eagle_llama3", input_types=model.input_types()
    ) as graph:
        inputs = model._unflatten_graph_inputs(graph.inputs)
        outputs = model(inputs)
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        graph.output(*outputs)


def test_input_types_with_structured_output() -> None:
    """Test that input types include the bitmask triple when structured
    output is enabled.

    The overlap path binds three inputs in order: pinned bitmask source,
    the int64[2] wait payload consumed by ``mo.wait_host_value_with_dep``,
    and the device-side bitmask scratch destination.
    """
    config = create_dummy_eagle_llama3_config(enable_structured_output=True)
    model = UnifiedEagleLlama3(config)

    # Verify input types include the bitmask triple when structured
    # output is enabled.
    input_types = model.input_types()
    # Expected: 24 mandatory inputs + 3 bitmask (pinned, wait_payload,
    # device_bitmask_scratch) = 27 total
    assert len(input_types) == 27, (
        f"Expected 27 input types (with bitmask triple), got {len(input_types)}"
    )

    # The trailing three inputs are pinned_bitmask (packed int32 tensor),
    # wait_payload (int64 buffer), and device_bitmask_scratch (packed int32
    # buffer). The bitmask is stored as packed int32 (one bit per vocab token)
    # so the GPU acceptance sampler unpacks and applies it in one fused pass.
    pinned_type = input_types[-3]
    payload_type = input_types[-2]
    scratch_type = input_types[-1]
    assert pinned_type.dtype.to_numpy() == "int32", (
        f"Expected pinned bitmask dtype int32, got {pinned_type.dtype}"
    )
    assert payload_type.dtype.to_numpy() == "int64", (
        f"Expected wait_payload dtype int64, got {payload_type.dtype}"
    )
    assert scratch_type.dtype.to_numpy() == "int32", (
        f"Expected device_bitmask_scratch dtype int32, got {scratch_type.dtype}"
    )
