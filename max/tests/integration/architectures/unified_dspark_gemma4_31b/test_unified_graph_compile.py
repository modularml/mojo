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
"""Compile test for the unified speculators-DSpark Gemma4 graph.

Builds the full merge -> target -> reject -> materialize -> block ->
markov graph at REAL 31B dimensions (60-layer 5:1 sliding/full target with
16 sliding + 4 global KV heads, 5-layer 16x256 draft, 262k/32k vocabs) with
synthetic (zero) weights, and compiles it on GPU. Guards the
graph-signature contract (`input_types` vs `_unflatten_graph_inputs` over
the nested {target: {sliding, full}, draft} KV tree), the anchor-slot drop
(block_size 8 -> 7 drafts), and the three-output spec-decode ABI.

The target geometry is hardcoded from google/gemma-4-31B-it's config.json;
the draft geometry parses from the checked-in RedHat speculators config
through the production parser. Weight VALUES are irrelevant here (the graph
is compiled and loaded, never executed), so zeros keep the ~62 GiB target
allocation cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from max.driver import Accelerator, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.speculators_common import (
    DSparkSpeculatorsDraftConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.model_config import (
    UnifiedDSparkGemma4_31BConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.unified_dspark_gemma4_31b import (
    UnifiedDSparkGemma4_31B,
)
from max.pipelines.lib.config import SpeculativeConfig

MAX_DTYPE = DType.bfloat16

# google/gemma-4-31B-it text_config (the values the KV tree and the taps
# depend on; the rest rides in via _make_target_config below).
HIDDEN = 5376
INTERMEDIATE = 21504
VOCAB = 262144
NUM_LAYERS = 60
# 5 sliding then 1 full, repeated: 50 sliding / 10 full layers.
LAYER_TYPES = [
    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
    for i in range(NUM_LAYERS)
]
N_HEADS = 32
N_KV_HEADS = 16
HEAD_DIM = 256
N_GLOBAL_KV_HEADS = 4
GLOBAL_HEAD_DIM = 512

BLOCK_SIZE = 8
NUM_DRAFTS = BLOCK_SIZE - 1
# vLLM aux ids [1, 17, 29, 47, 58] shifted to MAX layer-output ids.
TARGET_LAYER_IDS = (0, 16, 28, 46, 57)
MASK_TOKEN_ID = 4

TESTDATA = Path(__file__).parent / "testdata"


def _make_target_config(
    devices: list[DeviceRef],
) -> Gemma4ForConditionalGenerationConfig:
    sliding_layers = sum(1 for t in LAYER_TYPES if t == "sliding_attention")
    global_layers = sum(1 for t in LAYER_TYPES if t == "full_attention")
    sliding_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=sliding_layers,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=N_GLOBAL_KV_HEADS,
        head_dim=GLOBAL_HEAD_DIM,
        num_layers=global_layers,
        devices=devices,
        page_size=128,
    )
    kv_params = MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )
    text_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        devices=devices,
        page_size=128,
    )
    text_config = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_activation="gelu_tanh",
        max_position_embeddings=262144,
        max_seq_len=4096,
        rms_norm_eps=1e-6,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=1024,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=None,
        rope_local_base_freq=10000.0,
        sliding_window_pattern=-1,
        dtype=MAX_DTYPE,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=N_GLOBAL_KV_HEADS,
        global_head_dim=GLOBAL_HEAD_DIM,
        attention_k_eq_v=True,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=0.25
        ),
        global_rope_theta=1_000_000.0,
        sliding_window_rope_theta=10000.0,
        layer_types=LAYER_TYPES,
    )
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=MAX_DTYPE,
        kv_params=kv_params,
        text_config=text_config,
        vision_config=None,
        image_token_index=258880,
        tie_word_embeddings=True,
    )


def _make_draft_config() -> DSparkSpeculatorsDraftConfig:
    with open(TESTDATA / "redhat_speculator_config.json") as f:
        config_json = json.load(f)
    return DSparkSpeculatorsDraftConfig.from_huggingface_config(
        config_json, max_seq_len=8192
    )


def _make_unified_config(
    devices: list[DeviceRef], *, num_speculative_tokens: int
) -> UnifiedDSparkGemma4_31BConfig:
    target_config = _make_target_config(devices)
    draft_config = _make_draft_config()
    assert draft_config.block_size == BLOCK_SIZE
    assert draft_config.target_layer_ids == TARGET_LAYER_IDS
    draft_kv_params = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=devices,
        page_size=128,
    )
    config = UnifiedDSparkGemma4_31BConfig(
        target=target_config,
        draft=draft_config,
        draft_kv_params=draft_kv_params,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=num_speculative_tokens,
        ),
        target_layer_ids=list(TARGET_LAYER_IDS),
        mask_token_id=MASK_TOKEN_ID,
        block_size=BLOCK_SIZE,
    )
    config.validate_dspark_fields()
    return config


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


def test_unified_dspark_31b_graph_compiles(
    session: InferenceSession, device: Device
) -> None:
    config = _make_unified_config(
        [DeviceRef.GPU()], num_speculative_tokens=NUM_DRAFTS
    )

    nn_model = UnifiedDSparkGemma4_31B(config)
    # sample_from_anchor=false: 8-slot block, 7 drafts.
    assert nn_model.block_size == BLOCK_SIZE
    assert nn_model.num_speculative_tokens == NUM_DRAFTS

    # The speculators checkpoint's keys must land at these names via the
    # verbatim 'draft.' prefix mapping (no target aliasing in this arch).
    raw_names = set(nn_model.raw_state_dict().keys())
    assert "draft.markov_head.markov_w1.weight" in raw_names
    assert "draft.markov_head.markov_w2.weight" in raw_names
    assert "draft.d2t" in raw_names
    assert "draft.embed_tokens.weight" in raw_names
    assert "draft.lm_head.weight" in raw_names

    torch_dtypes = {
        MAX_DTYPE: torch.bfloat16,
        DType.float32: torch.float32,
        DType.int64: torch.int64,
    }
    state_dict = {}
    for key, weight in nn_model.raw_state_dict().items():
        shape = tuple(int(d) for d in weight.shape)
        state_dict[key] = torch.zeros(shape, dtype=torch_dtypes[weight.dtype])
    nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)

    with Graph(
        "unified_dspark_gemma4_31b_test",
        input_types=nn_model.input_types(),
    ) as graph:
        values = nn_model._unflatten_graph_inputs(graph.inputs)
        outputs = nn_model(values)
        assert len(outputs) == 3
        next_draft_tokens = outputs[2]
        # One draft per mask slot: the anchor slot is dropped.
        assert int(next_draft_tokens.shape[1]) == NUM_DRAFTS
        assert next_draft_tokens.dtype == DType.int64
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=nn_model.state_dict())
    assert compiled is not None


@pytest.mark.parametrize("num_speculative_tokens", [4, 10])
def test_unified_dspark_31b_graph_stages_at_configured_k(
    num_speculative_tokens: int,
) -> None:
    """An honored ``num_speculative_tokens=K`` sizes the staged draft
    output to ``[batch, K]``, below AND above the trained width.

    Staging-only (no compile, no weights): guards the effective-K plumbing
    from the speculative config through the block forward and the markov
    unroll. The draft block is causal and width-generic: K=4 truncates the
    trained 8-slot block prefix-stably, K=10 runs the extra slots as
    extrapolation.
    """
    config = _make_unified_config(
        [DeviceRef.GPU()], num_speculative_tokens=num_speculative_tokens
    )
    assert (
        config.speculative_config.num_speculative_tokens
        == num_speculative_tokens
    )
    assert config.effective_block_size == num_speculative_tokens + 1

    nn_model = UnifiedDSparkGemma4_31B(config)
    assert nn_model.block_size == num_speculative_tokens + 1
    assert nn_model.num_speculative_tokens == num_speculative_tokens

    # Staging without loading values: assign the hierarchical weight names
    # (``load_state_dict`` does this in the compile test above) so sibling
    # ``weight`` leaves don't collide when the graph registers them.
    for name, weight in nn_model.raw_state_dict().items():
        weight.name = name

    with Graph(
        f"unified_dspark_gemma4_31b_k{num_speculative_tokens}_test",
        input_types=nn_model.input_types(),
    ) as graph:
        values = nn_model._unflatten_graph_inputs(graph.inputs)
        outputs = nn_model(values)
        assert len(outputs) == 3
        next_draft_tokens = outputs[2]
        assert int(next_draft_tokens.shape[1]) == num_speculative_tokens
        assert next_draft_tokens.dtype == DType.int64
        graph.output(*outputs)


def test_unified_dspark_31b_graph_stages_with_structured_output() -> None:
    """The structured-output signature unflattens the bitmask triple and the
    masked acceptance path stages end to end.

    Staging-only (no compile, no weights): guards the in-graph
    ``apply_overlap_bitmask`` insertion (host-wait, H2D scratch copy, and the
    ``num_steps + 1`` row trim) feeding ``token_bitmasks`` into the
    acceptance sampler.
    """
    config = _make_unified_config(
        [DeviceRef.GPU()], num_speculative_tokens=NUM_DRAFTS
    )
    nn_model = UnifiedDSparkGemma4_31B(config, enable_structured_output=True)

    for name, weight in nn_model.raw_state_dict().items():
        weight.name = name

    with Graph(
        "unified_dspark_gemma4_31b_so_test",
        input_types=nn_model.input_types(),
    ) as graph:
        values = nn_model._unflatten_graph_inputs(graph.inputs)
        assert values.pinned_bitmask is not None
        assert values.wait_payload is not None
        assert values.device_bitmask_scratch is not None
        outputs = nn_model(values)
        assert len(outputs) == 3
        graph.output(*outputs)
