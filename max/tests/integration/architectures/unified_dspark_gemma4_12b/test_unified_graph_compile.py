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
"""Compile test for the unified DSpark Gemma4 graph.

Builds the full merge -> target -> reject -> materialize -> block ->
markov graph with tiny synthetic configs and compiles it on GPU. Guards
the graph-signature contract (`input_types` vs `_unflatten_graph_inputs`
over the nested {target: {sliding, full}, draft} KV tree), the shared
embed/lm_head aliasing, and the three-output spec-decode ABI.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.layer import Module
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_12b import (
    DSparkGemma4DraftConfig,
    UnifiedDSparkGemma4_12B,
    UnifiedDSparkGemma4_12BConfig,
)
from max.pipelines.lib.config import SpeculativeConfig

MAX_DTYPE = DType.bfloat16

HIDDEN = 64
INTERMEDIATE = 128
VOCAB = 256
NUM_LAYERS = 6
LAYER_TYPES = [
    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
    for i in range(NUM_LAYERS)
]
HEAD_DIM = 32
GLOBAL_HEAD_DIM = 64
N_HEADS = 2
N_KV_HEADS = 2
N_GLOBAL_KV_HEADS = 1
BLOCK_SIZE = 7
TARGET_LAYER_IDS = (1, 3)
MASK_TOKEN_ID = 4


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
        max_position_embeddings=1024,
        max_seq_len=1024,
        rms_norm_eps=1e-6,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=HEAD_DIM,
        sliding_window=512,
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
        image_token_index=0,
        tie_word_embeddings=True,
    )


def _make_draft_config() -> DSparkGemma4DraftConfig:
    return DSparkGemma4DraftConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=len(TARGET_LAYER_IDS),
        num_attention_heads=N_HEADS,
        num_key_value_heads=1,
        head_dim=GLOBAL_HEAD_DIM,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        hidden_activation="gelu_tanh",
        final_logit_softcapping=30.0,
        rope_theta=1_000_000.0,
        partial_rotary_factor=0.25,
        max_seq_len=1024,
        block_size=BLOCK_SIZE,
        mask_token_id=MASK_TOKEN_ID,
        target_layer_ids=TARGET_LAYER_IDS,
        markov_rank=8,
        markov_head_type="vanilla",
    )


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


def test_unified_dspark_graph_compiles(
    session: InferenceSession, device: Device
) -> None:
    devices = [DeviceRef.GPU()]
    target_config = _make_target_config(devices)
    draft_config = _make_draft_config()
    draft_kv_params = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=devices,
        page_size=128,
    )
    config = UnifiedDSparkGemma4_12BConfig(
        target=target_config,
        draft=draft_config,
        draft_kv_params=draft_kv_params,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=BLOCK_SIZE,
        ),
        target_layer_ids=list(TARGET_LAYER_IDS),
        mask_token_id=MASK_TOKEN_ID,
        block_size=BLOCK_SIZE,
    )
    config.validate_dspark_fields()

    nn_model = UnifiedDSparkGemma4_12B(config)
    assert nn_model.num_speculative_tokens == BLOCK_SIZE

    # Alias the shared modules exactly as the pipeline model does.
    assert isinstance(nn_model.target.lm_head, Module)
    nn_model.draft.embed_tokens = nn_model.target.embed_tokens
    nn_model.draft.lm_head = nn_model.target.lm_head

    # The DSpark checkpoint's markov table must land at these names via the
    # verbatim 'draft.' prefix mapping.
    raw_names = set(nn_model.raw_state_dict().keys())
    assert "draft.markov_head.markov_w1.weight" in raw_names
    assert "draft.markov_head.markov_w2.weight" in raw_names

    torch.manual_seed(20260729)
    shared_prefixes = ("draft.embed_tokens.", "draft.lm_head.")
    torch_dtypes = {MAX_DTYPE: torch.bfloat16, DType.float32: torch.float32}
    state_dict = {}
    for key, weight in nn_model.raw_state_dict().items():
        if key.startswith(shared_prefixes):
            continue
        shape = tuple(int(d) for d in weight.shape)
        state_dict[key] = (
            torch.randn(shape, dtype=torch_dtypes[weight.dtype]) * 0.02
        )
    nn_model.load_state_dict(state_dict, weight_alignment=1, strict=False)

    with Graph(
        "unified_dspark_gemma4_12b_test",
        input_types=nn_model.input_types(),
    ) as graph:
        values = nn_model._unflatten_graph_inputs(graph.inputs)
        outputs = nn_model(values)
        assert len(outputs) == 3
        next_draft_tokens = outputs[2]
        # next_draft_tokens carries one draft per block position (all 7).
        assert int(next_draft_tokens.shape[1]) == BLOCK_SIZE
        assert next_draft_tokens.dtype == DType.int64
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=nn_model.state_dict())
    assert compiled is not None
