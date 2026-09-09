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
"""The fused graph's weight names, and the load that must accept them.

The draft head shares the target's embedding module by reference. That makes
the shared weight reachable under two attribute paths but present in the
namespace once, because the name walk dedupes by module identity, so the
checkpoint half of the load must not carry a second name for it. Supplying one
makes ``_check_weights_match`` refuse the load outright, since it rejects any
``draft.*`` key the graph does not consume.
"""

from __future__ import annotations

from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.unified_mtp_qwen3_5.model import (
    _check_weights_match,
    _merge_state_dicts,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.unified_mtp_qwen3_5 import (
    UnifiedMTPQwen3_5,
)

HIDDEN = 32
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 16
VOCAB = 64


def _config() -> Qwen3_5Config:
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=HEAD_DIM,
    )
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=2,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
        state_pool_dtype=None,
    )


def test_the_shared_embedding_has_exactly_one_name() -> None:
    nn_model = UnifiedMTPQwen3_5(_config())
    assert nn_model.draft.embed_tokens is nn_model.target.embed_tokens

    names = set(nn_model.raw_state_dict().keys())
    assert "target.embed_tokens.weight" in names
    assert "draft.embed_tokens.weight" not in names, (
        "the walk dedupes by module identity, so a second name here would mean"
        " the draft head stopped sharing the target's embedding module"
    )


def test_the_checkpoint_namespace_loads() -> None:
    """The merged namespace is exactly the set of names the graph declares.

    Builds the checkpoint halves the way the adapter splits them, pins the
    merge against the graph's own names (every one, and nothing else), then
    runs the load that ``_check_weights_match`` guards.
    """
    nn_model = UnifiedMTPQwen3_5(_config())
    raw = nn_model.raw_state_dict()

    def _halve(prefix: str) -> dict[str, Buffer]:
        return {
            name[len(prefix) :]: Buffer.zeros(
                shape=w.shape.static_dims, dtype=w.dtype
            )
            for name, w in raw.items()
            if name.startswith(prefix)
        }

    full_state_dict = _merge_state_dicts(_halve("target."), _halve("draft."))
    assert set(full_state_dict) == set(raw), (
        "the merged namespace must be exactly the graph's declared names"
    )

    _check_weights_match(
        expected=set(raw.keys()), provided=set(full_state_dict.keys())
    )
    nn_model.load_state_dict(
        full_state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=False,
    )
    # The draft reads the constant the target loaded, under the target's name.
    assert (
        nn_model.draft.embed_tokens.weight.name == "target.embed_tokens.weight"
    )
    assert "target.embed_tokens.weight" in nn_model.state_dict()


def test_an_aliased_embedding_is_refused() -> None:
    """A ``draft.*`` key the graph does not consume fails the load.

    The shared embedding is declared under the target's name only, so a
    ``draft.`` name for it is one of those keys: unsupplyable, rather than a
    harmless duplicate.
    """
    nn_model = UnifiedMTPQwen3_5(_config())
    expected = set(nn_model.raw_state_dict().keys())
    provided = expected | {"draft.embed_tokens.weight"}

    try:
        _check_weights_match(expected=expected, provided=provided)
    except ValueError as e:
        assert "draft.embed_tokens.weight" in str(e)
    else:
        raise AssertionError("an unconsumed draft.* key must fail the load")
