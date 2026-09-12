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
"""Structural checks on the DFlash2 drafter: weight contract and mask choice."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.attention import AttentionWithRope
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.dflash2_qwen3_5 import (
    DFlash2GroupedConv,
    DFlash2Qwen3_5,
    DFlash2TransformerBlock,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config

HIDDEN = 64
HEAD_DIM = 16
LAYERS = 2
GROUP_SIZE = 8
BLOCK_SIZE = 8
TAPS = 2
RANK = 4
TOP_K = 3
VOCAB = 128


def _config(sliding_window: int | None = 32) -> Llama3Config:
    device = DeviceRef.CPU()
    return Llama3Config(
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=LAYERS,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=128,
        interleaved_rope_weights=False,
        vocab_size=VOCAB,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=2,
            head_dim=HEAD_DIM,
            num_layers=LAYERS,
            page_size=128,
            devices=[device],
        ),
        rms_norm_eps=1e-6,
        attention_multiplier=HEAD_DIM**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        sliding_window=sliding_window,
    )


def _drafter(
    layer_types: list[str] | None = None, sliding_window: int | None = 32
) -> DFlash2Qwen3_5:
    return DFlash2Qwen3_5(
        _config(sliding_window),
        num_context_features=LAYERS,
        block_size=BLOCK_SIZE,
        conv_kernel_size=TAPS,
        conv_group_size=GROUP_SIZE,
        selector_rank=RANK,
        selector_top_k=TOP_K,
        layer_types=layer_types,
    )


def test_weight_names_match_the_checkpoint_layout() -> None:
    """The drafter must consume exactly the names z-lab ships, and own no more.

    ``embed_tokens`` and ``lm_head`` are borrowed from the target, so they are
    absent here: a drafter that declared them would silently load a second,
    uninitialised copy.
    """
    names = set(_drafter().state_dict())

    expected = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    expected |= {
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "candidate_selector.hidden_projection.weight",
    }
    for layer in range(LAYERS):
        expected |= {
            f"layers.{layer}.input_layernorm.weight",
            f"layers.{layer}.post_attention_layernorm.weight",
            f"layers.{layer}.self_attn.q_norm.weight",
            f"layers.{layer}.self_attn.k_norm.weight",
            f"layers.{layer}.mlp.gate_proj.weight",
            f"layers.{layer}.mlp.up_proj.weight",
            f"layers.{layer}.mlp.down_proj.weight",
            f"layers.{layer}.attention_conv.base_kernel",
            f"layers.{layer}.attention_conv.kernel_projection.weight",
            f"layers.{layer}.mlp_conv.base_kernel",
            f"layers.{layer}.mlp_conv.kernel_projection.weight",
        }
        expected |= {
            f"layers.{layer}.self_attn.{proj}_proj.weight"
            for proj in ("q", "k", "v", "o")
        }

    assert names == expected


def test_conv_shapes_follow_the_checkpoint() -> None:
    layer = _drafter().layers[0]
    assert isinstance(layer, DFlash2TransformerBlock)
    conv = layer.attention_conv
    num_groups = HIDDEN // GROUP_SIZE
    assert list(conv.base_kernel.shape) == [2, TAPS, HIDDEN]
    assert list(conv.kernel_projection.weight.shape) == [
        2 * TAPS * num_groups,
        HIDDEN,
    ]


@pytest.mark.parametrize(
    ("layer_types", "expected"),
    [
        (None, [MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK] * LAYERS),
        (
            ["sliding_attention", "full_attention"],
            [
                MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK,
                MHAMaskVariant.NULL_MASK,
            ],
        ),
    ],
)
def test_attention_is_never_causal(
    layer_types: list[str] | None, expected: list[MHAMaskVariant]
) -> None:
    """DFlash2 attends bidirectionally inside the block.

    A causal variant here would still produce fluent-looking drafts while
    destroying acceptance, so pin the selection rather than the output.
    """
    for layer, want in zip(_drafter(layer_types).layers, expected, strict=True):
        assert isinstance(layer, DFlash2TransformerBlock)
        attention = layer.self_attn
        assert isinstance(attention, AttentionWithRope)
        assert attention.mask_variant == want


def test_sliding_layer_without_a_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="sliding_window"):
        _drafter(["sliding_attention"] * LAYERS, sliding_window=None)


def test_conv_rejects_geometry_the_checkpoint_cannot_produce() -> None:
    with pytest.raises(ValueError, match="must divide"):
        DFlash2GroupedConv(
            HIDDEN,
            taps=TAPS,
            group_size=7,
            block_size=BLOCK_SIZE,
            dtype=DType.float32,
            device=DeviceRef.CPU(),
        )
    with pytest.raises(ValueError, match="exceeds block_size"):
        DFlash2GroupedConv(
            HIDDEN,
            taps=BLOCK_SIZE + 1,
            group_size=GROUP_SIZE,
            block_size=BLOCK_SIZE,
            dtype=DType.float32,
            device=DeviceRef.CPU(),
        )
