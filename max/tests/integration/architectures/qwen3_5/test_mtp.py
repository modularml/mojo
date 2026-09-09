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

"""Structural tests for the Qwen3.5 MTP (NextN) draft head.

Every failure mode this head has is silent. Loading it against a mismatched
name set drops a tensor and drafts badly; taking ``config.dtype`` instead of
``config.compute_dtype`` reads the BF16 body as packed NVFP4; quantizing the
draft against the MLP scheme invents scale tensors the checkpoint does not
ship. None of those raise, so they are pinned here.

The numerical gate -- the draft's hidden state against a hand-written torch
reference, and the embedding-first ``fc`` concat order -- needs a GPU and the
real checkpoint; it lives in ``mach/docs/qwen38_27b/mtp-spec.md``.
"""

from __future__ import annotations

import functools

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.mtp import Qwen3_5MTP
from max.pipelines.architectures.qwen3_5.quantization import (
    Qwen3_5QuantScheme,
    parse_quant_scheme,
)
from max.pipelines.architectures.qwen3_5.qwen3_5 import (
    Qwen3_5FullAttentionBlock,
)

# Qwen3.8-27B's real geometry, shrunk only where the head does not care.
HIDDEN = 5120
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
INTERMEDIATE = 17408
VOCAB = 248320
EPS = 1e-6

# The 15 `mtp.*` tensors the checkpoint ships, with the `mtp.` prefix stripped
# the way a weight adapter would. Verified against the checkpoint inventory in
# `mach/docs/qwen38_27b/checkpoint-audit.md`.
CHECKPOINT_MTP_WEIGHTS = {
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "fc.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "norm.weight",
}

CHECKPOINT_MTP_SHAPES = {
    "fc.weight": (HIDDEN, 2 * HIDDEN),
    "layers.0.self_attn.q_proj.weight": (2 * HEADS * HEAD_DIM, HIDDEN),
    "layers.0.self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
    "layers.0.self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
    "layers.0.self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
    "layers.0.self_attn.q_norm.weight": (HEAD_DIM,),
    "layers.0.self_attn.k_norm.weight": (HEAD_DIM,),
    "layers.0.mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
    "layers.0.mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
    "layers.0.mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
    "norm.weight": (HIDDEN,),
}


def _nvfp4_scheme() -> Qwen3_5QuantScheme | None:
    """The MIXED_PRECISION scheme of the real checkpoint, at one layer.

    Only what the parser reads: the packed-NVFP4 storage dtype it implies, and
    the bf16 compute dtype it carries alongside.
    """
    return parse_quant_scheme(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "lm_head": {"quant_algo": "NVFP4", "group_size": 16},
                "model.language_model.layers.0.mlp.gate_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.language_model.layers.0.mlp.up_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.language_model.layers.0.mlp.down_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.language_model.layers.0.self_attn.q_proj": {
                    "quant_algo": "FP8"
                },
                "model.language_model.layers.0.self_attn.k_proj": {
                    "quant_algo": "FP8"
                },
                "model.language_model.layers.0.self_attn.v_proj": {
                    "quant_algo": "FP8"
                },
                "model.language_model.layers.0.self_attn.o_proj": {
                    "quant_algo": "FP8"
                },
            },
        },
        {
            "embed_tokens.weight": WeightData.from_numpy(
                np.zeros((8, 4), dtype=np.float32), "embed_tokens.weight"
            ).astype(DType.bfloat16)
        },
        1,
    )


def _build(
    device: DeviceRef | None = None,
    embed_tokens: VocabParallelEmbedding | None = None,
) -> Qwen3_5MTP:
    """The draft head under a MIXED_PRECISION (packed-NVFP4) config."""
    device = DeviceRef.GPU(0) if device is None else device
    scheme = _nvfp4_scheme()
    assert scheme is not None
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=HEAD_DIM,
    )
    config = Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=1,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=4096,
        intermediate_size=INTERMEDIATE,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        # A MIXED_PRECISION export resolves the model dtype to packed NVFP4
        # storage; the draft body is BF16 and must not follow it.
        dtype=DType.uint8,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        norm_dtype=DType.bfloat16,
        rms_norm_eps=EPS,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=["full_attention"],
        partial_rotary_factor=0.25,
        quant_scheme=scheme,
        use_subgraphs=False,
    )
    assert config.dtype == DType.uint8
    assert config.compute_dtype == DType.bfloat16
    return Qwen3_5MTP(
        config,
        embed_tokens=embed_tokens
        if embed_tokens is not None
        else VocabParallelEmbedding(
            VOCAB, HIDDEN, config.compute_dtype, [device]
        ),
        rope=Llama3RotaryEmbedding(
            dim=HIDDEN,
            n_heads=HEADS,
            theta=1e7,
            max_seq_len=4096,
            head_dim=int(HEAD_DIM * 0.25),
            interleaved=True,
            scaling_params=None,
        ),
        create_norm=functools.partial(
            RMSNorm,
            HIDDEN,
            dtype=DType.bfloat16,
            eps=EPS,
            weight_offset=1.0,
            multiply_before_cast=False,
        ),
        kv_layer_idx=0,
    )


def test_weight_names_are_the_checkpoint_names() -> None:
    """The head loads from the checkpoint's own names, minus the `mtp.` prefix.

    `load_state_dict(strict=False)` drops a mismatch in either direction
    without a word, so a rename here is a silently worse draft rather than a
    load error.
    """
    mtp = _build()
    names = set(mtp.raw_state_dict())
    # `embed_tokens` is the target's table, injected rather than owned.
    names.discard("embed_tokens.weight")
    assert names == CHECKPOINT_MTP_WEIGHTS


def test_weight_shapes_match_the_checkpoint() -> None:
    mtp = _build()
    weights = mtp.raw_state_dict()
    for name, shape in CHECKPOINT_MTP_SHAPES.items():
        assert tuple(int(d) for d in weights[name].shape) == shape, name


def test_the_body_is_bf16_under_a_packed_nvfp4_config() -> None:
    """All 15 `mtp.*` tensors ship BF16 even when the MLPs are NVFP4.

    Taking `config.dtype` (`uint8` for a packed NVFP4 export) instead of
    `config.compute_dtype` reads two 4-bit codes per byte out of BF16 bytes:
    plausible tensors, garbage drafts, no error.
    """
    mtp = _build()
    for name, weight in mtp.raw_state_dict().items():
        if name == "embed_tokens.weight":
            continue
        assert weight.dtype == DType.bfloat16, f"{name} is {weight.dtype}"


def test_no_quantization_scales_are_expected() -> None:
    """The draft is unquantized, so it must not ask for scale tensors.

    The checkpoint ships none under `mtp.`; a quantized construction would
    look for `weight_scale` and fail the load, or worse, be handed the MLP's.
    """
    mtp = _build()
    scales = [
        name
        for name in mtp.raw_state_dict()
        if name.endswith(("weight_scale", "weight_scale_2", "input_scale"))
    ]
    assert scales == []


def test_the_fusion_projection_takes_two_hidden_states() -> None:
    """`fc` consumes `concat(normed_embedding, normed_hidden)`, embedding first.

    Only the width is checkable without running the graph; the order is gated
    numerically against the torch reference (see the module docstring).
    """
    mtp = _build()
    assert mtp.fc.weight.shape[-1] == 2 * HIDDEN
    assert mtp.fc.weight.shape[0] == HIDDEN


def test_the_norms_carry_the_unit_offset() -> None:
    """Every `mtp.*` norm is `normalize(x) * (1 + w)`.

    The stored weights are largely negative -- `pre_fc_norm_embedding` runs
    `[-0.750, -0.186]` -- so reading them as a plain multiplier flips the sign
    of the whole embedding branch.
    """
    mtp = _build()
    block = mtp.layers[0]
    assert isinstance(block, Qwen3_5FullAttentionBlock)
    for norm in (
        mtp.pre_fc_norm_embedding,
        mtp.pre_fc_norm_hidden,
        mtp.norm,
        # The decoder layer's norms load from `mtp.*` names too, and its Q/K
        # norms are built by the attention rather than `create_norm`, so they
        # carry the offset independently.
        block.input_layernorm,
        block.post_attention_layernorm,
        block.self_attn.q_norm,
        block.self_attn.k_norm,
    ):
        assert norm.weight_offset == 1.0


def test_the_embedding_table_is_shared_not_copied() -> None:
    """`mtp_use_dedicated_embeddings: false`: one table, two consumers.

    The checkpoint ships neither `mtp.embed_tokens` nor `mtp.lm_head`.
    """
    embed = VocabParallelEmbedding(
        VOCAB, HIDDEN, DType.bfloat16, [DeviceRef.GPU(0)]
    )
    mtp = _build(embed_tokens=embed)
    # Asserting after a reassignment would only prove Python allows it; the
    # table has to survive `__init__` uncopied for the two graphs to
    # reference one weight.
    assert mtp.embed_tokens is embed
