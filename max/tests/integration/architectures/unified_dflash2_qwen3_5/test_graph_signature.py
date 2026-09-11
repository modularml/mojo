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
"""The fused DFlash2 graph's input signature, pinned slot by slot.

Mach's spec-step executor binds this signature positionally, so a reordering
or an added slot is an ABI break that the engine can only report as an arity
mismatch (or, worse, cannot report at all when the count happens to match).
The MTP graph is asserted from Mach's side at 223 slots; this asserts the
DFlash2 graph from MAX's side, and asserts that the two agree slot for slot
apart from the draft leaf's geometry, which is the one thing that differs.
"""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.state_cache import attn_cache
from max.pipelines.architectures.unified_dflash2_qwen3_5.model_config import (
    DRAFT_SLIDING_WINDOW,
    UnifiedDflash2Qwen3_5Config,
)
from max.pipelines.architectures.unified_dflash2_qwen3_5.unified_dflash2_qwen3_5 import (
    UnifiedDflash2Qwen3_5,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.unified_mtp_qwen3_5 import (
    UnifiedMTPQwen3_5,
)
from max.pipelines.speculative.config import SpeculativeConfig

HIDDEN = 64
VOCAB = 128
BLOCK = 8
DRAFT_LAYERS = 5
DRAFT_KV_HEADS = 2
DRAFT_HEAD_DIM = 16
PAGE_SIZE = 32
# Three linear-attention layers and one full-attention layer: the pool tail is
# per linear layer, so a mix is what makes the count formula falsifiable.
LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]
NUM_LINEAR = 3


def _target_config() -> Qwen3_5Config:
    device = DeviceRef.CPU()
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=len(LAYER_TYPES),
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            devices=[device],
            n_kv_heads=1,
            head_dim=16,
            num_layers=1,
            page_size=PAGE_SIZE,
        ),
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=16**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=list(LAYER_TYPES),
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
    )


def _fused_config() -> UnifiedDflash2Qwen3_5Config:
    device = DeviceRef.CPU()
    target = _target_config()
    draft_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=DRAFT_KV_HEADS,
        head_dim=DRAFT_HEAD_DIM,
        num_layers=DRAFT_LAYERS,
        page_size=PAGE_SIZE,
        window_size=DRAFT_SLIDING_WINDOW,
    )
    draft = Llama3Config(
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=DRAFT_KV_HEADS,
        num_hidden_layers=DRAFT_LAYERS,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=False,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=draft_kv,
        rms_norm_eps=1e-6,
        attention_multiplier=DRAFT_HEAD_DIM**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        sliding_window=DRAFT_SLIDING_WINDOW,
    )
    return UnifiedDflash2Qwen3_5Config(
        target=target,
        draft=draft,
        draft_kv_params=draft_kv,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash2", num_speculative_tokens=BLOCK - 1
        ),
        target_layer_ids=[0, 1, 2, 3, 3],
        layer_types=["sliding_attention"] * DRAFT_LAYERS,
        mask_token_id=VOCAB - 1,
        block_size=BLOCK,
        conv_kernel_size=2,
        conv_group_size=8,
        selector_rank=4,
        selector_top_k=3,
    )


def _signature(
    enable_structured_output: bool = True,
) -> tuple[TensorType | BufferType, ...]:
    config = _fused_config()
    module = UnifiedDflash2Qwen3_5(
        config, enable_structured_output=enable_structured_output
    )
    return module.input_types(config.get_kv_params())


def test_slot_count_matches_the_declared_formula() -> None:
    """``15 + 21D`` at ``D = 1``.

    The same formula the Qwen3.5 MTP graph satisfies: five ragged/host inputs,
    signals, two 6-slot KV leaves, batch_context_lengths, the eight-entry
    sampling tail, the bitmask triple, then three slots for each of the two
    state leaves -- its pool, the rows addressing it, and its shadow pool.
    The layer count no longer enters: a leaf is one pool however many layers
    index it.
    """
    types = _signature()
    assert len(types) == 15 + 21 * 1


def test_the_prefix_and_sampling_tail_are_the_canonical_ones() -> None:
    types = _signature()
    prefix = [(t.dtype, tuple(str(d) for d in t.shape)) for t in types[:5]]
    assert prefix == [
        (DType.int64, ("total_seq_len",)),
        (DType.uint32, ("input_row_offsets_len",)),
        (DType.uint32, ("input_row_offsets_len",)),
        (DType.int64, ("return_n_logits",)),
        (DType.int64, ("2",)),
    ]
    # host_input_row_offsets and return_n_logits are host-side.
    cpu = DeviceRef.CPU()
    assert types[2].device == cpu and types[3].device == cpu

    # draft_tokens .. in_thinking_phase, then the bitmask triple.
    tail = [(t.dtype, tuple(str(d) for d in t.shape)) for t in types[19:30]]
    assert tail == [
        (DType.int64, ("batch_size", "num_steps")),
        (DType.uint64, ("batch_size",)),
        (DType.float32, ("batch_size",)),
        (DType.int64, ("batch_size",)),
        (DType.int64, ()),
        (DType.float32, ("batch_size",)),
        (DType.float32, ()),
        (DType.bool, ("batch_size",)),
        (
            DType.int32,
            ("batch_size", "num_bitmask_positions", "packed_vocab_size"),
        ),
        (DType.int64, ("2",)),
        (
            DType.int32,
            ("batch_size", "num_bitmask_positions", "packed_vocab_size"),
        ),
    ]


def test_the_draft_leaf_is_the_drafters_own_windowed_geometry() -> None:
    """Slots 12-17: five drafter layers, bf16, bounded at the drafter's window.

    Every one of these is a silent-wrong-answer if it drifts from the Mach
    registry's draft group: a wrong layer count reads another layer's K/V, a
    wrong dtype reinterprets the bytes, and an unbounded leaf turns a
    per-slot constant into a cost that scales with ``--max-length``.
    """
    config = _fused_config()
    tree = config.get_kv_params()
    assert isinstance(tree, MultiKVCacheParams)
    draft = tree.children["draft"]
    assert isinstance(draft, MHAKVCacheParams)
    assert draft.num_layers == DRAFT_LAYERS
    assert draft.n_kv_heads == DRAFT_KV_HEADS
    assert draft.head_dim == DRAFT_HEAD_DIM
    assert draft.dtype == DType.bfloat16
    assert draft.window_size == DRAFT_SLIDING_WINDOW
    assert draft.group_id.is_sliding_window()
    # One page size across the tree, or the manager and the graph disagree.
    target = tree.children["target"]
    assert isinstance(target, MHAKVCacheParams)
    assert draft.page_size == target.page_size

    # Slot 12: the draft leaf's blocks, right after the target leaf's six.
    types = _signature()
    blocks = types[12]
    assert isinstance(blocks, BufferType)
    assert blocks.dtype == DType.bfloat16
    assert [str(d) for d in blocks.shape[1:]] == [
        "2",
        str(DRAFT_LAYERS),
        str(PAGE_SIZE),
        str(DRAFT_KV_HEADS),
        str(DRAFT_HEAD_DIM),
    ]


def test_a_quantized_target_leaf_does_not_quantize_the_draft_leaf() -> None:
    """``--kv-cache-dtype float8_e4m3fn`` must not reach the drafter's cache.

    The drafter fills it with its own unquantized projections, so a rewritten
    dtype is not a precision trade -- it is a reinterpretation of the bytes.
    """
    config = _fused_config()
    config.target.kv_params = replace(
        attn_cache(config.target.kv_params), dtype=DType.float8_e4m3fn
    )
    tree = config.get_kv_params()
    assert isinstance(tree, MultiKVCacheParams)
    target, draft = tree.children["target"], tree.children["draft"]
    assert isinstance(target, MHAKVCacheParams)
    assert isinstance(draft, MHAKVCacheParams)
    assert target.dtype == DType.float8_e4m3fn
    assert draft.dtype == DType.bfloat16


def test_the_state_pool_tail_matches_the_mtp_graphs() -> None:
    """Each state leaf's live pool, the rows addressing it, then its shadow
    pool -- the order Mach's Qwen slot layout already binds."""
    config = _fused_config()
    module = UnifiedDflash2Qwen3_5(config, enable_structured_output=True)
    fused = module.input_types(config.get_kv_params())
    mtp_kv = MultiKVCacheParams.from_params(
        {
            "target": config.target.kv_params,
            "draft": replace(attn_cache(config.target.kv_params), num_layers=1),
        }
    )
    mtp = UnifiedMTPQwen3_5(
        config.target,
        speculative_config=config.speculative_config,
        enable_structured_output=True,
    ).input_types(mtp_kv)

    assert len(fused) == len(mtp), (
        "the two graphs must present the same slot count, so Mach's Qwen"
        " layout binds both"
    )
    # A leaf contributes its pool, its rows and its shadow pool, per device,
    # so the tail is sized by the state leaves rather than by the layers.
    tail_start = len(fused) - 3 * len(module.state_regions)
    for a, b in zip(fused[tail_start:], mtp[tail_start:], strict=True):
        assert a.dtype == b.dtype
        assert [str(d) for d in a.shape] == [str(d) for d in b.shape]


def test_structured_output_off_drops_exactly_the_bitmask_triple() -> None:
    assert len(_signature(False)) == len(_signature(True)) - 3
