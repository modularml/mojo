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
"""Prefill on the fused DFlash2 graph, at the prompt lengths that crashed.

A prefill step hands the graph a zero-width draft axis, and the graph does not
agree with itself about that dim: the tensor shapes say zero -- ``recovered``
really is ``[batch, 0]`` -- while ``RaggedTokenMerger``'s ``_shape_to_scalar``
of the same dim can yield the compiled K. The accepted count therefore reaches
the committed-token gather as anything in ``[0, K]``, and ``ops.gather_nd`` is
not bounds-checked: against an unpadded width-1 table an index of K read off
the end of the row and returned 0. That 0 was every prompt's first token, and
it then became the drafter's block anchor, which is broadcast into a gather of
the selector's ``[248320, 256]`` predecessor codebook -- where an out-of-range
id is a device assert rather than a wrong answer.

Lengths 121 and 200 are the ones that turned that read into an assert in
serving; 137 and 282 only produced the wrong token. Both are covered here,
since a fix validated only on the quiet lengths proves nothing.

The inputs mirror Mach's per-phase contract: draft width 0 on prefill,
``return_n_logits = draft_width + 1``, and KV metadata shaped the way
``make_spec_kv_metadata`` shapes it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn.comm.allreduce import Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    MHAKVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.unified_dflash2_qwen3_5.model import GRAPH_NAME
from max.pipelines.architectures.unified_dflash2_qwen3_5.model_config import (
    UnifiedDflash2Qwen3_5Config,
)
from max.pipelines.architectures.unified_dflash2_qwen3_5.unified_dflash2_qwen3_5 import (
    UnifiedDflash2Qwen3_5,
)
from max.pipelines.speculative.config import (
    MAGIC_DRAFT_TOKEN_ID,
    SpeculativeConfig,
)

CRASHING_LENGTHS = (121, 200)
QUIET_LENGTHS = (137, 282)

H, VOCAB, PAGE = 64, 128, 32
BLOCK, TOPK = 8, 3
LAYER_TYPES = ["linear_attention", "full_attention"] * 2
NUM_LINEAR = 2
NUM_FULL = 2
DRAFT_LAYERS, DRAFT_KVH, DRAFT_HD = 2, 2, 16
CONV_KERNEL, LK, LV, LKD, LVD = 4, 1, 2, 128, 128
CONV_DIM = 2 * LK * LKD + LV * LVD
WINDOW = 2048


gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()


def make_config() -> UnifiedDflash2Qwen3_5Config:
    tkv = MHAKVCacheParams(
        dtype=DType.float32,
        devices=[gpu],
        n_kv_heads=1,
        head_dim=16,
        num_layers=NUM_FULL,
        page_size=PAGE,
    )
    target = Qwen3_5Config(
        hidden_size=H,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=len(LAYER_TYPES),
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=4096,
        intermediate_size=H * 2,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=tkv,
        norm_dtype=DType.float32,
        rms_norm_eps=1e-6,
        attention_multiplier=16**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[gpu],
        clip_qkv=None,
        layer_types=list(LAYER_TYPES),
        linear_key_head_dim=LKD,
        linear_value_head_dim=LVD,
        linear_num_key_heads=LK,
        linear_num_value_heads=LV,
        linear_conv_kernel_dim=CONV_KERNEL,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
        state_pool_dtype=DType.float32,
    )
    dkv = MHAKVCacheParams(
        dtype=DType.float32,
        devices=[gpu],
        n_kv_heads=DRAFT_KVH,
        head_dim=DRAFT_HD,
        num_layers=DRAFT_LAYERS,
        page_size=PAGE,
        window_size=WINDOW,
    )
    draft = Llama3Config(
        hidden_size=H,
        num_attention_heads=4,
        num_key_value_heads=DRAFT_KVH,
        num_hidden_layers=DRAFT_LAYERS,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=4096,
        intermediate_size=H * 2,
        interleaved_rope_weights=False,
        vocab_size=VOCAB,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=dkv,
        rms_norm_eps=1e-6,
        attention_multiplier=DRAFT_HD**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[gpu],
        clip_qkv=None,
        sliding_window=WINDOW,
    )
    return UnifiedDflash2Qwen3_5Config(
        target=target,
        draft=draft,
        draft_kv_params=dkv,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash2", num_speculative_tokens=BLOCK - 1
        ),
        target_layer_ids=[0, 1, 2, 3],
        layer_types=["sliding_attention"] * DRAFT_LAYERS,
        mask_token_id=VOCAB - 1,
        block_size=BLOCK,
        conv_kernel_size=2,
        conv_group_size=8,
        selector_rank=4,
        selector_top_k=TOPK,
    )


MAGIC = MAGIC_DRAFT_TOKEN_ID
Step = Callable[..., list[np.ndarray]]


def _build() -> Step:
    cfg = make_config()
    nn = UnifiedDflash2Qwen3_5(cfg, enable_structured_output=False)
    rng = np.random.default_rng(3)
    nn.load_state_dict(
        {
            n: (
                rng.standard_normal([int(d) for d in w.shape], dtype=np.float32)
                * 0.05
            )
            for n, w in nn.raw_state_dict().items()
        },
        weight_alignment=1,
        strict=False,
        override_quantization_encoding=True,
    )
    registry = nn.state_dict()

    kvp = cfg.get_kv_params()
    assert isinstance(kvp, MultiKVCacheParams)
    dev = Accelerator()
    session = InferenceSession(devices=[dev])

    with Graph(GRAPH_NAME, input_types=nn.input_types(kvp)) as g:
        (tokens, row_offsets, host_row_offsets, ret_n, dp_splits, *rest) = (
            g.inputs
        )
        it = iter(rest)
        sigs = [next(it).buffer]
        tree = kvp.unflatten_kv_inputs(it)
        assert isinstance(tree, MultiKVCacheInputs)
        tleaf, dleaf = tree.children["target"], tree.children["draft"]
        assert isinstance(tleaf, KVCacheInputs) and isinstance(
            dleaf, KVCacheInputs
        )
        next(it)  # batch_context_lengths
        dt = next(it).tensor
        seed, temp, tk, mk, tp, mtp_, think = (
            next(it).tensor for _ in range(7)
        )
        pin = wait = scratch = None
        # One pool per device per leaf, then the rows addressing them, then
        # the shadows -- the order `input_types` lays the tail out in.
        live_conv, live_rec = next(it).buffer, next(it).buffer
        conv_rows, rec_rows = next(it).tensor, next(it).tensor
        shadow_conv, shadow_rec = next(it).buffer, next(it).buffer
        out = nn(
            tokens=tokens.tensor,
            input_row_offsets=row_offsets.tensor,
            draft_tokens=dt,
            signal_buffers=sigs,
            target_kv=list(tleaf.inputs),
            draft_kv=list(dleaf.inputs),
            return_n_logits=ret_n.tensor,
            host_input_row_offsets=host_row_offsets.tensor,
            data_parallel_splits=dp_splits.tensor,
            seed=seed,
            temperature=temp,
            top_k=tk,
            max_k=mk,
            top_p=tp,
            min_top_p=mtp_,
            in_thinking_phase=think,
            live_conv_pools=[live_conv],
            live_recurrent_pools=[live_rec],
            live_conv_row_ids=[conv_rows],
            live_recurrent_row_ids=[rec_rows],
            shadow_conv_pools=[shadow_conv],
            shadow_recurrent_pools=[shadow_rec],
            pinned_bitmask=pin,
            wait_payload=wait,
            device_bitmask_scratch=scratch,
        )
        g.output(*out)

    model = session.load(g, weights_registry=registry)
    sig = Signals.allocate([dev])

    def b(x: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(x)).to(dev)

    def h(x: np.ndarray) -> Buffer:
        # ascontiguousarray promotes 0-d to (1,), which breaks the scalar slots.
        a = np.asarray(x)
        return Buffer.from_numpy(a if a.ndim == 0 else np.ascontiguousarray(a))

    MAXPAGES, SLOTS = 512, 4
    tblocks = Buffer.from_numpy(
        np.zeros((MAXPAGES + 1, 2, NUM_FULL, PAGE, 1, 16), np.float32)
    ).to(dev)
    dblocks = Buffer.from_numpy(
        np.zeros(
            (MAXPAGES + 1, 2, DRAFT_LAYERS, PAGE, DRAFT_KVH, DRAFT_HD),
            np.float32,
        )
    ).to(dev)

    def step(
        P: int,
        num_steps: int,
        cached: int,
        drafts: list[int] | None = None,
    ) -> list[np.ndarray]:
        """One forward with Mach's contract. Returns (num_accepted, next, drafts)."""
        batch = 1
        active = P if num_steps == 0 else 1
        target_width = active + num_steps
        abs_max_cache = cached + active + num_steps + BLOCK
        lut_w = 24
        lut = np.arange(lut_w, dtype=np.uint32)[None, :]
        toks = np.random.default_rng(P).integers(
            0, VOCAB - 1, active, dtype=np.int64
        )
        dtv = (
            np.full((batch, num_steps), MAGIC, dtype=np.int64)
            if drafts is None
            else np.array(drafts, dtype=np.int64).reshape(batch, num_steps)
        )
        args = [
            b(toks),
            b(np.array([0, active], np.uint32)),
            h(np.array([0, active], np.uint32)),
            h(np.array([num_steps + 1], np.int64)),
            h(np.array([0, 1], np.int64)),
            *sig,
            tblocks,
            b(np.array([cached] * batch, np.uint32)),
            b(lut),
            h(np.array([target_width], np.uint32)),
            h(np.array([abs_max_cache], np.uint32)),
            h(np.array([batch, target_width, 1, abs_max_cache], np.int64)),
            dblocks,
            b(np.array([cached] * batch, np.uint32)),
            b(lut),
            h(np.array([target_width], np.uint32)),
            h(np.array([abs_max_cache], np.uint32)),
            h(np.array([batch, target_width, 1, abs_max_cache], np.int64)),
            h(np.array([0], np.int32)),
            b(dtv),
            b(np.array([7], np.uint64)),
            b(np.zeros(batch, np.float32)),
            b(np.ones(batch, np.int64)),
            h(np.array(1, np.int64).reshape(())),
            b(np.ones(batch, np.float32)),
            h(np.array(1.0, np.float32).reshape(())),
            b(np.zeros(batch, np.bool_)),
        ]

        def conv_pool() -> Buffer:
            # One pool per leaf now: a block's layers are consecutive rows,
            # so the row count is pages times layers.
            return Buffer.from_numpy(
                np.zeros(
                    (SLOTS * NUM_LINEAR, CONV_DIM, CONV_KERNEL - 1), np.float32
                )
            ).to(dev)

        def rec_pool() -> Buffer:
            return Buffer.from_numpy(
                np.zeros((SLOTS * NUM_LINEAR, LV, LKD, LVD), np.float32)
            ).to(dev)

        # Request i holds block i, whose layers are rows i*L..i*L+L-1.
        rows = b(
            np.arange(batch * NUM_LINEAR, dtype=np.uint32).reshape(
                batch, NUM_LINEAR
            )
        )
        args += [conv_pool(), rec_pool(), rows, rows, conv_pool(), rec_pool()]
        o = model.execute(*args)
        return [np.array(x.to(CPU()).to_numpy()) for x in o]

    return step


@pytest.fixture(scope="module")
def step() -> Step:
    """Compiles the fused graph once; returns a one-forward driver."""
    return _build()


@pytest.mark.parametrize("prompt_len", CRASHING_LENGTHS + QUIET_LENGTHS)
def test_prefill_commits_a_real_token(step: Step, prompt_len: int) -> None:
    """Every id the graph emits at prefill has to be a real vocabulary id.

    The committed token is the drafter's block anchor, and the selector
    gathers a codebook row with it, so an out-of-range value here is a device
    assert. Before the committed-token table was padded this was 0 for every
    prompt, and at 121 and 200 the drafts that followed asserted.
    """
    _, next_tokens, drafts = step(prompt_len, 0, 0)
    assert 0 <= int(next_tokens[0]) < VOCAB
    assert ((drafts >= 0) & (drafts < VOCAB)).all()


def test_prefill_does_not_commit_one_stuck_token(step: Step) -> None:
    """Different prompts must commit different tokens.

    The bug's signature was a *universal* id 0: an off-the-end read returns
    the same value whatever the prompt, so a per-prompt range check alone
    would still pass. Comparing across prompts is what catches it.
    """
    committed = {
        p: int(step(p, 0, 0)[1][0]) for p in CRASHING_LENGTHS + QUIET_LENGTHS
    }
    assert set(committed.values()) != {0}, f"universal id-0: {committed}"
    assert len(set(committed.values())) > 1, (
        f"every prompt committed the same token: {committed}"
    )


def test_decode_is_unaffected(step: Step) -> None:
    """Decode was healthy before the fix and has to stay that way."""
    num_accepted, next_tokens, drafts = step(1, BLOCK - 1, 200)
    assert int(num_accepted[0]) == 0, "all-magic drafts accept nothing"
    assert 0 <= int(next_tokens[0]) < VOCAB
    assert drafts.shape == (1, BLOCK - 1)


def test_prefill_reports_no_accepted_tokens(step: Step) -> None:
    """A prefill step accepted nothing, and must report nothing.

    Mach advances the sequence by this count, so a prefill row claiming the
    compiled K would skip K tokens. It read K until the empty-axis reduction
    was fixed in the kernels (a reduce over a zero-extent axis returned
    without writing, so the guard downstream of it never ran).
    """
    for prompt_len in CRASHING_LENGTHS + QUIET_LENGTHS:
        num_accepted, _, _ = step(prompt_len, 0, 0)
        assert int(num_accepted[0]) == 0, f"prompt_len={prompt_len}"


def test_the_count_still_rises_when_a_draft_is_accepted(step: Step) -> None:
    """Guard against a fix that pins the count to zero instead of correcting it.

    Every other case here drafts all-magic tokens, which are always rejected,
    so a count wrongly forced to zero would look identical. Feeding back the
    target's own token at the first draft slot makes that slot acceptable, and
    the count has to follow.
    """
    _, next_tokens, _ = step(1, BLOCK - 1, 200)
    targets_own = int(next_tokens[0])
    drafts = [targets_own] + [MAGIC] * (BLOCK - 2)
    num_accepted, _, _ = step(1, BLOCK - 1, 200, drafts=drafts)
    assert int(num_accepted[0]) >= 1, (
        "the target's own token at slot 0 must be accepted; a count stuck at"
        " zero would pass every other case in this file"
    )
