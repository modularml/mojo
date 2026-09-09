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

"""Compiles and runs the Qwen3.5 MTP draft head.

The structural tests in ``test_mtp.py`` never execute ``__call__``, so graph
compilation, the KV argument order and the fusion concat could all regress
while they stayed green.

The concat order is the interesting one. ``fc.weight`` is
``[hidden, 2 * hidden]`` and embedding-first, and reversing it compiles and
runs -- it just drafts badly. Rather than compare against a torch reference,
this pins it with a selector: setting ``fc.weight`` to ``[I | 0]`` keeps only
the first half of the concatenation, so the output must be **invariant** to
``hidden_states``. A reversed concat would make that half the target hidden
state and the output would move.

Invariance alone would also hold for a degenerate graph -- all-zero or NaN
output is invariant to everything -- so the same fixture asserts the output
still **moves** with the tokens.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferValue, DeviceRef, Graph, TensorType, TensorValue
from max.nn.comm import Signals
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.mtp import Qwen3_5MTP
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context

# Real head geometry, with the hidden size and vocab shrunk: this test is
# about wiring and the concat order, and a 5120-wide identity selector at the
# real vocab costs minutes of weight upload for no extra coverage.
HIDDEN = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
INTERMEDIATE = 512
VOCAB = 128
EPS = 1e-6
SEQ = 6
PAGE_SIZE = 128


def _config(device: DeviceRef) -> Qwen3_5Config:
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
    )
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=1,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=512,
        intermediate_size=INTERMEDIATE,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
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
        use_subgraphs=False,
    )


def _head(config: Qwen3_5Config, device: DeviceRef) -> Qwen3_5MTP:
    return Qwen3_5MTP(
        config,
        embed_tokens=VocabParallelEmbedding(
            VOCAB, HIDDEN, DType.bfloat16, [device]
        ),
        rope=Llama3RotaryEmbedding(
            dim=HIDDEN,
            n_heads=HEADS,
            theta=1e7,
            max_seq_len=512,
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


def _embedding_first_selector() -> np.ndarray:
    """``[I | 0]``: keeps the embedding half of the concatenation.

    ``fc`` is applied as ``x @ w.T`` with ``w`` of shape
    ``[hidden, 2 * hidden]``, so the identity block must sit in the first
    ``hidden`` columns.
    """
    selector = np.zeros((HIDDEN, 2 * HIDDEN), dtype=np.float32)
    selector[:, :HIDDEN] = np.eye(HIDDEN, dtype=np.float32)
    return selector


def _load_weights(head: Qwen3_5MTP, rng: np.random.Generator) -> None:
    """Loads random weights, except ``fc`` which is the selector.

    Loading before the graph is built is required, not incidental:
    ``load_state_dict`` assigns each weight its fully-qualified name, and
    without that every submodule contributes a ``Weight`` called ``weight``
    and the graph rejects the second one.
    """
    state: dict[str, torch.Tensor] = {}
    for name, weight in head.raw_state_dict().items():
        shape = tuple(int(d) for d in weight.shape)
        value = (
            _embedding_first_selector()
            if name == "fc.weight"
            # Small, so `(1 + w)` keeps the norms near unity and the block
            # stays numerically well behaved.
            else rng.normal(0.0, 0.02, shape).astype(np.float32)
        )
        state[name] = torch.from_numpy(value).to(torch.bfloat16)
    head.load_state_dict(state, weight_alignment=1)


@pytest.fixture(scope="module")
def compiled():  # noqa: ANN201
    """Compiles the draft head once with the embedding-first selector."""
    device = DeviceRef.GPU(0)
    accelerator = Accelerator(0)
    session = InferenceSession(devices=[accelerator])
    config = _config(device)
    head = _head(config, device)
    _load_weights(head, np.random.default_rng(7))

    kv_params = config.kv_params
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=4,
    )

    signals = Signals(devices=[device])
    tokens_type = TensorType(DType.int64, ["total_seq_len"], device=device)
    hidden_type = TensorType(
        DType.bfloat16, ["total_seq_len", HIDDEN], device=device
    )
    row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], device=device
    )
    kv_inputs = kv_params.get_symbolic_inputs()

    with Graph(
        "Qwen3_5MTP",
        input_types=(
            tokens_type,
            hidden_type,
            row_offsets_type,
            *signals.input_types(),
            *kv_inputs.flatten(),
        ),
    ) as graph:
        tokens, hidden, row_offsets, *rest = graph.inputs
        signal_buffers = [
            rest[i].buffer for i in range(len(signals.input_types()))
        ]
        kv_rest = iter(rest[len(signals.input_types()) :])
        kv_collections = kv_params.unflatten_kv_inputs(kv_rest).inputs
        assert isinstance(tokens, TensorValue)
        assert isinstance(hidden, TensorValue)
        assert isinstance(row_offsets, TensorValue)
        assert all(isinstance(b, BufferValue) for b in signal_buffers)
        graph.output(
            *head(
                tokens,
                [hidden],
                signal_buffers,
                list(kv_collections),
                [row_offsets],
            )
        )

    model = session.load(graph, weights_registry=head.state_dict())
    return model, kv_manager, accelerator


def _bf16(value: np.ndarray) -> Buffer:
    """A bf16 device-ready buffer from float32 data.

    DLPack has no bf16, so the bits are built as ``uint16`` and reinterpreted
    -- the same approach ``test_state_warmup.py`` uses. Truncating the
    mantissa is fine here: it is deterministic, and every comparison in this
    file is between two runs of the same graph.
    """
    bits = (value.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return Buffer.from_numpy(bits).view(DType.bfloat16)


def _from_bf16(buffer: Buffer) -> np.ndarray:
    """Reads a bf16 buffer back as float32, again via the uint16 view."""
    bits = buffer.to(CPU()).view(DType.uint16).to_numpy()
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _run(
    compiled,  # noqa: ANN001
    tokens: np.ndarray,
    hidden: np.ndarray,
) -> np.ndarray:
    """Runs one prefill step and returns the draft's normed hidden state."""
    model, kv_manager, accelerator = compiled

    context = create_text_context(np.asarray(tokens))
    kv_manager.claim(context)
    kv_manager.alloc(context)
    try:
        kv_runtime = kv_manager.runtime_inputs([[context]])
        # `Signals.allocate` is the documented chokepoint: it enables peer
        # access, zeroes the barrier counters and sentinel-initializes the
        # Lamport region. A hand-rolled `Buffer.zeros` is both wrongly
        # initialized and too large for the pooled allocator's chunk.
        signal_buffers = Signals.allocate([accelerator])
        outputs = model.execute(
            Buffer.from_numpy(tokens.astype(np.int64)).to(accelerator),
            _bf16(hidden).to(accelerator),
            Buffer.from_numpy(np.array([0, len(tokens)], dtype=np.uint32)).to(
                accelerator
            ),
            *signal_buffers,
            *kv_runtime.flatten(),
        )
        assert isinstance(outputs[0], Buffer)
        return _from_bf16(outputs[0])
    finally:
        kv_manager.release(context)


def test_the_draft_head_compiles_and_runs(compiled) -> None:  # noqa: ANN001
    """The forward path builds, loads and executes at real head geometry."""
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, VOCAB, SEQ, dtype=np.int64)
    hidden = rng.normal(0.0, 1.0, (SEQ, HIDDEN))

    out = _run(compiled, tokens, hidden)

    assert out.shape == (SEQ, HIDDEN)
    assert np.isfinite(out).all()


def test_the_fusion_concat_is_embedding_first(compiled) -> None:  # noqa: ANN001
    """``fc = [I | 0]`` selects the embedding, so hidden must not matter.

    Reversing the concatenation would route the target hidden state through
    the surviving identity block and this output would move with it.
    """
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, VOCAB, SEQ, dtype=np.int64)
    hidden_a = rng.normal(0.0, 1.0, (SEQ, HIDDEN))
    hidden_b = rng.normal(0.0, 5.0, (SEQ, HIDDEN))

    out_a = _run(compiled, tokens, hidden_a)
    out_b = _run(compiled, tokens, hidden_b)

    np.testing.assert_allclose(out_a, out_b, rtol=0, atol=0)


def test_the_output_still_moves_with_the_tokens(compiled) -> None:  # noqa: ANN001
    """Guards the invariance above from passing on a degenerate graph.

    An all-zero or NaN output is invariant to every input, so the selector
    test proves nothing on its own.
    """
    rng = np.random.default_rng(3)
    hidden = rng.normal(0.0, 1.0, (SEQ, HIDDEN))
    tokens_a = rng.integers(0, VOCAB, SEQ, dtype=np.int64)
    tokens_b = (tokens_a + VOCAB // 2) % VOCAB

    out_a = _run(compiled, tokens_a, hidden)
    out_b = _run(compiled, tokens_b, hidden)

    assert not np.allclose(out_a, out_b, rtol=1e-3, atol=1e-3)
