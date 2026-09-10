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
"""Qwen3.5's hidden-state tap, which the DFlash2 drafter reads its context from.

Two properties carry the whole tap contract, and both fail silently:

- The captures come out in **ascending layer order**, not in the order
  ``target_layer_ids`` was written. ``fc``'s column blocks are laid out that
  way in the checkpoint, so a permuted capture list feeds each block the wrong
  layer's states and only shows up as lost acceptance.
- The hook is **off unless asked for**, so the base Qwen3.5 graph is byte
  identical to what it was before DFlash2 existed.

The tapped layers straddle both Qwen3.5 layer kinds (5, 19, 33, 47 and 61 are
linear / full / linear / full / linear on the 27B), so a mixed stack is what
makes the test meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn.comm.allreduce import Signals
from max.nn.kv_cache import KVCacheInputs, MHAKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.qwen3_5 import Qwen3_5
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context

HIDDEN = 64
VOCAB = 128
PAGE_SIZE = 32
SEQ_LEN = 6
# linear / full / linear / full: the taps below straddle both kinds, the way
# the real [5, 19, 33, 47, 61] do.
LAYER_TYPES = ["linear_attention", "full_attention"] * 2
NUM_FULL = 2
NUM_LINEAR = 2
CONV_KERNEL = 4
LINEAR_KEY_HEADS = 1
LINEAR_VALUE_HEADS = 2
# The recurrence kernel is only compiled for 128 x 128 heads.
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
CONV_DIM = 2 * LINEAR_KEY_HEADS * LINEAR_KEY_DIM + (
    LINEAR_VALUE_HEADS * LINEAR_VALUE_DIM
)


def _config(target_layer_ids: list[int] | None) -> Qwen3_5Config:
    device = DeviceRef.GPU()
    config = Qwen3_5Config(
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
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.float32,
            devices=[device],
            n_kv_heads=1,
            head_dim=16,
            num_layers=NUM_FULL,
            page_size=PAGE_SIZE,
        ),
        norm_dtype=DType.float32,
        rms_norm_eps=1e-6,
        attention_multiplier=16**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=list(LAYER_TYPES),
        linear_key_head_dim=LINEAR_KEY_DIM,
        linear_value_head_dim=LINEAR_VALUE_DIM,
        linear_num_key_heads=LINEAR_KEY_HEADS,
        linear_num_value_heads=LINEAR_VALUE_HEADS,
        linear_conv_kernel_dim=CONV_KERNEL,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
        state_pool_dtype=DType.float32,
    )
    config.target_layer_ids = target_layer_ids
    return config


def _run(
    target_layer_ids: list[int] | None,
    *,
    return_all_hidden_states: bool = False,
) -> list[np.ndarray]:
    """Runs a tiny Qwen3.5 with fixed weights and returns its tap outputs."""
    config = _config(target_layer_ids)
    model = Qwen3_5(config)
    model.return_logits = ReturnLogits.LAST_TOKEN
    if return_all_hidden_states:
        model.return_hidden_states = ReturnHiddenStates.ALL
    else:
        model.return_hidden_states = (
            ReturnHiddenStates.SELECTED_LAYERS
            if target_layer_ids
            else ReturnHiddenStates.NONE
        )

    rng = np.random.default_rng(11)
    model.load_state_dict(
        {
            name: (
                rng.standard_normal(
                    [int(d) for d in weight.shape], dtype=np.float32
                )
                * 0.05
            )
            for name, weight in model.raw_state_dict().items()
        },
        weight_alignment=1,
        strict=False,
    )
    registry = model.state_dict()

    device = Accelerator()
    session = InferenceSession(devices=[device])
    with Graph(
        "qwen3_5_taps", input_types=model.input_types(config.kv_params)
    ) as graph:
        tokens, row_offsets, return_n_logits, *rest = graph.inputs
        it = iter(rest)
        signal_buffers = [next(it).buffer]
        leaf = config.kv_params.unflatten_kv_inputs(it)
        assert isinstance(leaf, KVCacheInputs)
        slot_idx = [next(it).tensor]
        pools = [[next(it).buffer for _ in range(NUM_LINEAR)] for _ in range(2)]
        outputs = model(
            tokens.tensor,
            list(leaf.inputs),
            return_n_logits.tensor,
            row_offsets.tensor,
            signal_buffers,
            slot_idx,
            [pools[0]],
            [pools[1]],
        )
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=registry)

    def buf(x: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(x)).to(device)

    kv_manager = PagedKVCacheManager(
        params=config.kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=2,
    )
    ctx = create_text_context(
        np.arange(SEQ_LEN, dtype=np.int64), max_length=128
    )
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)
    kv_inputs = list(
        kv_manager.runtime_inputs_for_leaf([[ctx]]).inputs[0].flatten()
    )

    results = compiled.execute(
        buf(np.arange(SEQ_LEN, dtype=np.int64)),
        buf(np.array([0, SEQ_LEN], dtype=np.uint32)),
        Buffer.from_numpy(np.array([1], dtype=np.int64)),
        *Signals.allocate([device]),
        *kv_inputs,
        buf(np.array([0], dtype=np.uint32)),
        *[
            buf(np.zeros((2, CONV_DIM, CONV_KERNEL - 1), dtype=np.float32))
            for _ in range(NUM_LINEAR)
        ],
        *[
            buf(
                np.zeros(
                    (2, LINEAR_VALUE_HEADS, LINEAR_KEY_DIM, LINEAR_VALUE_DIM),
                    dtype=np.float32,
                )
            )
            for _ in range(NUM_LINEAR)
        ],
    )
    # LAST_TOKEN logits, then one capture per tapped layer.
    return [np.array(r.to(CPU()).to_numpy()) for r in results[1:]]


def test_no_taps_requested_means_no_captures() -> None:
    """The base graph is unchanged: the hook costs nothing when unused."""
    assert _run(None) == []


@pytest.mark.parametrize("order", [[1, 3], [3, 1]])
def test_captures_come_out_in_ascending_layer_order(
    order: list[int],
) -> None:
    """Writing ``target_layer_ids`` backwards must not permute the captures.

    ``fc`` consumes one column block per tap in ascending layer order, so a
    list-ordered capture would feed every block the wrong layer.
    """
    ascending = _run([1, 3])
    assert len(ascending) == 2
    got = _run(order)
    for a, b in zip(ascending, got, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_tap_reads_its_own_layer() -> None:
    """Distinct layers, and each capture equal to that layer's own single-tap
    run -- so the list is a per-layer read, not the same tensor N times."""
    both = _run([1, 3])
    (only_1,) = _run([1])
    (only_3,) = _run([3])
    np.testing.assert_array_equal(both[0], only_1)
    np.testing.assert_array_equal(both[1], only_3)
    assert not np.array_equal(only_1, only_3)


def test_both_layer_kinds_are_tappable() -> None:
    """Layer 0 is linear-attention and layer 1 is full-attention; one hook
    covers both, which is what makes taps 5/19/33/47/61 free."""
    captures = _run([0, 1])
    assert len(captures) == 2
    assert captures[0].shape == captures[1].shape == (SEQ_LEN, HIDDEN)
    assert not np.array_equal(captures[0], captures[1])


def test_a_tap_captures_the_layers_output_not_its_input() -> None:
    """Tapping the last layer must equal the stack's own final hidden state.

    This is the off-by-one the whole tap rests on: the reference reads
    ``hidden_states[layer_id + 1]``, HuggingFace's name for the *output* of
    layer ``layer_id``. A hook that fired one layer early would still produce
    five plausible tap tensors in ascending order, and every other test here
    would still pass.
    """
    last = len(LAYER_TYPES) - 1
    (tapped,) = _run([last])
    (final,) = _run(None, return_all_hidden_states=True)
    np.testing.assert_array_equal(tapped, final)
