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

"""The Qwen3.5 state rollback, driven at a DFlash2 block rather than K = 2.

The Qwen3.5 MTP rollback is reused verbatim by the DFlash2 graph, so what
needs proving is that it is K-agnostic in fact and not only in argument. These
tests run the two real state kernels over an 8-row verify window and check the
live pools land exactly where a forward over the accepted prefix alone would
have left them.

The failure this guards is silent. transformers' ``LinearAttentionLayer.crop``
is a no-op, so the obvious port of DFlash v1's loop rolls back the 16
full-attention layers, leaves the 48 gated-DeltaNet layers advanced over the
whole block, and raises nothing -- it just generates text the target did not
choose. ``test_leaving_the_pools_advanced_is_detectable`` is that bug, asserted
to be visible.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Dim, Graph, TensorType, ops
from max.nn.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_recurrence_fwd,
)
from max.pipelines.architectures.qwen3_5.layers.gated_deltanet import (
    GatedDeltaReplayInputs,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.state_rollback import (
    accepted_row_plan,
    replay_state_pools,
    snapshot_state_pools,
)

BLOCK = 8
"""DFlash2 verifies an anchor plus seven drafts in one window."""

NUM_DRAFTS = BLOCK - 1
# The recurrence kernel is only compiled for 128 x 128 heads, and conv_dim is
# the layout the projections produce: 2 x num_key_heads x key_head_dim for
# Q/K plus num_value_heads x value_head_dim for V.
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
NUM_K_HEADS = 1
NUM_V_HEADS = 2
CONV_DIM = 2 * NUM_K_HEADS * KEY_HEAD_DIM + NUM_V_HEADS * VALUE_HEAD_DIM
CONV_KERNEL = 4
MAX_SLOTS = 4


def _rand(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def _run(
    batch: int, accepted: list[int], *, replay_full_window: bool = False
) -> dict[str, np.ndarray]:
    """Verifies a block, rolls back, and independently replays the prefix.

    Returns the live, shadow and reference pools. ``live`` is what the
    rollback produced; ``reference`` is what the same kernels produce over a
    host-sliced accepted prefix, built without ``accepted_row_plan``, so the
    two agreeing is not a tautology.
    """
    rng = np.random.default_rng(7)
    total = batch * BLOCK
    qkv = _rand(rng, total, CONV_DIM)
    conv_weight = _rand(rng, CONV_DIM, CONV_KERNEL)
    decay = -np.abs(_rand(rng, total, NUM_V_HEADS))
    beta = np.abs(_rand(rng, total, NUM_V_HEADS))
    merged_offsets = np.arange(batch + 1, dtype=np.uint32) * BLOCK
    slots = np.arange(batch, dtype=np.uint32)
    init_conv = _rand(rng, MAX_SLOTS, CONV_DIM, CONV_KERNEL - 1)
    init_rec = _rand(rng, MAX_SLOTS, NUM_V_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM)

    # The reference prefix, sliced on the host: rows [0, 1 + accepted) of each
    # request's window, which is what the step actually commits.
    keep = [1 + a for a in accepted]
    ref_rows = np.concatenate(
        [np.arange(b * BLOCK, b * BLOCK + keep[b]) for b in range(batch)]
    ).astype(np.int64)
    ref_offsets = np.concatenate([[0], np.cumsum(keep)]).astype(np.uint32)

    gpu = DeviceRef.GPU()
    conv_pool_type = BufferType(
        DType.float32, [MAX_SLOTS, CONV_DIM, CONV_KERNEL - 1], device=gpu
    )
    rec_pool_type = BufferType(
        DType.float32,
        [MAX_SLOTS, NUM_V_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
        device=gpu,
    )
    types: list[TensorType | BufferType] = [
        TensorType(DType.float32, [total, CONV_DIM], device=gpu),
        TensorType(DType.float32, [CONV_DIM, CONV_KERNEL], device=gpu),
        TensorType(DType.float32, [total, NUM_V_HEADS], device=gpu),
        TensorType(DType.float32, [total, NUM_V_HEADS], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        TensorType(DType.int64, ["batch_size"], device=gpu),
        TensorType(DType.uint32, ["batch_size"], device=gpu),
        TensorType(DType.int64, [len(ref_rows)], device=gpu),
        TensorType(DType.uint32, [batch + 1], device=gpu),
        conv_pool_type,  # live conv
        rec_pool_type,  # live recurrent
        conv_pool_type,  # shadow conv
        rec_pool_type,  # shadow recurrent
        conv_pool_type,  # reference conv
        rec_pool_type,  # reference recurrent
    ]

    with Graph("dflash2_block_rollback", input_types=types) as graph:
        (
            qkv_v,
            conv_w,
            decay_v,
            beta_v,
            offsets_v,
            accepted_v,
            slots_v,
            ref_rows_v,
            ref_offsets_v,
        ) = (v.tensor for v in graph.inputs[:9])
        live_conv, live_rec, shadow_conv, shadow_rec, ref_conv, ref_rec = (
            v.buffer for v in graph.inputs[9:]
        )

        batch_scalar = ops.shape_to_tensor([slots_v.shape[0]])[0]
        # One linear layer here, so a block's rows are the block itself and
        # the span the snapshot fills is just the batch.
        live_rows = ops.unsqueeze(slots_v, -1)
        snapshot_state_pools(
            [live_conv], [shadow_conv], [live_rows], batch_scalar
        )
        snapshot_state_pools(
            [live_rec], [shadow_rec], [live_rows], batch_scalar
        )
        shadow_slots = ops.range(
            start=0,
            stop=slots_v.shape[0],
            out_dim="batch_size",
            device=gpu,
            dtype=DType.uint32,
        )

        # The verify: the whole block, on the shadow pools.
        verify_conv = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv_v,
            conv_weight=conv_w,
            conv_state=shadow_conv,
            slot_idx=shadow_slots,
            input_row_offsets=offsets_v,
        )
        gated_delta_recurrence_fwd(
            qkv_conv_output=ops.silu(verify_conv),
            decay_per_token=decay_v,
            beta_per_token=beta_v,
            recurrent_state=shadow_rec,
            slot_idx=shadow_slots,
            input_row_offsets=offsets_v,
        )

        rows, replay_offsets = accepted_row_plan(
            offsets_v,
            accepted_v,
            ops.constant(NUM_DRAFTS, DType.int64, device=gpu),
            Dim(total),
            gpu,
        )
        if replay_full_window:
            rows = ops.range(
                start=0,
                stop=Dim(total),
                out_dim=Dim(total),
                device=gpu,
                dtype=DType.int64,
            )
            replay_offsets = offsets_v.cast(DType.int64)
        replay_state_pools(
            [[GatedDeltaReplayInputs(qkv_v, conv_w, decay_v, beta_v)]],
            [live_conv],
            [live_rec],
            [live_rows],
            [live_rows],
            rows,
            replay_offsets,
            [],
        )

        # The independent reference: the same kernels over the host-sliced
        # accepted prefix, from a pool seeded with the same initial values.
        ref_conv_out = gated_delta_conv1d_fwd(
            qkv_input_ragged=ops.gather(qkv_v, ref_rows_v, axis=0),
            conv_weight=conv_w,
            conv_state=ref_conv,
            slot_idx=slots_v,
            input_row_offsets=ref_offsets_v,
        )
        gated_delta_recurrence_fwd(
            qkv_conv_output=ops.silu(ref_conv_out),
            decay_per_token=ops.gather(decay_v, ref_rows_v, axis=0),
            beta_per_token=ops.gather(beta_v, ref_rows_v, axis=0),
            recurrent_state=ref_rec,
            slot_idx=slots_v,
            input_row_offsets=ref_offsets_v,
        )
        graph.output()

    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = session.load(graph)

    def pool(values: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(values)).to(device)

    buffers = {
        "live_conv": pool(init_conv),
        "live_rec": pool(init_rec),
        "shadow_conv": pool(np.zeros_like(init_conv)),
        "shadow_rec": pool(np.zeros_like(init_rec)),
        "ref_conv": pool(init_conv),
        "ref_rec": pool(init_rec),
    }
    model.execute(
        pool(qkv),
        pool(conv_weight),
        pool(decay),
        pool(beta),
        pool(merged_offsets),
        pool(np.array(accepted, dtype=np.int64)),
        pool(slots),
        pool(ref_rows),
        pool(ref_offsets),
        *buffers.values(),
    )
    return {
        name: np.array(buf.to(CPU()).to_numpy())
        for name, buf in buffers.items()
    }


@pytest.mark.parametrize("accepted", [[0], [3], [NUM_DRAFTS]])
def test_the_block_rollback_lands_on_the_accepted_prefix(
    accepted: list[int],
) -> None:
    """Bit-exact, at every acceptance length a block of 8 can produce."""
    pools = _run(1, accepted)
    np.testing.assert_array_equal(pools["live_conv"], pools["ref_conv"])
    np.testing.assert_array_equal(pools["live_rec"], pools["ref_rec"])


def test_each_request_rolls_back_to_its_own_length() -> None:
    """Three requests accepting 7 / 3 / 0 of their seven drafts."""
    pools = _run(3, [NUM_DRAFTS, 3, 0])
    np.testing.assert_array_equal(pools["live_conv"], pools["ref_conv"])
    np.testing.assert_array_equal(pools["live_rec"], pools["ref_rec"])


def test_leaving_the_pools_advanced_is_detectable() -> None:
    """The ``crop``-is-a-no-op bug, asserted to be visible.

    The shadow pools hold the state after all eight verified rows. If a port
    left the live pools there -- which is exactly what happens when a rollback
    that works for the 16 full-attention layers is applied to the 48
    gated-DeltaNet ones -- the state would differ from the accepted prefix's.
    A partial acceptance that did *not* differ would mean this test measures
    nothing.
    """
    pools = _run(1, [3])
    assert not np.array_equal(pools["shadow_rec"], pools["ref_rec"])
    assert not np.array_equal(pools["shadow_conv"], pools["ref_conv"])


def test_replaying_the_whole_window_is_detectable() -> None:
    """Replaying all eight rows instead of the accepted prefix must not pass.

    The mirror of the test above: it is the replay's *offsets*, not merely the
    fact that a replay happened, that carries the accepted length.
    """
    pools = _run(1, [3], replay_full_window=True)
    assert not np.array_equal(pools["live_rec"], pools["ref_rec"])
