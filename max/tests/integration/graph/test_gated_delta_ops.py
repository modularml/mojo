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
"""Graph-level tests for the slot-indexed Gated DeltaNet (GDN) GPU kernels.

The kernels are exposed via ``ops.inplace_custom`` and mutate a shared
``BufferType`` pool at slot ``slot_idx[batch_item]``. These tests build
minimal Graphs around each op and verify:

* output shape and finiteness on the per-token tensor;
* in-place mutation of the slots named in ``slot_idx``;
* untouched slots elsewhere in the pool (the new invariant the
  slot-indexed design introduces over the old gather/scatter path).

GDN ops are registered in the built-in ``kernels`` package — no
``custom_extensions`` are needed.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

# Single supported kernel size for K (compiled in the `kernels` package).
_KERNEL_SIZE = 4

# Single supported (KD, VD) for the recurrence kernel.
_KEY_HEAD_DIM = 128
_VALUE_HEAD_DIM = 128


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_gated_delta_conv1d_fwd_shapes_and_in_place(
    session: InferenceSession,
) -> None:
    """conv1d slot-indexed pool: shape, finiteness, and slot-isolation.

    Two sequences map to slots 1 and 3 of a 4-slot pool. After execute,
    those two slots must change relative to the initial random fill, and
    slots 0 and 2 must be byte-identical to the initial fill.
    """
    gpu = DeviceRef.GPU()
    batch = 2
    total_seq_len = 8
    conv_dim = 128  # multiple of CONV1D_BLOCK_DIM (128)
    max_slots = 4
    state_len = _KERNEL_SIZE - 1
    referenced_slots = [1, 3]
    untouched_slots = [s for s in range(max_slots) if s not in referenced_slots]
    # Two equal-length sequences: [4, 4] -> exclusive prefix offsets [0, 4, 8].
    offsets_np = np.array([0, 4, total_seq_len], dtype=np.uint32)
    slot_idx_np = np.asarray(referenced_slots, dtype=np.uint32)

    with Graph(
        "gdn_conv1d_slot_indexed_test",
        input_types=[
            TensorType(DType.float32, [total_seq_len, conv_dim], device=gpu),
            TensorType(DType.float32, [conv_dim, _KERNEL_SIZE], device=gpu),
            BufferType(
                DType.float32,
                [max_slots, conv_dim, state_len],
                device=gpu,
            ),
            TensorType(DType.uint32, [batch], device=gpu),
            TensorType(DType.uint32, [batch + 1], device=gpu),
        ],
    ) as graph:
        qkv_in = graph.inputs[0].tensor
        conv_w = graph.inputs[1].tensor
        conv_state = graph.inputs[2].buffer
        slot_idx = graph.inputs[3].tensor
        offsets = graph.inputs[4].tensor

        results = ops.inplace_custom(
            "gated_delta_conv1d_fwd",
            device=gpu,
            values=[qkv_in, conv_w, conv_state, slot_idx, offsets],
            out_types=[
                TensorType(DType.float32, [total_seq_len, conv_dim], device=gpu)
            ],
        )
        graph.output(results[0])

    model = session.load(graph)
    gpu_device = model.input_devices[0]

    rng = np.random.default_rng(42)
    qkv_np = rng.standard_normal((total_seq_len, conv_dim)).astype(np.float32)
    conv_w_np = rng.standard_normal((conv_dim, _KERNEL_SIZE)).astype(np.float32)
    pool_initial_np = rng.standard_normal(
        (max_slots, conv_dim, state_len)
    ).astype(np.float32)

    pool_buf = md.Buffer.from_numpy(pool_initial_np.copy()).to(gpu_device)
    outputs = model.execute(
        md.Buffer.from_numpy(qkv_np).to(gpu_device),
        md.Buffer.from_numpy(conv_w_np).to(gpu_device),
        pool_buf,
        md.Buffer.from_numpy(slot_idx_np).to(gpu_device),
        md.Buffer.from_numpy(offsets_np).to(gpu_device),
    )

    conv_out = torch.from_dlpack(outputs[0])
    assert conv_out.shape == torch.Size([total_seq_len, conv_dim])
    assert torch.all(torch.isfinite(conv_out))

    pool_after_np = torch.from_dlpack(pool_buf).cpu().numpy()
    for s in referenced_slots:
        assert not np.array_equal(pool_after_np[s], pool_initial_np[s]), (
            f"conv_state slot {s} should have been mutated"
        )
    for s in untouched_slots:
        np.testing.assert_array_equal(
            pool_after_np[s],
            pool_initial_np[s],
            err_msg=f"conv_state slot {s} must be untouched",
        )


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_gated_delta_recurrence_fwd_shapes_and_in_place(
    session: InferenceSession,
) -> None:
    """recurrence slot-indexed pool: shape, finiteness, and slot-isolation.

    Mirrors the conv1d test for the recurrence kernel: two sequences map
    to slots 1 and 3 of a 4-slot pool; only those slots may be mutated.
    """
    gpu = DeviceRef.GPU()
    batch = 2
    total_seq_len = 8
    num_value_heads = 2
    num_key_heads = 2
    kd = _KEY_HEAD_DIM
    vd = _VALUE_HEAD_DIM
    value_dim = num_value_heads * vd
    key_dim = num_key_heads * kd
    conv_dim = value_dim + 2 * key_dim
    max_slots = 4
    referenced_slots = [1, 3]
    untouched_slots = [s for s in range(max_slots) if s not in referenced_slots]
    offsets_np = np.array([0, 4, total_seq_len], dtype=np.uint32)
    slot_idx_np = np.asarray(referenced_slots, dtype=np.uint32)

    with Graph(
        "gdn_recurrence_slot_indexed_test",
        input_types=[
            TensorType(DType.float32, [total_seq_len, conv_dim], device=gpu),
            TensorType(
                DType.float32, [total_seq_len, num_value_heads], device=gpu
            ),
            TensorType(
                DType.float32, [total_seq_len, num_value_heads], device=gpu
            ),
            BufferType(
                DType.float32,
                [max_slots, num_value_heads, kd, vd],
                device=gpu,
            ),
            TensorType(DType.uint32, [batch], device=gpu),
            TensorType(DType.uint32, [batch + 1], device=gpu),
        ],
    ) as graph:
        qkv_conv_out = graph.inputs[0].tensor
        decay = graph.inputs[1].tensor
        beta = graph.inputs[2].tensor
        rec_state = graph.inputs[3].buffer
        slot_idx = graph.inputs[4].tensor
        offsets = graph.inputs[5].tensor

        results = ops.inplace_custom(
            "gated_delta_recurrence_fwd",
            device=gpu,
            values=[qkv_conv_out, decay, beta, rec_state, slot_idx, offsets],
            out_types=[
                TensorType(
                    DType.float32, [total_seq_len, value_dim], device=gpu
                )
            ],
        )
        graph.output(results[0])

    model = session.load(graph)
    gpu_device = model.input_devices[0]

    rng = np.random.default_rng(42)
    qkv_np = rng.standard_normal((total_seq_len, conv_dim)).astype(np.float32)
    # decay and beta in (0, 1) for numerical stability.
    decay_np = (
        (np.abs(rng.standard_normal((total_seq_len, num_value_heads))) * 0.5)
        .clip(0.0, 1.0)
        .astype(np.float32)
    )
    beta_np = (
        (np.abs(rng.standard_normal((total_seq_len, num_value_heads))) * 0.5)
        .clip(0.0, 1.0)
        .astype(np.float32)
    )
    pool_initial_np = rng.standard_normal(
        (max_slots, num_value_heads, kd, vd)
    ).astype(np.float32)

    pool_buf = md.Buffer.from_numpy(pool_initial_np.copy()).to(gpu_device)
    outputs = model.execute(
        md.Buffer.from_numpy(qkv_np).to(gpu_device),
        md.Buffer.from_numpy(decay_np).to(gpu_device),
        md.Buffer.from_numpy(beta_np).to(gpu_device),
        pool_buf,
        md.Buffer.from_numpy(slot_idx_np).to(gpu_device),
        md.Buffer.from_numpy(offsets_np).to(gpu_device),
    )

    rec_out = torch.from_dlpack(outputs[0])
    assert rec_out.shape == torch.Size([total_seq_len, value_dim])
    assert torch.all(torch.isfinite(rec_out))

    pool_after_np = torch.from_dlpack(pool_buf).cpu().numpy()
    for s in referenced_slots:
        assert not np.array_equal(pool_after_np[s], pool_initial_np[s]), (
            f"recurrent_state slot {s} should have been mutated"
        )
    for s in untouched_slots:
        np.testing.assert_array_equal(
            pool_after_np[s],
            pool_initial_np[s],
            err_msg=f"recurrent_state slot {s} must be untouched",
        )
