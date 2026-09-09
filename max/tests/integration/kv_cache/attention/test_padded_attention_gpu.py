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
"""Test pipelines padded attention layer."""

import math
from functools import partial

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, TensorType
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from test_common.modular_graph_test import are_all_tensor_values
from torch.nn.functional import scaled_dot_product_attention


def null_mask_max_flash_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    dtype = torch_dtype_to_max(q.dtype)
    _batch, _q_seq_len, nheads, head_dim = q.shape

    # Graph types.
    q_type = TensorType(
        dtype,
        shape=["batch", "q_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        dtype,
        shape=["batch", "kv_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])

    # Stage ops.

    # Construct and compile the MAX graph flash attention.
    graph = Graph(
        "flash_attn",
        forward=partial(
            flash_attention_gpu,
            scale=math.sqrt(1.0 / head_dim),
            mask_variant=MHAMaskVariant.NULL_MASK,
        ),
        input_types=[
            q_type,
            kv_type,
            kv_type,
        ],
    )

    # Compile model.
    model = session.load(graph)

    # Execute.
    output = model.execute(q.detach(), k.detach(), v.detach())[0]
    assert isinstance(output, Buffer)
    return torch.from_dlpack(output)


@pytest.mark.parametrize(
    "q_seqlen,k_seqlen",
    [
        (20, 20),
        (128, 128),
        # TODO(KERN-1634): support num_keys != seq_len.
        # (2, 3),
    ],
)
def test_null_mask_flash_attention_gpu(q_seqlen: int, k_seqlen: int) -> None:
    head_dim = 128
    batch_size = 1
    nheads = 6
    nheads_k = 6
    torch_device = "cuda"
    torch_dtype = torch.bfloat16

    q_shape = (batch_size, q_seqlen, nheads, head_dim)
    kv_shape = (batch_size, k_seqlen, nheads_k, head_dim)

    q = torch.randn(q_shape, device=torch_device, dtype=torch_dtype)
    k = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)
    v = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)

    out_max = null_mask_max_flash_attn(q, k, v).squeeze()

    out_flash_attn = (
        scaled_dot_product_attention(
            q.to(torch_device).permute(0, 2, 1, 3),
            k.to(torch_device).permute(0, 2, 1, 3),
            v.to(torch_device).permute(0, 2, 1, 3),
            attn_mask=None,
            is_causal=False,
            scale=math.sqrt(1.0 / head_dim),
        )
        .permute(0, 2, 1, 3)
        .squeeze()
    )

    torch.testing.assert_close(out_max, out_flash_attn, rtol=1e-2, atol=2e-2)


def padded_max_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_length: torch.Tensor,
) -> torch.Tensor:
    dtype = torch_dtype_to_max(q.dtype)
    _batch, _q_seq_len, nheads, head_dim = q.shape

    # Graph types.
    q_type = TensorType(
        dtype,
        shape=["batch", "q_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        dtype,
        shape=["batch", "kv_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    valid_length_type = TensorType(
        DType.uint32,
        shape=["batch"],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])

    # Construct and compile the MAX graph flash attention.
    def construct() -> Graph:
        with Graph(
            "padded_flash_attn",
            input_types=[
                q_type,
                kv_type,
                kv_type,
                valid_length_type,
            ],
        ) as g:
            assert are_all_tensor_values(g.inputs)
            q, k, v, valid_length = g.inputs

            result = flash_attention_gpu(
                q,
                k,
                v,
                scale=math.sqrt(1.0 / head_dim),
                mask_variant=MHAMaskVariant.NULL_MASK,
                valid_length=valid_length,
            )
            g.output(result)
        return g

    graph = construct()

    # Compile model.
    model = session.load(graph)

    # Execute.
    output = model.execute(
        q.detach(), k.detach(), v.detach(), valid_length.detach()
    )[0]
    assert isinstance(output, Buffer)
    return torch.from_dlpack(output)


@pytest.mark.parametrize(
    "actual_seq_len, padded_seq_len",
    [
        (128, 160),
        (20, 128),
    ],
)
def test_padded_flash_attention_gpu(
    actual_seq_len: int, padded_seq_len: int
) -> None:
    head_dim = 128
    batch_size = 1
    nheads = 1
    torch_device = "cuda"
    torch_dtype = torch.bfloat16

    actual_shape = (batch_size, actual_seq_len, nheads, head_dim)
    padded_shape = (batch_size, padded_seq_len, nheads, head_dim)

    q_actual = torch.randn(actual_shape, dtype=torch_dtype, device=torch_device)
    k_actual = torch.randn(actual_shape, dtype=torch_dtype, device=torch_device)
    v_actual = torch.randn(actual_shape, dtype=torch_dtype, device=torch_device)

    q_padded = torch.ones(padded_shape, dtype=torch_dtype, device=torch_device)
    k_padded = torch.ones(padded_shape, dtype=torch_dtype, device=torch_device)
    v_padded = torch.ones(padded_shape, dtype=torch_dtype, device=torch_device)

    # Copy actual data to the beginning of padded tensors
    q_padded[:, :actual_seq_len, :, :] = q_actual
    k_padded[:, :actual_seq_len, :, :] = k_actual
    v_padded[:, :actual_seq_len, :, :] = v_actual

    # Run null mask flash attention on actual length data
    out_null_mask = null_mask_max_flash_attn(q_actual, k_actual, v_actual)

    # Run padded flash attention on padded data with valid_length
    valid_length = torch.tensor(
        [actual_seq_len], dtype=torch.uint32, device=torch_device
    )
    out_padded = padded_max_flash_attn(
        q_padded, k_padded, v_padded, valid_length
    )

    # Compare only the first actual_seq_len positions
    out_padded_trimmed = out_padded[:, :actual_seq_len]

    # Assert that the results are close
    torch.testing.assert_close(
        out_null_mask, out_padded_trimmed, rtol=1e-2, atol=2e-2
    )
