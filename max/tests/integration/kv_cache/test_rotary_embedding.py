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

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hypothesis import assume, settings
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, TensorValueLike, ops
from max.nn import (
    DynamicRotaryEmbedding,
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
    LongRoPEScalingParams,
    RotaryEmbedding,
)
from max.nn.kernels import (
    fused_qk_ragged_rope,
    rope_ragged,
    rope_ragged_with_position_ids,
    rope_split_store_ragged,
)
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from test_common.modular_graph_test import (
    are_all_tensor_values,
    modular_graph_test,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs

MAX_SEQ_LEN = 2**14
ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-7


def torch_freqs_cis(dim: int, theta: float):  # noqa: ANN201
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(
        MAX_SEQ_LEN * 2.0, device=freqs.device, dtype=torch.float32
    )
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def torch_llama3_freqs_cis(  # noqa: ANN201
    dim: int,
    theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    orig_max_position: int,
):
    inv_freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    low_freq_wavelen = orig_max_position / low_freq_factor
    high_freq_wavelen = orig_max_position / high_freq_factor

    wave_len = 2 * math.pi / inv_freqs
    if low_freq_factor != high_freq_factor:
        smooth = (orig_max_position / wave_len - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = 0
    freqs = torch.where(
        wave_len < high_freq_wavelen,
        inv_freqs,
        torch.where(
            wave_len > low_freq_wavelen,
            inv_freqs / factor,
            (1 - smooth) * inv_freqs / factor + smooth * inv_freqs,
        ),
    )
    t = torch.arange(
        MAX_SEQ_LEN * 2.0, device=freqs.device, dtype=torch.float32
    )
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def torch_dynamic_rope_freqs_cis(dim: int, theta: float, max_seq_len: int):  # noqa: ANN201
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    t = torch.arange(max_seq_len * 2.0, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


@dataclass
class RopeParams:
    dim: int
    n_heads: int
    theta: float

    @property
    def head_dim(self):  # noqa: ANN201
        return self.dim // self.n_heads


def load_and_execute_numpy(
    session: InferenceSession, graph: Graph
) -> np.ndarray:
    model = session.load(graph)
    results = model.execute()
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, Buffer)
    return result.to_numpy()


@pytest.mark.parametrize(
    "params",
    [RopeParams(dim=512, n_heads=16, theta=5e5)],
)
@pytest.mark.parametrize("dtype", [DType.float32])
def test_freqs_cis(
    session: InferenceSession, dtype: DType, params: RopeParams
) -> None:
    with Graph("freqs_cis", input_types=[]) as graph:
        rope = RotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            MAX_SEQ_LEN,
            head_dim=params.head_dim,
        )
        graph.output(rope.freqs_cis)
    result = load_and_execute_numpy(session, graph)

    # Handle flattened freqs_cis format - reshape back to 3D to extract real/imaginary
    if len(result.shape) == 2:
        d0, d1 = result.shape  # (max_seq_len * 2, head_dim)
        result = result.reshape(
            (d0, d1 // 2, 2)
        )  # (max_seq_len * 2, head_dim // 2, 2)

    # freqs_cis result is stacked along a new dimension - real goes first, then imaginary.
    # The result is a tensor with shape (..., 2) where the last dimension holds [real, imaginary]
    # We extract and convert into a complex tensor type before comparing them.
    result_cis_complex = result[:, :, 0] + 1j * result[:, :, 1]
    expected = torch_freqs_cis(params.head_dim, params.theta)
    np.testing.assert_allclose(
        result_cis_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "base_params",
    [RopeParams(dim=512, n_heads=16, theta=5e5)],
)
@pytest.mark.parametrize(
    "scaling_params",
    [
        Llama3RopeScalingParams(
            factor=4.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=8192,
        ),
    ],
)
@pytest.mark.parametrize("dtype", [DType.float32])
def test_llama3_freqs_cis(
    session: InferenceSession,
    dtype: DType,
    base_params: RopeParams,
    scaling_params: Llama3RopeScalingParams,
) -> None:
    with Graph("freqs_cis", input_types=[]) as graph:
        rope = Llama3RotaryEmbedding(
            base_params.dim,
            base_params.n_heads,
            base_params.theta,
            MAX_SEQ_LEN,
            scaling_params=scaling_params,
            head_dim=base_params.head_dim,
        )
        graph.output(rope.freqs_cis)
    result = load_and_execute_numpy(session, graph)
    d0, d1 = result.shape
    result = result.reshape(d0, d1 // 2, 2)
    # freqs_cis result is stacked along a new dimension - real goes first, then imaginary.
    # The result is a tensor with shape (..., 2) where the last dimension holds [real, imaginary]
    # We extract and convert into a complex tensor type before comparing them.
    result_cis_complex = result[:, :, 0] + 1j * result[:, :, 1]
    expected = torch_llama3_freqs_cis(
        base_params.head_dim,
        base_params.theta,
        scaling_params.factor,
        scaling_params.low_freq_factor,
        scaling_params.high_freq_factor,
        scaling_params.orig_max_position,
    )
    np.testing.assert_allclose(
        result_cis_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "dim, n_heads, theta, short_seq_len, long_seq_len",
    [(512, 16, 5e5, 8192, 16384)],
)
def test_dynamic_rope_freqs_cis(
    session: InferenceSession,
    dim: int,
    n_heads: int,
    theta: float,
    short_seq_len: int,
    long_seq_len: int,
) -> None:
    """Test that DynamicRotaryEmbedding behaves identically to RotaryEmbedding
    for short sequences, and correctly expands the freqs_cis buffer for long
    sequences."""
    head_dim = dim // n_heads

    # Test short sequence: should have the same behavior as default RoPE.
    with Graph("dynamic_rope_short", input_types=[]) as graph:
        rope = DynamicRotaryEmbedding(
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=short_seq_len,
            head_dim=head_dim,
        )
        graph.output(rope.freqs_cis)

    # Manually reshape and recombine the real and imaginary components into a
    # complex-valued array for comparison against the expected result.
    result = load_and_execute_numpy(session, graph)
    d0, d1 = result.shape
    result = result.reshape((d0, d1 // 2, 2))
    result_complex = result[:, :, 0] + 1j * result[:, :, 1]

    expected = torch_dynamic_rope_freqs_cis(head_dim, theta, short_seq_len)
    np.testing.assert_allclose(
        result_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )

    # Test long sequence: should dynamically expand.
    with Graph("dynamic_rope_long", input_types=[]) as graph:
        rope = DynamicRotaryEmbedding(
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=short_seq_len,
            head_dim=head_dim,
        )
        # Simulate runtime position_ids that require growing buffer.
        dummy_position_ids = ops.range(
            0, long_seq_len, 1, dtype=DType.int64, device=DeviceRef.CPU()
        )
        rope.maybe_update_freqs(dummy_position_ids)
        graph.output(rope.freqs_cis)

    # Manually reshape and recombine the real and imaginary components into a
    # complex-valued array for comparison against the expected result.
    result = load_and_execute_numpy(session, graph)
    d0, d1 = result.shape
    result = result.reshape((d0, d1 // 2, 2))
    result_complex = result[:, :, 0] + 1j * result[:, :, 1]

    expected = torch_dynamic_rope_freqs_cis(head_dim, theta, long_seq_len)
    np.testing.assert_allclose(
        result_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


class CannedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, freqs_cis: TensorValueLike) -> None:
        self._freqs_cis = freqs_cis


def torch_rope(
    x: torch.Tensor, freqs_cis: torch.Tensor, cache: torch.Tensor
) -> torch.Tensor:
    start_pos = cache.shape[0]
    seq_len = x.shape[1]
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
    freqs_cis = torch.view_as_complex(freqs_cis.reshape(seq_len, -1, 2))
    return apply_rotary_emb(x, freqs_cis)


def _torch_rope_ragged(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    input_row_offsets: np.ndarray,
    start_pos: np.ndarray,
) -> torch.Tensor:
    expected = torch.empty_like(x)
    for batch_idx in range(len(start_pos)):
        row_start = int(input_row_offsets[batch_idx])
        row_end = int(input_row_offsets[batch_idx + 1])
        seq_len = row_end - row_start
        if seq_len == 0:
            continue
        freqs_slice = freqs_cis[
            int(start_pos[batch_idx]) : int(start_pos[batch_idx]) + seq_len
        ]
        freqs_complex = torch.view_as_complex(
            freqs_slice.reshape(seq_len, -1, 2)
        )
        segment = x[row_start:row_end].unsqueeze(0)
        expected[row_start:row_end] = apply_rotary_emb(
            segment, freqs_complex
        ).squeeze(0)
    return expected


def _reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, x_)
    return torch.view_as_real(x_ * freqs_cis).flatten(3).type_as(x)


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(
            DType.float32,
            ["batch", "seqlen", "n_kv_heads", 32],
            device=DeviceRef.CPU(),
        )
    ],
)
@pytest.mark.parametrize("start_pos", [0, 15])
def test_rope(
    session: InferenceSession, input_type: TensorType, start_pos: Dim
) -> None:
    _, _seqlen, _, head_dim = input_type.shape
    freqs_cis_type = TensorType(
        input_type.dtype, [MAX_SEQ_LEN, head_dim], device=DeviceRef.CPU()
    )
    cachelike = TensorType(DType.int64, [start_pos], device=DeviceRef.CPU())
    with Graph(
        "rope", input_types=[input_type, freqs_cis_type, cachelike]
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, freqs_cis, cache = graph.inputs
        freqs_cis = freqs_cis.reshape((MAX_SEQ_LEN, -1, 2))  # as complex
        start_pos = cache.shape[0]
        seq_len = x.shape[1]
        rope = CannedRotaryEmbedding(freqs_cis)
        graph.output(rope(x, start_pos, seq_len))

    @modular_graph_test(session, graph, max_magnitude=1.0)
    @settings(max_examples=10)
    def test_correctness(
        execute: Callable[[Sequence[Buffer]], Buffer],
        inputs: Sequence[Buffer],
        torch_inputs: Sequence[torch.Tensor],
    ) -> None:
        x, _freqs_cis, cache = inputs
        start_pos = cache.shape[0]
        seq_len = x.shape[1]
        assume(start_pos + seq_len < MAX_SEQ_LEN)
        result = execute(inputs).to_numpy()
        expected = torch_rope(*torch_inputs).detach().numpy()

        np.testing.assert_allclose(
            result,
            expected,
            atol=ACCURACY_ATOL,
            rtol=ACCURACY_RTOL,
            equal_nan=True,
        )


def test_rope_ragged(session: InferenceSession) -> None:
    max_seq_len = 32
    n_heads = 2
    head_dim = 8
    prompt_lens = [3, 5]
    total_seq_len = sum(prompt_lens)

    input_row_offsets = np.array([0, 3, total_seq_len], dtype=np.uint32)
    start_pos = np.array([0, 4], dtype=np.uint32)

    torch.manual_seed(0)
    input_data = torch.randn(
        total_seq_len, n_heads, head_dim, dtype=torch.float32
    )
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = torch.view_as_real(freqs_complex).reshape(max_seq_len, head_dim)

    input_type = TensorType(
        DType.float32,
        [total_seq_len, n_heads, head_dim],
        device=DeviceRef.CPU(),
    )
    offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], device=DeviceRef.CPU()
    )
    start_pos_type = TensorType(
        DType.uint32, ["start_pos_len"], device=DeviceRef.CPU()
    )
    freqs_type = TensorType(
        DType.float32, [max_seq_len, head_dim], device=DeviceRef.CPU()
    )

    with Graph(
        "rope_ragged",
        input_types=[input_type, offsets_type, start_pos_type, freqs_type],
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        inp, offsets, starts, freqs_in = graph.inputs
        graph.output(
            rope_ragged(
                inp.tensor,
                offsets.tensor,
                starts.tensor,
                freqs_in.tensor,
            )
        )

    model = session.load(graph)
    result = model(
        input_data.numpy(),
        input_row_offsets,
        start_pos,
        freqs_cis.numpy(),
    )[0].to_numpy()

    expected = _torch_rope_ragged(
        input_data, freqs_cis, input_row_offsets, start_pos
    ).numpy()
    np.testing.assert_allclose(
        result,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


def test_rope_ragged_with_position_ids(session: InferenceSession) -> None:
    max_seq_len = 32
    n_heads = 2
    head_dim = 8
    total_seq_len = 6

    torch.manual_seed(1)
    input_data = torch.randn(
        total_seq_len, n_heads, head_dim, dtype=torch.float32
    )
    position_ids = np.array([0, 2, 1, 3, 7, 4], dtype=np.uint32)

    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = torch.view_as_real(freqs_complex).reshape(max_seq_len, head_dim)

    input_type = TensorType(
        DType.float32,
        [total_seq_len, n_heads, head_dim],
        device=DeviceRef.CPU(),
    )
    freqs_type = TensorType(
        DType.float32, [max_seq_len, head_dim], device=DeviceRef.CPU()
    )
    position_ids_type = TensorType(
        DType.uint32, [total_seq_len], device=DeviceRef.CPU()
    )

    with Graph(
        "rope_ragged_with_position_ids",
        input_types=[input_type, freqs_type, position_ids_type],
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        inp, freqs_in, pos_ids = graph.inputs
        graph.output(
            rope_ragged_with_position_ids(
                inp.tensor,
                freqs_in.tensor,
                pos_ids.tensor,
            )
        )

    model = session.load(graph)
    result = model(
        input_data.numpy(),
        freqs_cis.numpy(),
        position_ids,
    )[0].to_numpy()

    freqs_gathered = freqs_cis[position_ids]
    freqs_gathered_complex = torch.view_as_complex(
        freqs_gathered.reshape(total_seq_len, -1, 2)
    )
    expected = apply_rotary_emb(
        input_data.unsqueeze(0), freqs_gathered_complex
    ).squeeze(0)
    np.testing.assert_allclose(
        result,
        expected.numpy(),
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )


@pytest.mark.parametrize("use_position_ids", [False, True])
def test_kv_cache_ragged_rope(
    session: InferenceSession, use_position_ids: bool
) -> None:
    num_q_heads = 32
    head_dim = 128
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=head_dim,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    input_type = TensorType(
        DType.float32,
        ["total_seq_len", num_q_heads, head_dim],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], device=DeviceRef.CPU()
    )
    freqs_cis_type = TensorType(
        DType.float32,
        [MAX_SEQ_LEN, head_dim],
        device=DeviceRef.CPU(),
    )

    num_sections = 3
    position_ids_type = None
    if use_position_ids:
        position_ids_type = TensorType(
            DType.uint32,
            [num_sections, "total_seq_len"],
            device=DeviceRef.CPU(),
        )

    mrope_section = [16, 24, 24]

    def construct() -> Graph:
        input_types = [
            input_type,
            input_row_offsets_type,
            freqs_cis_type,
            *kv_params.flattened_kv_inputs(),
        ]

        if use_position_ids:
            assert position_ids_type is not None
            input_types.insert(3, position_ids_type)

        graph_name = (
            "call_ragged_qk_rope_with_position_ids"
            if use_position_ids
            else "call_ragged_qk_rope"
        )

        with Graph(graph_name, input_types=input_types) as g:
            inp = g.inputs[0]
            input_row_offsets = g.inputs[1]
            freqs_cis = g.inputs[2]

            kv_start = 4 if use_position_ids else 3
            (
                blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                _attention_dispatch_metadata,
            ) = g.inputs[kv_start:]

            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
            )

            position_ids = g.inputs[3].tensor if use_position_ids else None

            result = fused_qk_ragged_rope(
                kv_params,
                inp.tensor,
                input_row_offsets.tensor,
                kv_collection,
                freqs_cis.tensor,
                layer_idx,
                position_ids=position_ids,
                mrope_section=mrope_section if use_position_ids else None,
            )
            g.output(result)
        return g

    g = construct()

    input_row_offsets = Buffer(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=8
    )

    # Build provided_inputs with correct indices based on use_position_ids
    offset = 1 if use_position_ids else 0
    assert kv_runtime_inputs.attention_dispatch_metadata is not None
    provided_inputs = {
        1: input_row_offsets,
        3 + offset: kv_runtime_inputs.kv_blocks,
        4 + offset: kv_runtime_inputs.cache_lengths,
        5 + offset: kv_runtime_inputs.lookup_table,
        6 + offset: kv_runtime_inputs.max_prompt_length,
        7 + offset: kv_runtime_inputs.max_cache_length,
        8 + offset: kv_runtime_inputs.attention_dispatch_metadata,
    }

    if use_position_ids:
        position_ids_data = np.tile(
            np.arange(total_seq_len, dtype=np.uint32), (num_sections, 1)
        )
        provided_inputs[3] = Buffer.from_numpy(position_ids_data)

    @modular_graph_test(
        session,
        g,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offsets_len": len(prompt_lens) + 1,
        },
        provided_inputs=provided_inputs,
    )
    @settings(max_examples=10)
    def test_runs_without_nan(
        execute: Callable[[Sequence[Buffer]], Buffer],
        inputs: Sequence[Buffer],
        torch_inputs: Sequence[torch.Tensor],
    ) -> None:
        result = execute(list(inputs)).to_numpy()
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


@pytest.mark.parametrize("use_position_ids", [False, True])
def test_rope_split_store_ragged(
    session: InferenceSession, use_position_ids: bool
) -> None:
    """Tests rope_split_store_ragged compiles and produces valid output."""
    num_q_heads = 32
    head_dim = 128
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=head_dim,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    combined_dim = (num_q_heads + 2 * kv_params.n_kv_heads) * head_dim

    qkv_type = TensorType(
        DType.float32,
        ["total_seq_len", combined_dim],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], device=DeviceRef.CPU()
    )
    freqs_cis_type = TensorType(
        DType.float32,
        [MAX_SEQ_LEN, head_dim],
        device=DeviceRef.CPU(),
    )

    num_sections = 3
    mrope_section = [16, 24, 24]
    position_ids_type = None
    if use_position_ids:
        position_ids_type = TensorType(
            DType.uint32,
            [num_sections, "total_seq_len"],
            device=DeviceRef.CPU(),
        )

    def construct() -> Graph:
        input_types = [
            qkv_type,
            input_row_offsets_type,
            freqs_cis_type,
            *kv_params.flattened_kv_inputs(),
        ]

        if use_position_ids:
            assert position_ids_type is not None
            input_types.insert(3, position_ids_type)

        graph_name = (
            "rope_split_store_with_position_ids"
            if use_position_ids
            else "rope_split_store"
        )

        with Graph(graph_name, input_types=input_types) as g:
            qkv = g.inputs[0]
            input_row_offsets = g.inputs[1]
            freqs_cis = g.inputs[2]

            kv_start = 4 if use_position_ids else 3
            (
                blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                _attention_dispatch_metadata,
            ) = g.inputs[kv_start:]

            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
            )

            position_ids = g.inputs[3].tensor if use_position_ids else None

            result = rope_split_store_ragged(
                kv_params,
                qkv=qkv.tensor,
                input_row_offsets=input_row_offsets.tensor,
                freqs_cis=freqs_cis.tensor,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=num_q_heads,
                position_ids=position_ids,
                mrope_section=mrope_section if use_position_ids else None,
            )
            g.output(result)
        return g

    g = construct()

    input_row_offsets = Buffer(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=8
    )

    offset = 1 if use_position_ids else 0
    assert kv_runtime_inputs.attention_dispatch_metadata is not None
    provided_inputs = {
        1: input_row_offsets,
        3 + offset: kv_runtime_inputs.kv_blocks,
        4 + offset: kv_runtime_inputs.cache_lengths,
        5 + offset: kv_runtime_inputs.lookup_table,
        6 + offset: kv_runtime_inputs.max_prompt_length,
        7 + offset: kv_runtime_inputs.max_cache_length,
        8 + offset: kv_runtime_inputs.attention_dispatch_metadata,
    }

    if use_position_ids:
        position_ids_data = np.tile(
            np.arange(total_seq_len, dtype=np.uint32), (num_sections, 1)
        )
        provided_inputs[3] = Buffer.from_numpy(position_ids_data)

    @modular_graph_test(
        session,
        g,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offsets_len": len(prompt_lens) + 1,
        },
        provided_inputs=provided_inputs,
    )
    @settings(max_examples=10)
    def test_runs_without_nan(
        execute: Callable[[Sequence[Buffer]], Buffer],
        inputs: Sequence[Buffer],
        torch_inputs: Sequence[torch.Tensor],
    ) -> None:
        result = execute(list(inputs)).to_numpy()
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


def torch_longrope_freqs_cis(
    dim: int,
    theta: float,
    max_seq_len: int,
    short_factor: list[float],
    long_factor: list[float],
    original_max_position: int,
) -> torch.Tensor:
    """PyTorch reference implementation of LongRoPE frequency computation with stitched table."""
    # Compute base inverse frequencies
    inv_freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )

    # Apply short scaling factors
    short_factors_tensor = torch.tensor(
        short_factor[: len(inv_freqs)], dtype=torch.float32
    )
    scaled_inv_freqs_short = inv_freqs / short_factors_tensor

    # Apply long scaling factors
    long_factors_tensor = torch.tensor(
        long_factor[: len(inv_freqs)], dtype=torch.float32
    )
    scaled_inv_freqs_long = inv_freqs / long_factors_tensor

    # Generate position ids for the "short" part (0 to original_max_position)
    t_short = torch.arange(original_max_position, dtype=torch.float32)

    # Generate position ids for the "long" part (original_max_position to max_seq_len*2)
    t_long = torch.arange(
        original_max_position, max_seq_len * 2.0, dtype=torch.float32
    )

    # Compute frequencies for both parts
    freqs_short = torch.outer(t_short, scaled_inv_freqs_short)
    freqs_long = torch.outer(t_long, scaled_inv_freqs_long)

    # Concatenate the two parts
    freqs_combined = torch.cat([freqs_short, freqs_long], dim=0)

    # Convert to complex
    freqs_cis = torch.polar(torch.ones_like(freqs_combined), freqs_combined)
    return freqs_cis


@pytest.mark.skip(reason="SERVSYS-1192")
@pytest.mark.parametrize(
    "params",
    [RopeParams(dim=3072, n_heads=32, theta=10000.0)],
)
@pytest.mark.parametrize("dtype", [DType.float32])
def test_longrope_scaling(
    session: InferenceSession, dtype: DType, params: RopeParams
) -> None:
    """Test LongRoPE frequency scaling with different scaling factors for short and long sequences.

    This test verifies that LongRoPE correctly applies frequency scaling parameters:
    - short_factor: scaling factors for shorter sequences (default 1.0)
    - long_factor: scaling factors for longer sequences (2x the short factors)
    - Ensures proper shape and numerical stability of the frequency embeddings
    - Validates numerical correctness against PyTorch reference implementation
    """
    max_seq_len = 32768
    original_max_position = 4096

    assert max_seq_len > original_max_position, (
        "this ensures the long scaling factor is actually being computed"
    )

    # Create scaling params with long factors being 2x short factors
    scaling_params = LongRoPEScalingParams(
        short_factor=[1.0] * (params.head_dim // 2),
        long_factor=[2.0] * (params.head_dim // 2),
        original_max_position=original_max_position,
        max_position_embeddings=max_seq_len,
    )

    with Graph("longrope_freqs_cis", input_types=[]) as graph:
        rope = LongRoPERotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            max_seq_len,
            head_dim=params.head_dim,
            scaling_params=scaling_params,
        )
        graph.output(rope.freqs_cis)

    result = load_and_execute_numpy(session, graph)

    # Basic shape and validity checks - enforce flattened 2D shape
    assert len(result.shape) == 2, (
        f"Expected 2D tensor, but got shape {result.shape}"
    )
    assert result.shape[0] == max_seq_len * 2
    assert result.shape[1] == params.head_dim
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Numerical validation against PyTorch reference
    # Reshape the validated 2D tensor to extract real/imaginary parts
    d0, d1 = result.shape  # (max_seq_len * 2, head_dim)
    result = result.reshape(
        (d0, d1 // 2, 2)
    )  # (max_seq_len * 2, head_dim // 2, 2)

    result_cis_complex = result[:, :, 0] + 1j * result[:, :, 1]
    expected = torch_longrope_freqs_cis(
        params.head_dim,
        params.theta,
        max_seq_len,
        scaling_params.short_factor,
        scaling_params.long_factor,
        scaling_params.original_max_position,
    )

    np.testing.assert_allclose(
        result_cis_complex,
        expected,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )

    # Test short sequence behavior (should use short_factor)
    short_max_seq_len = original_max_position // 2

    with Graph("longrope_short_seq", input_types=[]) as graph:
        rope_short = LongRoPERotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            short_max_seq_len,
            head_dim=params.head_dim,
            scaling_params=scaling_params,
        )
        graph.output(rope_short.freqs_cis)

    result_short = load_and_execute_numpy(session, graph)

    # Validate short sequence uses short_factor
    if len(result_short.shape) == 2:
        d0, d1 = result_short.shape
        result_short = result_short.reshape((d0, d1 // 2, 2))

    result_short_complex = result_short[:, :, 0] + 1j * result_short[:, :, 1]
    expected_short = torch_longrope_freqs_cis(
        params.head_dim,
        params.theta,
        short_max_seq_len,
        scaling_params.short_factor,
        scaling_params.long_factor,
        scaling_params.original_max_position,
    )

    np.testing.assert_allclose(
        result_short_complex,
        expected_short,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )

    # Test without scaling (should behave like standard RoPE)
    with Graph("longrope_no_scaling", input_types=[]) as graph:
        rope_no_scale = LongRoPERotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            4096,  # smaller max_seq_len
            head_dim=params.head_dim,
            scaling_params=None,  # No scaling
        )
        graph.output(rope_no_scale.freqs_cis)

    result_no_scale = load_and_execute_numpy(session, graph)

    # Should behave like standard RoPE when no scaling params
    assert result_no_scale.shape[0] == 4096 * 2
    assert result_no_scale.shape[1] == params.head_dim

    # Compare with standard RoPE for validation
    with Graph("standard_rope", input_types=[]) as graph:
        rope_standard = RotaryEmbedding(
            params.dim,
            params.n_heads,
            params.theta,
            4096,
            head_dim=params.head_dim,
        )
        graph.output(rope_standard.freqs_cis)

    result_standard = load_and_execute_numpy(session, graph)

    # LongRoPE without scaling should match standard RoPE
    np.testing.assert_allclose(
        result_no_scale,
        result_standard,
        atol=ACCURACY_ATOL,
        rtol=ACCURACY_RTOL,
        equal_nan=True,
    )
