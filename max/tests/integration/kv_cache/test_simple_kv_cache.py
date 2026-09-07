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
"""Tests the paged KV cache inputs that the kernel tests build their batches
from."""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    padded_lut_cols,
)
from test_common.simple_kv_cache import (
    block_ids_for_batch,
    paged_kv_cache_inputs,
)

_PAGE_SIZE = 128


def _kv_params(**overrides: object) -> MHAKVCacheParams:
    kwargs: dict[str, object] = {
        "dtype": DType.float32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "num_layers": 2,
        "page_size": _PAGE_SIZE,
        "devices": [DeviceRef.CPU()],
    }
    kwargs.update(overrides)
    return MHAKVCacheParams(**kwargs)  # type: ignore[arg-type]


def test_block_ids_are_sequential_and_disjoint() -> None:
    assert block_ids_for_batch([1, _PAGE_SIZE, _PAGE_SIZE + 1], _PAGE_SIZE) == [
        [0],
        [1],
        [2, 3],
    ]


def test_lookup_table_maps_each_request_to_its_pages() -> None:
    prompt_lens = [_PAGE_SIZE + 1, 1]
    inputs = paged_kv_cache_inputs(_kv_params(), prompt_lens, total_num_pages=8)

    lut = inputs.lookup_table.to_numpy()
    # Two pages for the first request, one for the second.
    assert lut.shape == (2, padded_lut_cols(2))
    np.testing.assert_array_equal(lut[0, :2], [0, 1])
    np.testing.assert_array_equal(lut[1, :1], [2])

    # Every unassigned column -- the second request's tail and the SIMD padding
    # both rows over-read -- resolves to the null page rather than a real one.
    assert (lut[0, 2:] == 8).all()
    assert (lut[1, 1:] == 8).all()

    # The null page is allocated, so those reads stay in bounds.
    assert inputs.kv_blocks.shape[0] == 9


def test_prefill_lengths() -> None:
    inputs = paged_kv_cache_inputs(_kv_params(), [10, 30])

    np.testing.assert_array_equal(inputs.cache_lengths.to_numpy(), [0, 0])
    assert inputs.max_prompt_length.to_numpy()[0] == 30
    assert inputs.max_cache_length.to_numpy()[0] == 30
    # Defaulted page pool: one page per request plus the null page.
    assert inputs.kv_blocks.shape[0] == 3


def test_decode_lengths_and_page_growth() -> None:
    """A request past its first page keeps the pages it already filled."""
    inputs = paged_kv_cache_inputs(
        _kv_params(), [1], cache_lengths=[_PAGE_SIZE], total_num_pages=8
    )

    np.testing.assert_array_equal(inputs.cache_lengths.to_numpy(), [_PAGE_SIZE])
    assert inputs.max_prompt_length.to_numpy()[0] == 1
    assert inputs.max_cache_length.to_numpy()[0] == _PAGE_SIZE + 1
    np.testing.assert_array_equal(inputs.lookup_table.to_numpy()[0, :2], [0, 1])


def test_blocks_and_scales_match_the_declared_graph_inputs() -> None:
    params = _kv_params(
        dtype=DType.float8_e4m3fn,
        head_dim=128,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=64
        ),
    )
    inputs = paged_kv_cache_inputs(params, [1], total_num_pages=4)
    symbolic = params.get_symbolic_inputs().inputs[0]

    assert inputs.kv_scales is not None
    assert symbolic.kv_scales is not None
    # Page count is the one runtime-only dim; the layout dims must agree with
    # what the graph declares or the kernel reads the wrong strides.
    assert list(inputs.kv_blocks.shape[1:]) == [
        int(dim) for dim in symbolic.kv_blocks.shape.static_dims
    ]
    assert list(inputs.kv_scales.shape[1:]) == [
        int(dim) for dim in symbolic.kv_scales.shape.static_dims
    ]
    assert not inputs.kv_scales.to_numpy().any()


def test_rejects_a_batch_that_outgrows_the_page_pool() -> None:
    with pytest.raises(ValueError, match="needs 2 pages"):
        paged_kv_cache_inputs(_kv_params(), [_PAGE_SIZE + 1], total_num_pages=1)


def test_rejects_mismatched_length_sequences() -> None:
    with pytest.raises(ValueError, match="one entry per request"):
        paged_kv_cache_inputs(_kv_params(), [1, 2], cache_lengths=[0])


def test_rejects_a_multi_device_cache() -> None:
    params = _kv_params(devices=[DeviceRef.CPU(), DeviceRef.CPU(1)])
    with pytest.raises(ValueError, match="single device"):
        paged_kv_cache_inputs(params, [1])
