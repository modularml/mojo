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
"""Tests for the fixed-capacity behavior of ModelInputCachingAllocator."""

from __future__ import annotations

import logging

import pytest
from max.driver import CPU
from max.dtype import DType
from max.pipelines.lib.interfaces.batch_processor import (
    ModelInputCachingAllocator,
)

# Small enough that unit tests don't allocate the production 8 MiB backing.
TEST_CAPACITY_BYTES = 4096


@pytest.fixture
def small_capacity(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(
        ModelInputCachingAllocator,
        "FIXED_CAPACITY_BYTES",
        TEST_CAPACITY_BYTES,
    )
    return TEST_CAPACITY_BYTES


def test_different_shapes_share_one_backing(small_capacity: int) -> None:
    """Ragged shapes under one name alias a single owning allocation."""
    allocator = ModelInputCachingAllocator()
    device = CPU()

    first = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.int64,
        shape=(8,),
        device=device,
    )
    second = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.int64,
        shape=(64,),
        device=device,
    )
    first_again = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.int64,
        shape=(8,),
        device=device,
    )

    assert first._data_ptr() == second._data_ptr()
    assert first_again is first
    assert first.shape == (8,)
    assert first.dtype == DType.int64
    assert second.shape == (64,)
    assert second.dtype == DType.int64


def test_different_names_do_not_alias(small_capacity: int) -> None:
    allocator = ModelInputCachingAllocator()
    device = CPU()

    tokens = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.int64,
        shape=(8,),
        device=device,
    )
    row_offsets = allocator.alloc(
        name="ragged_input_row_offsets",
        dtype=DType.uint32,
        shape=(9,),
        device=device,
    )

    assert tokens._data_ptr() != row_offsets._data_ptr()


def test_same_name_different_dtype_reuses_raw_backing(
    small_capacity: int,
) -> None:
    """The backing is raw bytes, so a dtype change reuses the allocation."""
    allocator = ModelInputCachingAllocator()
    device = CPU()

    as_int64 = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.int64,
        shape=(4,),
        device=device,
    )
    as_uint32 = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.uint32,
        shape=(4,),
        device=device,
    )

    assert as_int64._data_ptr() == as_uint32._data_ptr()
    assert as_int64.dtype == DType.int64
    assert as_uint32.dtype == DType.uint32
    assert as_uint32.shape == (4,)


def test_exact_capacity_boundary_uses_fixed_backing(
    small_capacity: int,
) -> None:
    allocator = ModelInputCachingAllocator()
    device = CPU()

    exact = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.uint8,
        shape=(small_capacity,),
        device=device,
    )
    tiny = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.uint8,
        shape=(1,),
        device=device,
    )

    assert exact._data_ptr() == tiny._data_ptr()
    assert exact.shape == (small_capacity,)


def test_zero_sized_shape_uses_fixed_backing(small_capacity: int) -> None:
    allocator = ModelInputCachingAllocator()
    device = CPU()

    empty = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.int64,
        shape=(0,),
        device=device,
    )
    nonempty = allocator.alloc(
        name="ragged_input_tokens",
        dtype=DType.int64,
        shape=(4,),
        device=device,
    )

    assert empty.shape == (0,)
    assert empty._data_ptr() == nonempty._data_ptr()


def test_oversized_warns_once_and_reuses_exact_key(
    small_capacity: int, caplog: pytest.LogCaptureFixture
) -> None:
    allocator = ModelInputCachingAllocator()
    device = CPU()
    oversized_shape = (small_capacity + 1,)

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        first = allocator.alloc(
            name="ragged_input_tokens",
            dtype=DType.uint8,
            shape=oversized_shape,
            device=device,
        )
        warnings_after_alloc = [
            rec for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        second = allocator.alloc(
            name="ragged_input_tokens",
            dtype=DType.uint8,
            shape=oversized_shape,
            device=device,
        )

    assert len(warnings_after_alloc) == 1
    message = warnings_after_alloc[0].getMessage()
    assert "ragged_input_tokens" in message
    assert str(small_capacity + 1) in message
    assert str(small_capacity) in message
    assert str(oversized_shape) in message
    assert str(DType.uint8) in message
    assert str(device) in message

    # Reuse returns the same cached owning buffer without a second warning.
    assert second is first
    assert [
        rec for rec in caplog.records if rec.levelno == logging.WARNING
    ] == warnings_after_alloc


def test_oversized_shapes_allocate_separately_and_each_warns(
    small_capacity: int, caplog: pytest.LogCaptureFixture
) -> None:
    allocator = ModelInputCachingAllocator()
    device = CPU()

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        first = allocator.alloc(
            name="ragged_input_tokens",
            dtype=DType.uint8,
            shape=(small_capacity + 1,),
            device=device,
        )
        second = allocator.alloc(
            name="ragged_input_tokens",
            dtype=DType.uint8,
            shape=(small_capacity + 2,),
            device=device,
        )

    assert first._data_ptr() != second._data_ptr()
    assert first.shape == (small_capacity + 1,)
    assert second.shape == (small_capacity + 2,)
    assert (
        len([rec for rec in caplog.records if rec.levelno == logging.WARNING])
        == 2
    )
