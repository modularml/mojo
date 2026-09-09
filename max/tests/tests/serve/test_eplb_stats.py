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
"""Unit tests for EP profiling schema (metadata only).

These tests are pure-Python, no GPU, no serve, no ZMQ.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from max.pipelines.lib.eplb_stats import (
    EplbStatsAccumulator,
    EplbStatsMetadata,
    EplbStatsSnapshot,
)


class TestEplbStatsMetadata:
    def test_construct_valid(self) -> None:
        md = EplbStatsMetadata(
            num_layers=60,
            num_moe_layers=60,
            moe_layer_indices=tuple(range(60)),
            num_logical_experts=384,
            num_experts_per_token=8,
        )
        assert md.num_moe_layers == 60
        assert md.num_logical_experts == 384
        assert md.num_experts_per_token == 8

    @pytest.mark.parametrize(
        ("layers", "experts", "topk"),
        [
            (0, 384, 8),
            (-1, 384, 8),
            (60, 0, 8),
            (60, -1, 8),
            (60, 384, 0),
            (60, 384, -1),
            (60, 8, 9),  # topk > experts
        ],
    )
    def test_invalid_shape_rejected(
        self, layers: int, experts: int, topk: int
    ) -> None:
        with pytest.raises(AssertionError):
            EplbStatsMetadata(
                num_layers=layers,
                num_moe_layers=layers,
                moe_layer_indices=tuple(range(max(layers, 0))),
                num_logical_experts=experts,
                num_experts_per_token=topk,
            )

    def test_to_dict_round_trip(self) -> None:
        original = EplbStatsMetadata(
            num_layers=60,
            num_moe_layers=60,
            moe_layer_indices=tuple(range(60)),
            num_logical_experts=384,
            num_experts_per_token=8,
        )
        restored = EplbStatsMetadata.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_missing_field(self) -> None:
        with pytest.raises(ValueError, match="Missing required field"):
            EplbStatsMetadata.from_dict(
                {"num_moe_layers": 1, "num_logical_experts": 2}
            )

    def test_from_dict_wrong_type(self) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            EplbStatsMetadata.from_dict(
                {
                    "num_moe_layers": 1,
                    "num_logical_experts": "2",
                    "num_experts_per_token": 1,
                }
            )

    def test_frozen(self) -> None:
        md = EplbStatsMetadata(
            num_layers=1,
            num_moe_layers=1,
            moe_layer_indices=(0,),
            num_logical_experts=2,
            num_experts_per_token=1,
        )
        with pytest.raises(Exception):
            md.num_moe_layers = 2  # type: ignore[misc]


def _md(layers: int = 2, experts: int = 4, topk: int = 2) -> EplbStatsMetadata:
    return EplbStatsMetadata(
        num_layers=layers,
        num_moe_layers=layers,
        moe_layer_indices=tuple(range(layers)),
        num_logical_experts=experts,
        num_experts_per_token=topk,
    )


class TestEplbStatsSnapshot:
    def test_from_array_copies(self) -> None:
        metadata = _md()
        hist = np.zeros(
            (metadata.num_moe_layers, metadata.num_logical_experts),
            dtype=np.int64,
        )
        snap = EplbStatsSnapshot.from_array(metadata, hist, total_tokens=0)
        # Mutating the source must NOT affect the snapshot.
        hist[0, 0] = 999
        assert snap.histogram[0, 0] == 0

    def test_histogram_is_read_only(self) -> None:
        metadata = _md()
        hist = np.ones(
            (metadata.num_moe_layers, metadata.num_logical_experts),
            dtype=np.int64,
        )
        snap = EplbStatsSnapshot.from_array(metadata, hist)
        with pytest.raises(ValueError):
            snap.histogram[0, 0] = 42

    def test_shape_mismatch_rejected(self) -> None:
        metadata = _md(layers=2, experts=4)
        bad = np.zeros((2, 5), dtype=np.int64)
        with pytest.raises(AssertionError, match="shape mismatch"):
            EplbStatsSnapshot.from_array(metadata, bad)

    def test_wrong_dtype_rejected(self) -> None:
        metadata = _md()
        bad = np.zeros(
            (metadata.num_moe_layers, metadata.num_logical_experts),
            dtype=np.int32,
        )
        with pytest.raises(AssertionError, match="dtype must be int64"):
            EplbStatsSnapshot(
                metadata=metadata,
                histogram=bad,  # type: ignore[arg-type]
            )

    def test_not_ndarray_rejected(self) -> None:
        metadata = _md()
        with pytest.raises(AssertionError, match="ndarray"):
            EplbStatsSnapshot(
                metadata=metadata,
                histogram=[[0, 0, 0, 0], [0, 0, 0, 0]],  # type: ignore[arg-type]
            )

    def test_negative_total_tokens_rejected(self) -> None:
        metadata = _md()
        hist = np.zeros(
            (metadata.num_moe_layers, metadata.num_logical_experts),
            dtype=np.int64,
        )
        with pytest.raises(AssertionError, match="total_tokens"):
            EplbStatsSnapshot.from_array(metadata, hist, total_tokens=-1)

    def test_realistic_shape(self) -> None:
        metadata = EplbStatsMetadata(
            num_layers=60,
            num_moe_layers=60,
            moe_layer_indices=tuple(range(60)),
            num_logical_experts=384,
            num_experts_per_token=8,
        )
        hist = np.zeros((60, 384), dtype=np.int64)
        hist[0, 0] = 10
        hist[59, 383] = 7
        snap = EplbStatsSnapshot.from_array(metadata, hist, total_tokens=17)
        assert snap.histogram.shape == (60, 384)
        assert snap.histogram[0, 0] == 10
        assert snap.histogram[59, 383] == 7
        assert snap.total_tokens == 17


class TestEplbStatsSnapshotJson:
    @staticmethod
    def _populated() -> EplbStatsSnapshot:
        metadata = EplbStatsMetadata(
            num_layers=2,
            num_moe_layers=2,
            moe_layer_indices=(0, 1),
            num_logical_experts=4,
            num_experts_per_token=2,
        )
        hist = np.array(
            [[10, 0, 5, 3], [2, 7, 1, 0]],
            dtype=np.int64,
        )
        return EplbStatsSnapshot.from_array(
            metadata,
            hist,
            total_tokens=9,
        )

    def test_to_dict_round_trip(self) -> None:
        original = self._populated()
        restored = EplbStatsSnapshot.from_dict(original.to_dict())
        assert restored.metadata == original.metadata
        np.testing.assert_array_equal(restored.histogram, original.histogram)
        assert restored.histogram.dtype == np.int64
        assert restored.total_tokens == original.total_tokens

    def test_round_trip_preserves_read_only(self) -> None:
        restored = EplbStatsSnapshot.from_dict(self._populated().to_dict())
        with pytest.raises(ValueError):
            restored.histogram[0, 0] = 0

    @pytest.mark.parametrize(
        "missing_field",
        ["metadata", "histogram", "non_draft_tokens"],
    )
    def test_from_dict_missing_required(self, missing_field: str) -> None:
        data = self._populated().to_dict()
        del data[missing_field]
        with pytest.raises(ValueError, match="Missing required field"):
            EplbStatsSnapshot.from_dict(data)

    def test_from_dict_histogram_wrong_shape(self) -> None:
        data = self._populated().to_dict()
        # metadata says 2x4, but we send 2x3.
        data["histogram"] = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(AssertionError, match="shape mismatch"):
            EplbStatsSnapshot.from_dict(data)

    def test_from_dict_histogram_non_numeric(self) -> None:
        data = self._populated().to_dict()
        data["histogram"] = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
        with pytest.raises(ValueError, match="int64 array"):
            EplbStatsSnapshot.from_dict(data)

    def test_from_dict_total_tokens_wrong_type(self) -> None:
        data = self._populated().to_dict()
        data["non_draft_tokens"] = "9"
        with pytest.raises(ValueError, match="non_draft_tokens must be an int"):
            EplbStatsSnapshot.from_dict(data)


class TestEplbStatsAccumulator:
    """Unit tests use devices=[] so no GPU buffers are allocated."""

    def test_initial_state_is_zero(self) -> None:
        acc = EplbStatsAccumulator(_md(), devices=[])
        snap = acc.snapshot()
        assert snap.histogram.sum() == 0
        assert snap.total_tokens == 0
        assert snap.metadata == _md()

    def test_metadata_property(self) -> None:
        md = _md(layers=3, experts=8, topk=2)
        acc = EplbStatsAccumulator(md, devices=[])
        assert acc.metadata == md

    def test_device_buffers_empty_when_no_devices(self) -> None:
        acc = EplbStatsAccumulator(_md(), devices=[])
        assert acc.device_buffers == []

    def test_record_batch_total_tokens_accumulates(self) -> None:
        acc = EplbStatsAccumulator(_md(), devices=[])
        acc.record_batch_total_tokens(17)
        acc.record_batch_total_tokens(8)
        assert acc.snapshot().total_tokens == 25

    def test_snapshot_is_independent_of_subsequent_updates(self) -> None:
        acc = EplbStatsAccumulator(_md(), devices=[])
        acc.record_batch_total_tokens(5)
        snap = acc.snapshot()
        acc.record_batch_total_tokens(10)
        assert snap.total_tokens == 5

    def test_snapshot_histogram_is_read_only(self) -> None:
        acc = EplbStatsAccumulator(_md(), devices=[])
        snap = acc.snapshot()
        with pytest.raises(ValueError):
            snap.histogram[0, 0] = 1

    def test_snapshot_histogram_shape_matches_metadata(self) -> None:
        md = _md(layers=3, experts=8)
        acc = EplbStatsAccumulator(md, devices=[])
        snap = acc.snapshot()
        assert snap.histogram.shape == (3, 8)
        assert snap.histogram.dtype == np.int64

    def test_reset_zeros_total_tokens(self) -> None:
        acc = EplbStatsAccumulator(_md(), devices=[])
        acc.record_batch_total_tokens(100)
        acc.reset()
        assert acc.snapshot().total_tokens == 0

    def test_concurrent_record_batch_total_tokens_is_atomic(self) -> None:
        # Without the lock around _total_tokens, concurrent += would
        # lose increments. With the lock the final count is exact.
        acc = EplbStatsAccumulator(_md(), devices=[])
        n_per_thread = 1000

        def worker() -> None:
            for _ in range(n_per_thread):
                acc.record_batch_total_tokens(1)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert acc.snapshot().total_tokens == 4 * n_per_thread
