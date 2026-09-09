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

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAAttnKey,
)
from max.nn.kv_cache.utils import MultiAttnKey
from max.pipelines.lib.graph_capture import ServeGraphCaptureRunner
from test_common.mocks.pipeline_model import MockModelInputs

MiB = 1024 * 1024


def _gk(batch_size: int) -> MultiAttnKey:
    """Builds the non-spec capture key the runner produces for ``batch_size``.

    The mock resolver returns ``MHAAttnKey(batch_size, q=1, num_partitions=1)``
    for every probed length, which the runner folds into a ``MultiAttnKey``
    keyed by ``"verify"`` (no ``"draft"`` without speculative decoding).
    """
    return MultiAttnKey.from_dict(
        {
            "verify": MHAAttnKey(
                batch_size=batch_size, max_prompt_length=1, num_partitions=1
            )
        }
    )


def _mock_kv_params() -> SimpleNamespace:
    """A minimal KVCacheParams stand-in exposing the dispatch resolver API."""
    return SimpleNamespace(
        devices=[DeviceRef.CPU()],
        is_mla=False,
        n_kv_heads_per_device=1,
        num_q_heads_per_device=1,
        is_fp8_kv_dtype=False,
        data_parallel_degree=1,
        num_draft_tokens_per_step=0,
        graph_capture_probe_cache_lengths=lambda max_cache_length, q: [1],
        resolve_attn_key=lambda batch_size, q, cache_length: MHAAttnKey(
            batch_size=batch_size, max_prompt_length=q, num_partitions=1
        ),
    )


@pytest.fixture
def host_memory_manager_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_SIZE", f"{64 * MiB}"
    )
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_CHUNK_PERCENT", "100"
    )
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SELF_CHECK", "True"
    )
    monkeypatch.setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_LOG", "False")


class OutputAllocatingModel:
    def __init__(self, output_bytes: int) -> None:
        self._device = CPU()
        self._output_bytes = output_bytes
        self.input_devices = [self._device]
        self.released_graph_keys: list[int] = []

    def capture(self, graph_key: int, *buffers: Buffer) -> list[Buffer]:
        del graph_key, buffers
        return [
            Buffer.zeros(
                (self._output_bytes,), dtype=DType.uint8, device=self._device
            )
        ]

    def replay(self, graph_key: int, *buffers: Buffer) -> None:
        del graph_key, buffers
        raise AssertionError("replay is not used in warmup")

    def debug_verify_replay(self, graph_key: int, *buffers: Buffer) -> None:
        del graph_key, buffers
        raise AssertionError("debug_verify_replay is not used in warmup")

    def release_captured_graph(self, graph_keys: int | Sequence[int]) -> None:
        # Mirrors the engine API surface. ``ServeGraphCaptureRunner`` only
        # forwards a single packed int; we record exactly that for assertions.
        assert isinstance(graph_keys, int)
        self.released_graph_keys.append(graph_keys)


def _make_kv_per_device() -> KVCacheInputsPerDevice[Buffer, Buffer]:
    max_prompt_length = np.array([0], dtype=np.uint32)
    max_cache_length = np.array([1], dtype=np.uint32)
    dispatch = np.array([1, 1, 1, 1], dtype=np.int64)
    return KVCacheInputsPerDevice(
        kv_blocks=Buffer.zeros((1,), dtype=DType.float32),
        cache_lengths=Buffer.from_numpy(np.array([0], dtype=np.uint32)),
        lookup_table=Buffer.from_numpy(np.array([0], dtype=np.uint32)),
        max_prompt_length=Buffer.from_numpy(max_prompt_length),
        max_cache_length=Buffer.from_numpy(max_cache_length),
        attention_dispatch_metadata=Buffer.from_numpy(dispatch),
    )


@contextmanager
def _warmup_model_inputs(
    batch_size: int, batch_characteristics: BatchCharacteristics
) -> Iterator[MockModelInputs]:
    yield MockModelInputs(
        active_batch_size=batch_size,
        eos_prob=0.0,
        kv_cache_inputs=KVCacheInputs[Buffer, Buffer](
            inputs=[_make_kv_per_device()]
        ),
    )


def test_warmup_pre_ready_releases_capture_outputs(
    host_memory_manager_config: None,
) -> None:
    del host_memory_manager_config

    # Warmup captures three graph entries (batch sizes 1..3). Each capture
    # allocates a fresh 48 MiB output buffer while the host memory manager is
    # capped at 64 MiB. If graph_capture warmup keeps owning output buffers
    # alive, the second/third capture should run out of memory. This test only
    # passes if warmup releases stored outputs to borrowed wrappers so the
    # memory manager can reuse the same backing allocation.
    output_bytes = 48 * MiB
    model = OutputAllocatingModel(output_bytes)
    kv_params = _mock_kv_params()
    runner = ServeGraphCaptureRunner(
        model=cast(Model, model),
        kv_params=cast(KVCacheParams, kv_params),
        warmup_model_inputs=_warmup_model_inputs,
        max_cache_length_upper_bound=1,
        max_batch_size=3,
    )

    runner.warmup_pre_ready()

    assert set(runner.graph_entries) == {_gk(1), _gk(2), _gk(3)}
    for _inputs, outputs in runner.graph_entries.values():
        assert outputs.logits.num_elements == output_bytes


def test_release_graph_drops_entry_and_forwards_to_model(
    host_memory_manager_config: None,
) -> None:
    del host_memory_manager_config

    output_bytes = 48 * MiB
    model = OutputAllocatingModel(output_bytes)
    kv_params = _mock_kv_params()
    runner = ServeGraphCaptureRunner(
        model=cast(Model, model),
        kv_params=cast(KVCacheParams, kv_params),
        warmup_model_inputs=_warmup_model_inputs,
        max_cache_length_upper_bound=1,
        max_batch_size=2,
    )

    runner.warmup_pre_ready()

    assert set(runner.graph_entries) == {_gk(1), _gk(2)}

    target = _gk(1)
    runner.release_graph(target)

    assert target not in runner.graph_entries
    assert set(runner.graph_entries) == {_gk(2)}
    assert len(model.released_graph_keys) == 1

    # Releasing the same key again is idempotent at the runner level and still
    # forwards to the model (engine-side release is also a no-op for unknown
    # keys, so this is safe).
    runner.release_graph(target)
    assert set(runner.graph_entries) == {_gk(2)}
    assert len(model.released_graph_keys) == 2
