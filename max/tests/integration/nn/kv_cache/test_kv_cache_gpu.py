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

import asyncio

import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context

# The multi-cache connector offload regression (SERVOPT-1254) was removed with
# the Python host tier: the host/disk tier is now the Rust ``rust_tiered``
# connector, whose pyo3 extension may only be depended on from an internal-only
# package. Multi-buffer offload/onload stays covered by the round-trip tests in
# ``integration/kv_cache/internal/dkv/test_rust_tiered_connector_gpu.py``.


def test_kv_cache_gpu() -> None:
    asyncio.run(_test_kv_cache_gpu())


async def _test_kv_cache_gpu() -> None:
    device = Accelerator()
    kv_params = MHAKVCacheParams(
        n_kv_heads=8,
        head_dim=128,
        dtype=DType.bfloat16,
        num_layers=32,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        max_batch_size=128,
    )
    context = create_text_context(np.empty(1))
    kv_manager.claim(context)
    kv_manager.alloc(context)
    batch = [context]
    kv_inputs = kv_manager.runtime_inputs([batch])
    assert isinstance(kv_inputs, KVCacheInputs)
    first_device_inputs = kv_inputs.inputs[0]
    assert isinstance(first_device_inputs, KVCacheInputsPerDevice)
    assert len(first_device_inputs.flatten()) == 6
    assert first_device_inputs.attention_dispatch_metadata is not None
