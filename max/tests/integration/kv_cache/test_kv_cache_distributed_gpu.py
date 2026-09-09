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


import numpy as np
import pytest
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams, MLAKVCacheParams
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context


@pytest.mark.asyncio
async def test_kv_cache_multi_gpu() -> None:
    num_devices = accelerator_count()

    if num_devices > 1:
        list_of_devices = [Accelerator(id=i) for i in range(num_devices)]
        inference_session = InferenceSession(devices=list_of_devices)
        kv_params = MHAKVCacheParams(
            n_kv_heads=8,
            head_dim=128,
            dtype=DType.bfloat16,
            num_layers=32,
            page_size=128,
            devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        )
        kv_manager = PagedKVCacheManager(
            params=kv_params,
            total_num_pages=8,
            session=inference_session,
            max_batch_size=128,
        )
        context = create_text_context(np.empty(1))
        kv_manager.claim(context)

        batch = [context]
        kv_manager.alloc(context)
        kv_inputs = kv_manager.runtime_inputs_for_leaf([batch])
        for i in range(num_devices):
            kv_inputs_per_device = kv_inputs.inputs[i]
            assert len(kv_inputs_per_device.flatten()) == 6
            assert kv_inputs_per_device.attention_dispatch_metadata is not None


@pytest.mark.asyncio
@pytest.mark.skipif(
    accelerator_count() < 2,
    reason="requires at least 2 GPUs",
)
async def test_mla_runtime_inputs_keep_dispatch_metadata_on_shard_device() -> (
    None
):
    devices = [Accelerator(id=i) for i in range(2)]
    session = InferenceSession(devices=devices)
    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=128,
        num_layers=1,
        page_size=128,
        num_q_heads=16,
        devices=[DeviceRef.GPU(i) for i in range(2)],
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )
    context = create_text_context(np.empty(1))
    kv_manager.claim(context)
    kv_manager.alloc(context)

    kv_inputs = kv_manager.runtime_inputs_for_leaf([[context]])

    for shard_idx, kv_inputs_per_device in enumerate(kv_inputs.inputs):
        assert kv_inputs_per_device.attention_dispatch_metadata is not None
        assert (
            kv_inputs_per_device.attention_dispatch_metadata.device
            == devices[shard_idx]
        )
        assert kv_inputs_per_device.attention_dispatch_metadata.device == (
            kv_inputs_per_device.kv_blocks.device
        )


# ``test_swapping_to_host_multi_gpu`` was removed with the Python host tier: the
# host/disk tier is now the Rust ``rust_tiered`` connector, whose pyo3 extension
# may only be depended on from an internal-only package. Its host-tier
# offload/onload coverage lives in
# ``internal/dkv/test_rust_tiered_connector_gpu.py``.
