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

"""Memory planner for the Qwen3.5 (GatedDeltaNet) architecture."""

from __future__ import annotations

from max.driver import Device
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from transformers import AutoConfig

from .model_config import Qwen3_5Config


class Qwen3_5MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for Qwen3.5 GatedDeltaNet SSM models.

    Accounts for the slot-indexed recurrent state pool.
    """

    _always_signal_buffers = True

    #: Set by :meth:`infer_max_batch_size`; consumed by
    #: :meth:`estimate_activation_memory` when the user left
    #: ``max_batch_size`` unset.
    _inferred_max_batch_size: int | None = None

    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        """Infers a memory-safe default ``max_batch_size``.

        Qwen3.5 has per-request GPU overhead (recurrent-state buffers)
        beyond the KV cache, so the framework's default inference can OOM;
        compute a bound that fits actual free memory instead.
        """
        assert isinstance(self._config, Qwen3_5Config)
        inferred = self._config.infer_optimal_batch_size(
            devices,
            weights_size=weights_size,
            device_memory_utilization=(
                pipeline_config.model.kv_cache.device_memory_utilization
            ),
        )
        self._inferred_max_batch_size = inferred
        return inferred

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserve GPU memory for the GatedDeltaNet state pool.

        The slot-indexed SSM kernels mutate the conv and recurrent pools in
        place; there are no working buffers and no graph-output pool, so
        peak footprint is a single ``max_batch x per_req`` allocation.

        Memory planning calls :meth:`infer_max_batch_size` before this
        method when the user left ``max_batch_size`` unset, so it is
        always known here.
        """
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        layer_types = Qwen3_5Config._get_layer_types(text_config)
        num_linear = sum(1 for lt in layer_types if lt == "linear_attention")

        nk = getattr(text_config, "linear_num_key_heads", 16)
        nv = getattr(text_config, "linear_num_value_heads", 48)
        kd = getattr(text_config, "linear_key_head_dim", 128)
        vd = getattr(text_config, "linear_value_head_dim", 128)
        kernel = getattr(text_config, "linear_conv_kernel_dim", 4)

        conv_dim = 2 * kd * nk + vd * nv
        # `state_dtype` is the one property every pool declarer reads: it
        # carries the `state_pool_dtype` override and, absent one, the compute
        # dtype. The encoding's storage dtype is 1-byte `uint8` for packed
        # NVFP4 and would under-reserve by 2x, or 4x against a float32 pool.
        assert isinstance(self._config, Qwen3_5Config)
        dtype_bytes = self._config.state_dtype.size_in_bytes
        bytes_per_layer = (
            conv_dim * (kernel - 1) * dtype_bytes + nv * kd * vd * dtype_bytes
        )
        per_req = num_linear * bytes_per_layer

        max_batch = pipeline_config.runtime.max_batch_size
        if max_batch is None:
            max_batch = self._inferred_max_batch_size
        assert max_batch is not None, (
            "infer_max_batch_size must run before estimate_activation_memory "
            "when max_batch_size is unset"
        )
        # 1x: single in-place pool — kernels mutate it via slot_idx.
        return max_batch * per_req if num_linear > 0 else 0
