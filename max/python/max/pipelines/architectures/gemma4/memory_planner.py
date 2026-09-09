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

"""Memory planner for the Gemma4 architecture."""

from __future__ import annotations

from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from transformers import AutoConfig

from .model_config import Gemma4ForConditionalGenerationConfig


class Gemma4MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for Gemma4 (vision-language) models.

    Reserves a per-device activation budget (a base sized from the KV cache
    dtype), scaled by the device count to match the total-across-devices
    budget in
    :meth:`MemoryEstimator.plan_from_sizes`.  Also provides vision
    cache entry byte estimation for the KV-and-vision-cache reservation path.
    """

    _always_signal_buffers = True

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates activation memory for Gemma4 models.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: Unused.

        Returns:
            Estimated activation memory in bytes, summed across all devices.
        """
        # FIXME: We arbitrarily set some memory for activation memory to leave
        # headroom for vision processing. We should determine this in a more
        # principled way.
        # Smaller KV cache dtypes (e.g. FP8) halve bytes_per_block, so the
        # same KV budget buys ~2x more blocks.  The scheduler admits work
        # based on available blocks, so it targets larger concurrent batches
        # whose activation tensors need proportionally more headroom.
        # TODO(MODELS-1544): investigate high activation memory estimates
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model,
            Gemma4ForConditionalGenerationConfig.DEFAULT_ENCODING,
        )
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )
        base = (30 // cache_dtype.size_in_bytes) * 1024**3
        return base * len(pipeline_config.model.device_specs)
