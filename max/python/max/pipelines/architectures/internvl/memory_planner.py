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

"""Memory planner for the InternVL architecture."""

from __future__ import annotations

from max.pipelines.architectures.internvl.tokenizer import InternVLImageConfig
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from transformers import AutoConfig


class InternVLMemoryPlanner(PagedMemoryPlanner):
    """Memory planner for InternVL vision-language models.

    Accounts for vision-encoder image buffers and language-model
    intermediate activations.
    """

    _always_signal_buffers = True

    def estimate_activation_memory(
        self, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Estimates the activation memory required for InternVL model execution.

        This accounts for the temporary memory buffers used during model execution,
        particularly for the vision encoder and language model activations.

        Based on empirical analysis of MGP buffer plans (GEX-2365):
        - Vision encoder uses ~128MiB per image.
        - Language model uses ~100KB per token for intermediate activations.

        These values come from printing the high water mark from the
        `mgp.buffer.plan` op, and verifying with GPU free memory at runtime.

        The vision encoder memory scales with the number of images that can be
        processed concurrently, which is limited by max_batch_input_tokens / num_image_tokens
        where num_image_tokens=256 for InternVL.

        TODO(GEX-2365): Replace this with a more general solution that analyzes
        the compiled graph's memory requirements directly.

        Args:
            pipeline_config: Pipeline configuration
            huggingface_config: HuggingFace model configuration

        Returns:
            Estimated activation memory in bytes
        """
        # Vision encoder memory estimation.
        vision_memory_per_image = 128 * 1024 * 1024  # 128 MiB per image

        image_config = InternVLImageConfig(
            huggingface_config,
            pipeline_config.model.vision_config_overrides,
        )

        # Maximum number of images that can be processed is limited by
        # how many image tokens fit in the target new tokens
        max_images = (
            pipeline_config.runtime.max_batch_input_tokens
            // image_config.num_image_token
        )
        # Ensure at least 1 image worth of memory.
        max_images = max(1, max_images)

        # Note: Each image can use up to max_dynamic_patch patches (default 12)
        # plus 1 for thumbnail if applicable.
        if not pipeline_config.runtime.enable_chunked_prefill:
            # When there's no chunked prefill, the number of images may overhang
            # by the maximum in a single request.
            # Since we only support a single image per request for now,
            # TODO(MODELS-638, E2EOPT-350): Adjust this after supporting
            # multi-image requests.
            max_images += image_config.max_dynamic_patch + 1

        vision_activation_memory = max_images * vision_memory_per_image

        # Language model memory estimation
        # ~100KB per token for intermediate activations
        llm_memory_per_token = 100 * 1024  # 100 KiB
        llm_activation_memory = (
            pipeline_config.runtime.max_batch_input_tokens
            * llm_memory_per_token
        )

        total_activation_memory = (
            vision_activation_memory + llm_activation_memory
        )

        # Multiply by the number of devices since the above analysis is per
        # device, but memory estimation uses total memory across all devices.
        return len(pipeline_config.model.device_specs) * total_activation_memory
