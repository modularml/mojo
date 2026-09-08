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
"""Tests for log_basic_config output.

Guards against a regression where max_seq_len was missing from server startup
logs. The value comes from the memory plan the caller passes in. (cache_memory
now logs from the memory planner, where the KV budget is computed, rather than
from this config logger.)

Also covers the speculative block: which acceptance knobs a run is using has
to be readable from the startup dump, or an A/B over them cannot be
attributed.
"""

from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock, patch

from max.driver import DeviceSpec
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MemoryPlan,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.logging_utils import log_basic_config
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.speculative import SpeculativeConfig


def _make_pipeline_config(
    max_length: int | None, speculative: SpeculativeConfig | None = None
) -> PipelineConfig:
    """Build a minimal PipelineConfig without triggering full validation."""
    model_config = MAXModelConfig.model_construct(
        model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
        device_specs=[DeviceSpec.cpu()],
        max_length=max_length,
        kv_cache=KVCacheConfig(),
    )
    model_config._huggingface_config = MagicMock()

    runtime = PipelineRuntimeConfig.model_construct()
    return PipelineConfig.model_construct(
        runtime=runtime,
        models=ModelManifest({"main": model_config}),
        speculative=speculative,
    )


def _capture_log_basic_config(
    config: PipelineConfig, memory_plan: MemoryPlan
) -> str:
    """Call log_basic_config with mocked registry and return all logged lines."""
    arch = MagicMock()
    arch.name = "LlamaForCausalLM"
    arch.task = PipelineTask.TEXT_GENERATION

    pipeline_cls = type("TextGenerationPipeline", (), {})

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = _Capture()
    logger = logging.getLogger("max.pipelines")
    original_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    try:
        with (
            patch(
                "max.pipelines.logging_utils.PIPELINE_REGISTRY.retrieve_architecture",
                return_value=arch,
            ),
            patch(
                "max.pipelines.logging_utils.get_pipeline_for_task",
                return_value=pipeline_cls,
            ),
        ):
            log_basic_config(config, memory_plan=memory_plan)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(original_level)

    return "\n".join(r.getMessage() for r in records)


class TestLogBasicConfigAfterResolve:
    """log_basic_config must reflect resolved config values."""

    def test_max_seq_len_present_after_resolve(self) -> None:
        """max_seq_len must show the resolved value, not None."""
        config = _make_pipeline_config(max_length=131072)
        memory_plan = MemoryPlan(
            planned_max_batch_size=1, footprint=0, planned_max_length=131072
        )
        output = _capture_log_basic_config(config, memory_plan)
        assert "131072" in output, (
            f"Expected max_seq_len=131072 in log output:\n{output}"
        )


class TestLogBasicConfigSpeculative:
    """The speculative acceptance knobs must be readable at startup."""

    def test_explicit_rejection_strategy_is_logged(self) -> None:
        """``--rejection-sampling-strategy residual`` has to show up.

        It logged nowhere before, so a run's acceptance rule was
        unverifiable from its own output.
        """
        config = _make_pipeline_config(
            max_length=4096,
            speculative=SpeculativeConfig(
                speculative_method="eagle",
                rejection_sampling_strategy="residual",
            ),
        )
        output = _capture_log_basic_config(
            config,
            MemoryPlan(
                planned_max_batch_size=1, footprint=0, planned_max_length=4096
            ),
        )

        assert "rejection_sampling_strategy" in output, output
        assert "residual" in output, output
        # Nothing consumes the field, so the dump has to say so rather than
        # imply the run is verifying drafts that way.
        assert "inert" in output, output

    def test_unset_rejection_strategy_is_logged(self) -> None:
        """The line appears even unset, so its absence means an old build."""
        config = _make_pipeline_config(
            max_length=4096,
            speculative=SpeculativeConfig(speculative_method="eagle"),
        )
        output = _capture_log_basic_config(
            config,
            MemoryPlan(
                planned_max_batch_size=1, footprint=0, planned_max_length=4096
            ),
        )

        assert re.search(r"rejection_sampling_strategy +: unset", output), (
            output
        )
        # The documented "typical-acceptance for eagle/mtp" default is not
        # applied anywhere in the code, so the dump must not claim it.
        assert "typical-acceptance" not in output, output

    def test_acceptance_deciding_knobs_are_logged(self) -> None:
        """The fields AcceptanceSampler actually dispatches on."""
        config = _make_pipeline_config(
            max_length=4096,
            speculative=SpeculativeConfig(
                speculative_method="mtp",
                synthetic_acceptance_rate=0.75,
                draft_proposal="sampled",
            ),
        )
        output = _capture_log_basic_config(
            config,
            MemoryPlan(
                planned_max_batch_size=1, footprint=0, planned_max_length=4096
            ),
        )

        for key, value in (
            ("synthetic_acceptance_rate", "0.75"),
            ("draft_proposal", "sampled"),
            ("use_greedy_acceptance", "False"),
        ):
            assert re.search(rf"{key} +: {value}", output), output
