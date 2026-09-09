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

import dataclasses
import logging
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from max.driver import CPU, DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm import Signals
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import (
    KVCacheConfig,
    MemoryEstimator,
)
from max.pipelines.lib.interfaces import (
    ArchConfigWithKVCache,
)
from max.pipelines.lib.memory_estimation import (
    MemoryPlan,
    _kv_params_per_layer_depth,
    _max_per_layer_buffer_count,
)
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DummyLlamaPipelineModel,
)

GIB = 1024**3


def _mha_params(num_layers: int, per_layer_buffers: bool) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        page_size=128,
        per_layer_buffers=per_layer_buffers,
    )


def test_per_layer_depth_single_buffer_pool_is_one() -> None:
    """A normal single-buffer pool contributes a multiplier of 1."""
    assert _kv_params_per_layer_depth(_mha_params(50, False)) == 1


def test_per_layer_depth_reports_num_layers() -> None:
    """A per-layer pool contributes its layer count as the multiplier."""
    assert _kv_params_per_layer_depth(_mha_params(50, True)) == 50


def test_per_layer_depth_tree_takes_largest_flagged_child() -> None:
    """In a tree, only per-layer children count; the largest wins."""
    root = MultiKVCacheParams.from_params(
        {
            "sliding_attention": _mha_params(50, True),
            "full_attention": _mha_params(10, False),
        }
    )
    assert _kv_params_per_layer_depth(root) == 50


def test_per_layer_depth_tree_all_single_buffer_is_one() -> None:
    """A tree with no per-layer child keeps the multiplier at 1."""
    root = MultiKVCacheParams.from_params(
        {"a": _mha_params(50, False), "b": _mha_params(10, False)}
    )
    assert _kv_params_per_layer_depth(root) == 1


def test_max_per_layer_buffer_count_non_kv_arch_is_one() -> None:
    """A non-KV arch config leaves the allocation cap unchanged."""
    assert _max_per_layer_buffer_count(object()) == 1  # type: ignore[arg-type]


def test_max_per_layer_buffer_count_reads_arch_kv_params() -> None:
    """The multiplier is read from the arch config's KV params."""
    arch = MagicMock(spec=ArchConfigWithKVCache)
    arch.get_kv_params.return_value = _mha_params(50, True)
    assert _max_per_layer_buffer_count(arch) == 50


def test_memory_estimation__raise_oom_error_weights_size_exceeds_available_memory() -> (
    None
):
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 5 * 1024 * 1024}
        with pytest.raises(
            RuntimeError, match="Model size exceeds available memory"
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=4096,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )

            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(
                mock_config, max_seq_len=mock_config.model.max_length or 4096
            )
            MemoryEstimator.plan_from_sizes(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                50 * 1024 * 1024,
                0,
            )


def test_memory_estimation__infer_optimal_batch_size() -> None:
    # Max batch size on CPU is always 1.
    inferred_batch_size = MemoryEstimator._infer_optimal_batch_size(
        arch_config=MagicMock(spec=ArchConfigWithKVCache),
        devices=[CPU()],
    )
    assert inferred_batch_size == 1


def _dummy_llama_config(
    max_length: int | None, max_batch_total_tokens: int | None = None
) -> DummyPipelineConfig:
    """A dummy config whose mocked HF config carries concrete KV-sizing ints."""
    config = DummyPipelineConfig(
        model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
        max_batch_size=None,
        max_length=max_length,
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_total_tokens=max_batch_total_tokens,
    )
    hf_config = config.model.huggingface_config
    hf_config.max_position_embeddings = 4096
    hf_config.num_key_value_heads = 8
    hf_config.hidden_size = 1024
    hf_config.num_hidden_layers = 4
    return config


def _set_kv_connector(
    cfg: DummyPipelineConfig, kv_connector: KVConnectorType
) -> DummyPipelineConfig:
    kv = cfg.model.kv_cache
    return cfg.model_copy(
        update={
            "models": cfg.models.with_override(
                "main",
                kv_cache=kv.model_copy(
                    update={
                        "kv_connector_config": kv.kv_connector_config.model_copy(
                            update={"type": kv_connector}
                        )
                    }
                ),
            )
        }
    )


def _estimate(config: DummyPipelineConfig, **kwargs: Any) -> MemoryPlan:
    """Runs a successful estimate against ample mocked device memory."""
    with patch(
        "max.driver.Device.stats", new_callable=PropertyMock
    ) as device_mock:
        device_mock.return_value = {"free_memory": 100 * GIB}
        arch_config = DUMMY_LLAMA_ARCH.config.initialize(
            config, max_seq_len=config.model.max_length or 4096
        )
        estimate = MemoryEstimator.plan_from_sizes(
            config,
            config.model,
            arch_config,
            [CPU()],
            GIB,
            0,
            **kwargs,
        )
        assert isinstance(estimate, MemoryPlan)
        return estimate


def test_estimate__requires_resolved_max_length() -> None:
    """Planning consumes the construction-resolved max_length; it no longer
    runs the architecture policy, so an unresolved config is an error."""
    config = _dummy_llama_config(max_length=None)
    with pytest.raises(ValueError, match="max_length is unresolved"):
        _estimate(config)


def test_estimate__carries_user_max_length() -> None:
    config = _dummy_llama_config(max_length=512)
    estimate = _estimate(config)
    assert estimate.planned_max_length == 512
    assert config.model.max_length == 512


def test_for_pipeline__defaults_max_batch_total_tokens_when_arch_requires() -> (
    None
):
    arch = dataclasses.replace(
        DUMMY_LLAMA_ARCH, requires_max_batch_context_length=True
    )
    with (
        patch(
            "max.pipelines.lib.memory_estimation.load_devices",
            return_value=[CPU()],
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 100 * GIB}
        config = _dummy_llama_config(max_length=512)
        plan = MemoryEstimator.plan(config, arch)
        assert (
            plan.planned_max_batch_total_tokens
            == plan.planned_max_length
            == 512
        )
        # The plan carries a user-set cap unchanged instead.
        config = _dummy_llama_config(
            max_length=512, max_batch_total_tokens=2048
        )
        plan = MemoryEstimator.plan(config, arch)
        assert plan.planned_max_batch_total_tokens == 2048


def test_plan__leaves_config_unchanged() -> None:
    """Planning resolves effective values onto the plan; the config keeps
    carrying the construction-resolved ``max_length`` unchanged."""
    config = _dummy_llama_config(max_length=4096)
    with (
        patch(
            "max.pipelines.lib.memory_estimation.load_devices",
            return_value=[CPU()],
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 100 * GIB}
        plan = MemoryEstimator.plan(config, DUMMY_LLAMA_ARCH)
    assert plan.planned_max_length == 4096
    assert config.model.max_length == 4096
    assert config.runtime.max_batch_total_tokens is None
    assert plan.device_specs == (DeviceSpec.cpu(),)


MIB = 1024**2


def _overcommitted_llama_config(
    max_length: int | None,
) -> DummyPipelineConfig:
    """A dummy config whose KV byte budget overcommits free device memory.

    The paged KV size is capped at the budget ``free_memory *
    device_memory_utilization - static``, so with the (unbounded) utilization
    above 1 the planner's total can exceed free memory -- the only way to
    reach the shrink-to-fit branch of ``plan_from_sizes``.
    """
    config = _dummy_llama_config(max_length=max_length)
    return config.model_copy(
        update={
            "models": config.models.with_override(
                "main",
                kv_cache=KVCacheConfig(device_memory_utilization=1.5),
            )
        }
    )


def test_shrink_to_fit__runs_for_resolved_default_max_length(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Construction resolves ``model.max_length`` for every config, so
    planning cannot infer user intent from the field being set; it must read
    the intent bit captured before resolution. Without ``--max-length``, an
    oversized default is shrunk on the plan instead of raising an OOM error,
    and the config keeps the construction-resolved value."""
    config = _overcommitted_llama_config(max_length=None)
    # Mirror PipelineConfig._resolve_max_length: intent was captured at
    # construction (None -> not user provided); the policy value is then
    # stored on the config.
    config = config.model_copy(
        update={"models": config.models.with_override("main", max_length=4096)}
    )

    with patch(
        "max.driver.Device.stats", new_callable=PropertyMock
    ) as device_mock:
        device_mock.return_value = {"free_memory": 64 * MIB}
        arch_config = DUMMY_LLAMA_ARCH.config.initialize(
            config, max_seq_len=config.model.max_length or 4096
        )
        with caplog.at_level(logging.WARNING):
            plan = MemoryEstimator.plan_from_sizes(
                config,
                config.model,
                arch_config,
                [CPU()],
                10 * MIB,
                40 * MIB,
            )

    assert "Truncated model's default max_length" in caplog.text
    # Budget int(64 MiB * 1.5) - 50 MiB static = 46 MiB -> 23 pages of the
    # dummy's 2 MiB page -> 23 * 128 tokens.
    assert plan.planned_max_length == 2944
    assert config.model.max_length == 4096


def test_shrink_to_fit__skipped_for_user_provided_max_length() -> None:
    """A user-provided ``--max-length`` is a hard cap: planning raises
    instead of silently shrinking it."""
    config = _overcommitted_llama_config(max_length=4096)

    with patch(
        "max.driver.Device.stats", new_callable=PropertyMock
    ) as device_mock:
        device_mock.return_value = {"free_memory": 64 * MIB}
        arch_config = DUMMY_LLAMA_ARCH.config.initialize(
            config, max_seq_len=config.model.max_length or 4096
        )
        with pytest.raises(RuntimeError, match="exceeds available memory"):
            MemoryEstimator.plan_from_sizes(
                config,
                config.model,
                arch_config,
                [CPU()],
                10 * MIB,
                40 * MIB,
            )


def test_plan__kv_clamp_bounds_plan_not_config() -> None:
    """The KV-capacity clamp lowers ``planned_max_length``, not the config.

    With 64 MiB of device memory, the KV budget is ``int(0.9 * 64 MiB) - 1000``
    weight bytes, which holds 28 pages of the dummy's 2 MiB page (2 KV tensors
    x 4 layers x 128 tokens x 8 heads x 128 head_dim x 2 bytes), so the real
    clamp computation bounds a request to 28 x 128 = 3584 tokens -- below the
    construction-resolved 4096 (the model's ``max_position_embeddings``),
    which stays on the config untouched.
    """
    arch = dataclasses.replace(
        DUMMY_LLAMA_ARCH, requires_max_batch_context_length=True
    )
    config = _dummy_llama_config(max_length=4096)
    with (
        patch(
            "max.pipelines.lib.memory_estimation.load_devices",
            return_value=[CPU()],
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 64 * 1024 * 1024}
        plan = MemoryEstimator.plan(config, arch)
    assert plan.planned_max_length == 3584
    assert plan.planned_max_batch_total_tokens == 3584
    assert config.model.max_length == 4096
    assert config.runtime.max_batch_total_tokens is None


def test_estimate__draft_bound_lowers_plan_max_length() -> None:
    """A draft model's own sequence limit bounds the plan, not the config."""
    config = _dummy_llama_config(max_length=4096)
    estimate = _estimate(config, draft_max_seq_len=1500)
    assert estimate.planned_max_length == 1500
    assert config.model.max_length == 4096


@pytest.mark.skip("TODO: AITLIB-238")
def test_memory_estimation__raise_oom_error_all_defaults_no_valid_solution() -> (
    None
):
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 30641 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(
                mock_config, max_seq_len=mock_config.model.max_length or 4096
            )
            MemoryEstimator.plan_from_sizes(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                30000 * 1024 * 1024,
                0,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_all_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with caplog.at_level(logging.WARNING):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(
                mock_config, max_seq_len=mock_config.model.max_length or 4096
            )
            MemoryEstimator.plan_from_sizes(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                35000 * 1024 * 1024,
                0,
            )

        assert "Truncated model's default max_length from" in caplog.text


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_length_set() -> None:
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
            match=r"Try reducing --max-length to \d+ .*supports batch size of",
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=100000,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(
                mock_config, max_seq_len=mock_config.model.max_length or 4096
            )
            MemoryEstimator.plan_from_sizes(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                35000 * 1024 * 1024,
                0,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set() -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=4096
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=100000,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(
                mock_config, max_seq_len=mock_config.model.max_length or 4096
            )
            MemoryEstimator.plan_from_sizes(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                40000 * 1024 * 1024,
                0,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set_and_max_length_set() -> (
    None
):
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=100000,
                max_length=4096,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(
                mock_config, max_seq_len=mock_config.model.max_length or 4096
            )
            MemoryEstimator.plan_from_sizes(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                40000 * 1024 * 1024,
                0,
            )


@pytest.mark.parametrize(
    "device_specs,kv_connector,expected_count_per_gpu",
    [
        # Single-device: no signal buffers in the default path.
        ([DeviceSpec.cpu()], KVConnectorType.null, 0),
        ([DeviceSpec.accelerator(id=0)], KVConnectorType.null, 0),
        # Multi-GPU baseline: one set per device for the main model.
        (
            [DeviceSpec.accelerator(id=i) for i in range(2)],
            KVConnectorType.null,
            1,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(4)],
            KVConnectorType.null,
            1,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(8)],
            KVConnectorType.null,
            1,
        ),
        # KV connectors fan MLA-replicated blocks out via plain P2P copies
        # (see rust_kv/kv-tier-connector/src/copy_engine.rs), not a signal-buffer
        # broadcast, so none of them add an extra set.
        (
            [DeviceSpec.accelerator(id=i) for i in range(2)],
            KVConnectorType.tiered,
            1,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(4)],
            KVConnectorType.rust_tiered,
            1,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(8)],
            KVConnectorType.tiered,
            1,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(2)],
            KVConnectorType.dkv,
            1,
        ),
    ],
)
def test_estimate_signal_buffer_memory__default(
    device_specs: list[DeviceSpec],
    kv_connector: KVConnectorType,
    expected_count_per_gpu: int,
) -> None:
    """``PipelineConfig.estimate_signal_buffer_memory`` returns
    ``NUM_BYTES * count_per_gpu * ngpus`` for the in-scope allocation sites."""
    cfg = DummyPipelineConfig(
        model_path="dummy",
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_size=1,
        max_length=1024,
        device_specs=device_specs,
    )
    cfg = _set_kv_connector(cfg, kv_connector)

    expected = Signals.NUM_BYTES * expected_count_per_gpu * len(device_specs)
    assert cfg.estimate_signal_buffer_memory() == expected


@pytest.mark.parametrize(
    "ngpus,kv_connector,expected_count_per_gpu",
    [
        # Single-GPU: mixin allocates one set even though the default would not.
        (1, KVConnectorType.null, 1),
        # Multi-GPU: mixin matches the default.
        (2, KVConnectorType.null, 1),
        (4, KVConnectorType.tiered, 1),
        (8, KVConnectorType.rust_tiered, 1),
    ],
)
def test_estimate_signal_buffer_memory__always_signal_buffers_mixin(
    ngpus: int,
    kv_connector: KVConnectorType,
    expected_count_per_gpu: int,
) -> None:
    """Planners with ``always_signal_buffers=True`` allocate one set even at
    single-GPU and match the default for multi-GPU."""
    device_specs = [DeviceSpec.accelerator(id=i) for i in range(ngpus)]
    cfg = DummyPipelineConfig(
        model_path="dummy",
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_size=1,
        max_length=1024,
        device_specs=device_specs,
    )
    cfg = _set_kv_connector(cfg, kv_connector)

    arch_config = DUMMY_LLAMA_ARCH.config.initialize(
        cfg, max_seq_len=cfg.model.max_length or 4096
    )
    planner_cls = PagedMemoryPlanner.with_activation_reservation(
        0, always_signal_buffers=True
    )
    planner = planner_cls(arch_config)
    got = planner.estimate_signal_buffer_memory(cfg)
    expected = Signals.NUM_BYTES * expected_count_per_gpu * max(ngpus, 1)
    assert got == expected
