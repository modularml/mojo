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

"""Lazy-trace tests for TensorParallelMoE and ExpertParallelMoE."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.functional_kernels import (
    moe_create_indices,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
)
from max.experimental.tensor import Tensor, default_dtype
from max.nn.comm.ep import EPConfig
from max.nn.comm.ep.ep_config import NUM_GROUPS
from max.nn.comm.ep.ep_manager import (
    EPBatchManager,
    EPCommBuffers,
    get_ep_local_sync_counters_size,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_moe import (
    ExpertParallelMoE,
    TensorParallelMoE,
)
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
    NVFP4Tensor,
)

_HIDDEN_DIM = 256
_MOE_DIM = 128
_NUM_EXPERTS = 4
_NUM_EXPERTS_PER_TOKEN = 2
_SEQ_LEN = 4

# FP8 block-scaled multi-device dims chosen so the (128, 128) weight-scale grid
# divides evenly across 2 devices on every sharded axis.
_FP8_HIDDEN_DIM = 256
_FP8_MOE_DIM = 512


# --------------------------------------------------------------------------- #
# Distributed dispatch of the MoE index kernel
# --------------------------------------------------------------------------- #


def test_moe_create_indices_forwards_keyword_only_args(
    mock_accelerator: MagicMock,
) -> None:
    """The NVFP4 scale-offset output survives the per-shard dispatch.

    ``needs_scales_offset`` is keyword-only, so the sharded-call path has to
    forward non-tensor keyword arguments to every per-device call; dropping
    them would silently cost the NVFP4 MoE its sixth index output.
    """
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))
        topk_ids = Tensor.zeros(
            [_SEQ_LEN * _NUM_EXPERTS_PER_TOKEN],
            dtype=DType.int32,
            device=replicated,
        )

        assert len(moe_create_indices(topk_ids, _NUM_EXPERTS)) == 5

        with_offsets = moe_create_indices(
            topk_ids, _NUM_EXPERTS, needs_scales_offset=True
        )
        assert len(with_offsets) == 6
        assert with_offsets[-1].mapping.mesh == mesh


# --------------------------------------------------------------------------- #
# TensorParallelMoE
# --------------------------------------------------------------------------- #


def test_tensor_parallel_moe_bf16(mock_accelerator: MagicMock) -> None:
    """TP shards every expert's moe_dim; forward returns a Partial sum."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))

        with default_dtype(DType.bfloat16):
            layer = TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
            ).to(mesh)

            gate_up = layer.gate_up_proj
            assert len(gate_up) == num_devices
            for i, shard in enumerate(gate_up):
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    _NUM_EXPERTS,
                    2 * _MOE_DIM // num_devices,
                    _HIDDEN_DIM,
                ]
                assert shard.device == devices[i]

            down = layer.down_proj
            assert len(down) == num_devices
            for shard in down:
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    _NUM_EXPERTS,
                    _HIDDEN_DIM,
                    _MOE_DIM // num_devices,
                ]

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.mapping.mesh == mesh
        # forward returns each device's partial sum; the single all-reduce that
        # resolves it lives in the transformer block after the MoE layer.
        assert out.mapping.to_placements() == (Partial(),)


def test_tensor_parallel_moe_bf16_shared_experts(
    mock_accelerator: MagicMock,
) -> None:
    """Routed + shared experts fold into one Partial sum (single all-reduce)."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))

        with default_dtype(DType.bfloat16):
            layer = TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                has_shared_experts=True,
                shared_experts_dim=_MOE_DIM,
            ).to(mesh)

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.mapping.mesh == mesh
        # The shared expert is computed inside the per-device local_map and
        # summed with the routed experts, so the layer output is a single
        # Partial sum (one all-reduce resolves both, in the transformer block).
        assert out.mapping.to_placements() == (Partial(),)


def test_tensor_parallel_moe_fp8_weights(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 TP shards both the packed data and the block-scale grid."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))

        layer = TensorParallelMoE(
            hidden_dim=_FP8_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_FP8_MOE_DIM,
            quant_config=fp8_quant_config,
        ).to(mesh)

        gate_up = layer.gate_up_proj
        assert isinstance(gate_up, list)
        assert isinstance(gate_up[0], FP8BlockTensor)
        assert list(gate_up[0].data.shape) == [
            _NUM_EXPERTS,
            2 * _FP8_MOE_DIM // num_devices,
            _FP8_HIDDEN_DIM,
        ]
        # Scale grid is the (128, 128)-block count of the sharded data.
        assert list(gate_up[0].weight_scale_inv.shape) == [
            _NUM_EXPERTS,
            2 * _FP8_MOE_DIM // num_devices // 128,
            _FP8_HIDDEN_DIM // 128,
        ]

        down = layer.down_proj
        assert isinstance(down, list)
        assert isinstance(down[0], FP8BlockTensor)
        assert list(down[0].data.shape) == [
            _NUM_EXPERTS,
            _FP8_HIDDEN_DIM,
            _FP8_MOE_DIM // num_devices,
        ]
        assert list(down[0].weight_scale_inv.shape) == [
            _NUM_EXPERTS,
            _FP8_HIDDEN_DIM // 128,
            _FP8_MOE_DIM // num_devices // 128,
        ]

        x = Tensor.zeros(
            [_SEQ_LEN, _FP8_HIDDEN_DIM],
            dtype=DType.bfloat16,
            device=replicated,
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _FP8_HIDDEN_DIM]
    assert out.mapping.mesh == mesh
    # The FP8 grouped matmul still runs per-device via local_map (see
    # quant_moe.TensorParallelMoE.apply_experts), but the layer's output
    # contract is unchanged: a Partial sum for the caller to all-reduce.
    assert out.mapping.to_placements() == (Partial(),)


# --------------------------------------------------------------------------- #
# ExpertParallelMoE
# --------------------------------------------------------------------------- #


def _build_ep_batch_manager(
    config: EPConfig, devices: list[Device]
) -> tuple[EPBatchManager, EPCommBuffers]:
    """Construct an EPBatchManager and placeholder comm buffers for it.

    Bypasses :meth:`EPBatchManager.comm_buffers` so the EP forward path can be
    traced under :func:`F.lazy` without wiring up real graph inputs. The
    returned buffers are what the caller threads into
    ``ExpertParallelMoE.forward``, which binds them onto the manager.
    """
    n_devices = config.n_gpus_per_node
    n_experts_for_counters = (
        config.n_experts // n_devices
        if config.use_allreduce
        else config.n_experts
    )
    counter_size = get_ep_local_sync_counters_size(n_experts_for_counters)

    atomic_counters = [
        [
            Tensor.zeros([counter_size], dtype=DType.int32, device=devices[i])
            for i in range(n_devices)
        ]
        for _ in range(NUM_GROUPS)
    ]

    def _make_ptrs() -> list[Tensor]:
        return [
            Tensor.zeros([n_devices], dtype=DType.uint64, device=CPU())
            for _ in range(NUM_GROUPS)
        ]

    comm = EPCommBuffers(
        atomic_counters=atomic_counters,
        send_buf_ptrs=_make_ptrs(),
        recv_buf_ptrs=_make_ptrs(),
        recv_count_ptrs=_make_ptrs(),
    )
    return EPBatchManager(config), comm


def _ep_config(dispatch_dtype: DType, num_devices: int, **kwargs) -> EPConfig:
    return EPConfig(
        dispatch_dtype=dispatch_dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=_HIDDEN_DIM,
        top_k=_NUM_EXPERTS_PER_TOKEN,
        n_experts=_NUM_EXPERTS,
        max_tokens_per_rank=_SEQ_LEN,
        n_gpus_per_node=num_devices,
        n_nodes=1,
        **kwargs,
    )


def test_expert_parallel_moe_bf16(mock_accelerator: MagicMock) -> None:
    """EP distributes whole experts; weights keep full moe_dim per device."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))

        ep_batch_manager, comm_buffers = _build_ep_batch_manager(
            _ep_config(DType.bfloat16, num_devices), devices
        )

        with default_dtype(DType.bfloat16):
            layer = ExpertParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                ep_batch_manager=ep_batch_manager,
            ).to(mesh)

            gate_up = layer.gate_up_proj
            assert len(gate_up) == num_devices
            for i, shard in enumerate(gate_up):
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    num_local_experts,
                    2 * _MOE_DIM,
                    _HIDDEN_DIM,
                ]
                assert shard.device == devices[i]

            down = layer.down_proj
            assert len(down) == num_devices
            for shard in down:
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    num_local_experts,
                    _HIDDEN_DIM,
                    _MOE_DIM,
                ]

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x, comm_buffers)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.mapping.mesh == mesh


def test_expert_parallel_moe_fp8_weights(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 EP stacks per-device FP8 expert weights (data + scale)."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        ep_batch_manager, _ = _build_ep_batch_manager(
            _ep_config(
                DType.float8_e4m3fn,
                num_devices,
                dispatch_quant_config=fp8_quant_config,
            ),
            devices,
        )

        layer = ExpertParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=fp8_quant_config,
            ep_batch_manager=ep_batch_manager,
        ).to(mesh)

        gate_up = layer.gate_up_proj
        assert len(gate_up) == num_devices
        for i, shard in enumerate(gate_up):
            assert isinstance(shard, FP8BlockTensor)
            assert list(shard.data.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM,
            ]
            assert shard.data.device == devices[i]

        down = layer.down_proj
        assert len(down) == num_devices
        for shard in down:
            assert isinstance(shard, FP8BlockTensor)
            assert list(shard.data.shape) == [
                num_local_experts,
                _HIDDEN_DIM,
                _MOE_DIM,
            ]


def test_tensor_parallel_moe_nvfp4_weights(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """NVFP4 TP co-shards the packed data and the 16-element block scales."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        layer = TensorParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=nvfp4_quant_config,
        ).to(mesh)

        gate_up = layer.gate_up_proj
        assert len(gate_up) == num_devices
        assert isinstance(gate_up[0], NVFP4Tensor)
        assert list(gate_up[0].data.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM // num_devices,
            _HIDDEN_DIM // 2,
        ]
        assert list(gate_up[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM // num_devices,
            _HIDDEN_DIM // 16,
        ]
        # gate and up share one global scale, so the fused bundle keeps one
        # entry per expert even though it stacks 2 * num_experts weights.
        assert list(gate_up[0].weight_scale_2.shape) == [_NUM_EXPERTS]
        assert list(gate_up[0].input_scale.shape) == [_NUM_EXPERTS]

        down = layer.down_proj
        assert len(down) == num_devices
        assert isinstance(down[0], NVFP4Tensor)
        assert list(down[0].data.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // 2 // num_devices,
        ]
        assert list(down[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // 16 // num_devices,
        ]
        # The per-tensor scales are replicated, not sharded.
        assert list(down[0].weight_scale_2.shape) == [_NUM_EXPERTS]
        assert list(down[0].input_scale.shape) == [_NUM_EXPERTS]


def test_tensor_parallel_moe_nvfp4_forward(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """NVFP4 TP forward returns a Partial sum for the caller to all-reduce."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))

        layer = TensorParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=nvfp4_quant_config,
        ).to(mesh)

        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM],
            dtype=DType.bfloat16,
            device=replicated,
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.mapping.mesh == mesh
    assert out.mapping.to_placements() == (Partial(),)


def test_expert_parallel_moe_nvfp4_weights(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """EP keeps whole NVFP4 experts per device, one global scale pair each."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        ep_batch_manager, _ = _build_ep_batch_manager(
            _ep_config(
                DType.uint8,
                num_devices,
                dispatch_quant_config=nvfp4_quant_config,
            ),
            devices,
        )

        layer = ExpertParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=nvfp4_quant_config,
            ep_batch_manager=ep_batch_manager,
        ).to(mesh)

        gate_up = layer.gate_up_proj
        assert len(gate_up) == num_devices
        for i, shard in enumerate(gate_up):
            assert isinstance(shard, NVFP4Tensor)
            assert list(shard.data.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 2,
            ]
            assert list(shard.weight_scale.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 16,
            ]
            # gate and up share one global scale per expert.
            assert list(shard.weight_scale_2.shape) == [num_local_experts]
            assert shard.data.device == devices[i]

        down = layer.down_proj
        assert len(down) == num_devices
        for shard in down:
            assert isinstance(shard, NVFP4Tensor)
            assert list(shard.data.shape) == [
                num_local_experts,
                _HIDDEN_DIM,
                _MOE_DIM // 2,
            ]
            assert list(shard.weight_scale_2.shape) == [num_local_experts]


def test_expert_parallel_moe_nvfp4_fused_swiglu_permutes_gate_up(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """The fused SwiGLU kernel gets row-interleaved gate/up weights.

    The permutation itself is shape-preserving (and value-checked in
    ``test_quant_ops``); what matters here is that the config flag reaches the
    weight property.
    """
    fused = dataclasses.replace(nvfp4_quant_config, can_use_fused_swiglu=True)
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        def _build(quant_config: QuantConfig) -> ExpertParallelMoE:
            ep_batch_manager, _ = _build_ep_batch_manager(
                _ep_config(
                    DType.uint8,
                    num_devices,
                    dispatch_quant_config=quant_config,
                ),
                devices,
            )
            return ExpertParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                quant_config=quant_config,
                ep_batch_manager=ep_batch_manager,
            ).to(mesh)

        chained = _build(nvfp4_quant_config)
        assert not chained._uses_fused_swiglu

        layer = _build(fused)
        assert layer._uses_fused_swiglu

        gate_up = layer.gate_up_proj
        for shard in gate_up:
            assert isinstance(shard, NVFP4Tensor)
            assert list(shard.data.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 2,
            ]
            assert list(shard.weight_scale.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 16,
            ]
