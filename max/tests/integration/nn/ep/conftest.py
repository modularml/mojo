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

import os
from dataclasses import dataclass

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Signals
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.moe import MoE, MoEGate, make_concatenated_gated_activation_fn
from max.nn.moe.expert_parallel import forward_moe_sharded_layers

"""
Fixtures for EP tests, including dummy weights.
"""

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64
WEIGHTS_STDDEV = 0.01
N_DEVICES = 4
TOP_K = 8


@dataclass
class CompiledEPModels:
    """Pre-compiled MoE and MoEGate models for EP integration tests."""

    moe_model: Model
    gate_model: Model
    ep_comm_init: EPCommInitializer
    devices: list[Accelerator]
    swiglu_limit: float = 0.0


@dataclass
class CompiledAllreduceEPModels:
    """Pre-compiled MoE and MoEGate models for allreduce EP integration tests.

    Unlike ``CompiledEPModels``, the graph takes a single input on device 0,
    broadcasts it to all devices, runs EP MoE with ``use_allreduce=True``,
    and collects partial results via allreduce sum.
    """

    moe_model: Model
    gate_model: Model
    ep_comm_init: EPCommInitializer
    signal_buffers: list[Buffer]
    devices: list[Accelerator]


@pytest.fixture
def moe_weights_fp8() -> dict[str, torch.Tensor]:
    """Generate FP8 weights on GPU for fast random number generation."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fp8_dtype = torch.float8_e4m3fn
    scale_dtype = torch.float32
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = torch.finfo(fp8_dtype).min

    moe_weights = {}

    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * WEIGHTS_STDDEV
    )

    FP8_WEIGHTS_MULTIPLIER = 100
    SCALE_MIN = 0.5 * (WEIGHTS_STDDEV / FP8_WEIGHTS_MULTIPLIER)
    SCALE_MAX = WEIGHTS_STDDEV / FP8_WEIGHTS_MULTIPLIER
    for expert_idx in range(NUM_EXPERTS):
        moe_weights[f"experts.{expert_idx}.gate_proj.weight"] = (
            (
                torch.randn(
                    MOE_DIM,
                    HIDDEN_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.gate_proj.weight_scale"] = (
            torch.rand(
                MOE_DIM // 128,
                HIDDEN_DIM // 128,
                dtype=scale_dtype,
                device=device,
            )
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

        moe_weights[f"experts.{expert_idx}.up_proj.weight"] = (
            (
                torch.randn(
                    MOE_DIM,
                    HIDDEN_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.up_proj.weight_scale"] = (
            torch.rand(
                MOE_DIM // 128,
                HIDDEN_DIM // 128,
                dtype=scale_dtype,
                device=device,
            )
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

        moe_weights[f"experts.{expert_idx}.down_proj.weight"] = (
            (
                torch.randn(
                    HIDDEN_DIM,
                    MOE_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.down_proj.weight_scale"] = (
            torch.rand(
                HIDDEN_DIM // 128,
                MOE_DIM // 128,
                dtype=scale_dtype,
                device=device,
            )
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

    moe_weights["shared_experts.down_proj.weight"] = (
        (
            torch.randn(
                HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16, device=device
            )
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.down_proj.weight_scale"] = (
        torch.rand(
            HIDDEN_DIM // 128, MOE_DIM // 128, dtype=scale_dtype, device=device
        )
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )
    moe_weights["shared_experts.gate_proj.weight"] = (
        (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.gate_proj.weight_scale"] = (
        torch.rand(
            MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype, device=device
        )
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )
    moe_weights["shared_experts.up_proj.weight"] = (
        (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.up_proj.weight_scale"] = (
        torch.rand(
            MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype, device=device
        )
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )

    return moe_weights


@pytest.fixture
def moe_weights_nvfp4() -> dict[str, torch.Tensor]:
    """Generate NVFP4 weights on GPU for fast random number generation."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fp4_scale_min = 50.0
    fp4_scale_max = 150.0

    def _add_fp4_proj(
        moe_weights: dict[str, torch.Tensor],
        prefix: str,
        out_dim: int,
        in_dim: int,
        weight_scale_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = torch.randint(
            0,
            256,
            (out_dim, in_dim // 2),
            dtype=torch.uint8,
            device=device,
        )
        weight_scale = (
            torch.rand(
                out_dim,
                weight.shape[1] // 8,
                dtype=torch.float32,
                device=device,
            )
            * (fp4_scale_max - fp4_scale_min)
            + fp4_scale_min
        ).to(torch.float8_e4m3fn)

        if weight_scale_2 is None:
            weight_scale_2 = (
                torch.rand((), dtype=torch.float32, device=device) * 1e-4
            )

        moe_weights[f"{prefix}.weight"] = weight
        moe_weights[f"{prefix}.weight_scale"] = weight_scale
        moe_weights[f"{prefix}.weight_scale_2"] = weight_scale_2
        moe_weights[f"{prefix}.input_scale"] = torch.ones(
            (), dtype=torch.float32, device=device
        )
        return weight_scale_2

    moe_weights = {}

    # Gate weights for router
    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    )

    # Individual expert weights -- gate_proj and up_proj share weight_scale_2.
    for expert_idx in range(NUM_EXPERTS):
        gate_up_scale_2 = _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.gate_proj",
            MOE_DIM,
            HIDDEN_DIM,
        )
        _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.up_proj",
            MOE_DIM,
            HIDDEN_DIM,
            weight_scale_2=gate_up_scale_2,
        )
        _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.down_proj",
            HIDDEN_DIM,
            MOE_DIM,
        )

    # Shared experts weights -- gate_proj and up_proj share weight_scale_2.
    shared_gate_up_scale_2 = _add_fp4_proj(
        moe_weights,
        "shared_experts.gate_proj",
        MOE_DIM,
        HIDDEN_DIM,
    )
    _add_fp4_proj(
        moe_weights,
        "shared_experts.up_proj",
        MOE_DIM,
        HIDDEN_DIM,
        weight_scale_2=shared_gate_up_scale_2,
    )
    _add_fp4_proj(
        moe_weights,
        "shared_experts.down_proj",
        HIDDEN_DIM,
        MOE_DIM,
    )

    return moe_weights


@pytest.fixture
def moe_weights_mxfp4() -> dict[str, torch.Tensor]:
    """Generate MXFP4 weights on GPU for fast random number generation."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _add_fp4_proj(
        moe_weights: dict[str, torch.Tensor],
        prefix: str,
        out_dim: int,
        in_dim: int,
    ) -> None:
        weight = torch.randint(
            0,
            256,
            (out_dim, in_dim // 2),
            dtype=torch.uint8,
            device=device,
        )
        weight_scale = (
            # torch.rand(
            torch.ones(
                out_dim,
                weight.shape[1] // 16,
                dtype=torch.float32,
                device=device,
            )
            * WEIGHTS_STDDEV
        ).to(torch.float8_e8m0fnu)

        moe_weights[f"{prefix}.weight"] = weight
        moe_weights[f"{prefix}.weight_scale"] = weight_scale

    moe_weights = {}

    # Gate weights for router
    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    )

    for expert_idx in range(NUM_EXPERTS):
        _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.gate_proj",
            MOE_DIM,
            HIDDEN_DIM,
        )
        _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.up_proj",
            MOE_DIM,
            HIDDEN_DIM,
        )
        _add_fp4_proj(
            moe_weights,
            f"experts.{expert_idx}.down_proj",
            HIDDEN_DIM,
            MOE_DIM,
        )

    _add_fp4_proj(
        moe_weights,
        "shared_experts.gate_proj",
        MOE_DIM,
        HIDDEN_DIM,
    )
    _add_fp4_proj(
        moe_weights,
        "shared_experts.up_proj",
        MOE_DIM,
        HIDDEN_DIM,
    )
    _add_fp4_proj(
        moe_weights,
        "shared_experts.down_proj",
        HIDDEN_DIM,
        MOE_DIM,
    )

    return moe_weights


@pytest.fixture
def moe_weights_mxfp8() -> dict[str, torch.Tensor]:
    """Generate MXFP8 weights on GPU for NVIDIA EP tests."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fp8_dtype = torch.float8_e4m3fn
    scale_dtype = torch.float8_e8m0fnu
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = torch.finfo(fp8_dtype).min
    weights_multiplier = 100
    scale_generator = torch.Generator(device="cpu")
    scale_generator.manual_seed(2026)

    def _add_mxfp8_proj(
        moe_weights: dict[str, torch.Tensor],
        prefix: str,
        out_dim: int,
        in_dim: int,
    ) -> None:
        weight = (
            torch.randn(
                out_dim,
                in_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            * weights_multiplier
        ).clamp(fp8_min, fp8_max)
        scale_bits = torch.randint(
            109,
            117,
            (out_dim, in_dim // 32),
            dtype=torch.uint8,
            device="cpu",
            generator=scale_generator,
        )
        moe_weights[f"{prefix}.weight"] = weight.to(fp8_dtype)
        moe_weights[f"{prefix}.weight_scale"] = scale_bits.view(scale_dtype).to(
            device
        )

    moe_weights = {}

    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    )

    for expert_idx in range(NUM_EXPERTS):
        _add_mxfp8_proj(
            moe_weights,
            f"experts.{expert_idx}.gate_proj",
            MOE_DIM,
            HIDDEN_DIM,
        )
        _add_mxfp8_proj(
            moe_weights,
            f"experts.{expert_idx}.up_proj",
            MOE_DIM,
            HIDDEN_DIM,
        )
        _add_mxfp8_proj(
            moe_weights,
            f"experts.{expert_idx}.down_proj",
            HIDDEN_DIM,
            MOE_DIM,
        )

    _add_mxfp8_proj(
        moe_weights,
        "shared_experts.gate_proj",
        MOE_DIM,
        HIDDEN_DIM,
    )
    _add_mxfp8_proj(
        moe_weights,
        "shared_experts.up_proj",
        MOE_DIM,
        HIDDEN_DIM,
    )
    _add_mxfp8_proj(
        moe_weights,
        "shared_experts.down_proj",
        HIDDEN_DIM,
        MOE_DIM,
    )

    return moe_weights


@pytest.fixture(scope="module")
def moe_weights() -> dict[str, torch.Tensor]:
    """Generate random BF16 weights on GPU for fast random number generation."""
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    moe_weights: dict[str, torch.Tensor] = {}

    moe_weights["gate.gate_score.weight"] = (
        torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * WEIGHTS_STDDEV
    )

    for expert_idx in range(NUM_EXPERTS):
        moe_weights[f"experts.{expert_idx}.gate_proj.weight"] = (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * WEIGHTS_STDDEV
        )
        moe_weights[f"experts.{expert_idx}.up_proj.weight"] = (
            torch.randn(
                MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device
            )
            * WEIGHTS_STDDEV
        )
        moe_weights[f"experts.{expert_idx}.down_proj.weight"] = (
            torch.randn(
                HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16, device=device
            )
            * WEIGHTS_STDDEV
        )

    moe_weights["shared_experts.gate_proj.weight"] = (
        torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
        * WEIGHTS_STDDEV
    )
    moe_weights["shared_experts.up_proj.weight"] = (
        torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
        * WEIGHTS_STDDEV
    )
    moe_weights["shared_experts.down_proj.weight"] = (
        torch.randn(HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16, device=device)
        * WEIGHTS_STDDEV
    )

    return moe_weights


def _build_compiled_ep_models(
    moe_weights: dict[str, torch.Tensor],
    swiglu_limit: float = 0.0,
) -> CompiledEPModels | None:
    """Compile MoE and MoEGate graphs once.

    Returns None when hardware requirements are not met.
    """
    if accelerator_count() < N_DEVICES:
        return None

    n_devices = N_DEVICES
    max_tokens_per_rank = 128
    dtype = DType.bfloat16

    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=dtype,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
    )

    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)

    moe = MoE(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=n_devices,
        dtype=dtype,
        apply_router_weight_first=False,
        gated_activation_fn=make_concatenated_gated_activation_fn(
            ops.silu, swiglu_limit
        )
        if swiglu_limit > 0
        else None,
        ep_batch_manager=ep_batch_manager,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)
    moe_weights_cpu = {k: v.cpu() for k, v in moe_weights.items()}
    moe.load_state_dict(moe_weights_cpu)

    ep_comm_init.ep_init(session)

    per_device_input_types: list[TensorType] = [
        TensorType(
            DType.bfloat16,
            (f"input_len_{i}", HIDDEN_DIM),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]

    with Graph(
        "EPMoE",
        input_types=[
            *per_device_input_types,
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        inputs_tensors = [x.tensor for x in graph.inputs[:n_devices]]
        ep_batch_manager.fetch_buffers(graph.inputs[n_devices:])
        outputs = forward_moe_sharded_layers(moe_shards, inputs_tensors)
        graph.output(*outputs)

    moe_model = session.load(graph, weights_registry=moe.state_dict())

    moe_gate = MoEGate(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        dtype=dtype,
    )
    moe_gate.sharding_strategy = ShardingStrategy.replicate(n_devices)
    moe_gate_shards = moe_gate.shard(devices_ref)

    gate_weight_dict = {
        "gate_score.weight": moe_weights_cpu["gate.gate_score.weight"]
    }
    moe_gate.load_state_dict(gate_weight_dict)

    with Graph(
        "MoEGate",
        input_types=per_device_input_types,
    ) as gate_graph:
        gate_inputs = [x.tensor for x in gate_graph.inputs[:n_devices]]
        gate_outputs: list[TensorValue] = []
        for moe_gate_shard, inp in zip(
            moe_gate_shards, gate_inputs, strict=False
        ):
            gate_outputs.extend(moe_gate_shard(inp))
        gate_graph.output(*gate_outputs)

    gate_model = session.load(
        gate_graph, weights_registry=moe_gate.state_dict()
    )

    return CompiledEPModels(
        moe_model=moe_model,
        gate_model=gate_model,
        ep_comm_init=ep_comm_init,
        devices=devices,
        swiglu_limit=swiglu_limit,
    )


@pytest.fixture(scope="module")
def compiled_ep_models(
    moe_weights: dict[str, torch.Tensor],
) -> CompiledEPModels | None:
    """Compile MoE and MoEGate graphs once, shared across parametrized runs.

    Returns None when hardware requirements are not met.
    """
    return _build_compiled_ep_models(moe_weights)


@pytest.fixture(scope="module")
def compiled_ep_models_swiglu(
    moe_weights: dict[str, torch.Tensor],
) -> CompiledEPModels | None:
    """Compile MoE and MoEGate graphs with swiglu_limit enabled.

    Returns None when hardware requirements are not met.
    """
    return _build_compiled_ep_models(moe_weights, swiglu_limit=10.0)


def _build_compiled_allreduce_ep_models(
    moe_weights: dict[str, torch.Tensor],
) -> CompiledAllreduceEPModels | None:
    """Compile MoE and MoEGate graphs with ``use_allreduce=True``.

    The graph takes a single input on device 0, broadcasts it to all devices,
    runs per-device EP dispatch/combine, and collects results via allreduce.

    Returns None when hardware requirements are not met.
    """
    if accelerator_count() < N_DEVICES:
        return None

    n_devices = N_DEVICES
    max_tokens_per_rank = 128
    dtype = DType.bfloat16

    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=dtype,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=1,
        fused_shared_expert=True,
        use_allreduce=True,
    )

    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)

    signals = Signals(devices=devices_ref)

    moe = MoE(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=n_devices,
        dtype=dtype,
        apply_router_weight_first=False,
        ep_batch_manager=ep_batch_manager,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)
    moe_weights_cpu = {k: v.cpu() for k, v in moe_weights.items()}
    moe.load_state_dict(moe_weights_cpu)

    ep_comm_init.ep_init(session)

    # Single input on device 0.
    input_type = TensorType(
        DType.bfloat16, ("input_len", HIDDEN_DIM), DeviceRef.GPU(0)
    )
    ep_input_types = ep_batch_manager.input_types()
    signal_input_types = signals.input_types()

    with Graph(
        "EPMoEAllreduce",
        input_types=[input_type, *ep_input_types, *signal_input_types],
    ) as graph:
        input_tensor = graph.inputs[0].tensor

        ep_offset = 1
        ep_batch_manager.fetch_buffers(
            graph.inputs[ep_offset : ep_offset + len(ep_input_types)]
        )

        signal_offset = ep_offset + len(ep_input_types)
        signal_bufs = [inp.buffer for inp in graph.inputs[signal_offset:]]

        broadcast_outputs = ops.distributed_broadcast(input_tensor, signal_bufs)
        moe_outputs = forward_moe_sharded_layers(moe_shards, broadcast_outputs)
        allreduce_outputs = ops.allreduce.sum(moe_outputs, signal_bufs)

        graph.output(*allreduce_outputs)

    moe_model = session.load(graph, weights_registry=moe.state_dict())

    # Gate graph: single input on device 0 -> broadcast -> per-device gate.
    moe_gate = MoEGate(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        dtype=dtype,
    )
    moe_gate.sharding_strategy = ShardingStrategy.replicate(n_devices)
    moe_gate_shards = moe_gate.shard(devices_ref)

    gate_weight_dict = {
        "gate_score.weight": moe_weights_cpu["gate.gate_score.weight"]
    }
    moe_gate.load_state_dict(gate_weight_dict)

    with Graph(
        "MoEGateAllreduce",
        input_types=[input_type, *signal_input_types],
    ) as gate_graph:
        gate_input = gate_graph.inputs[0].tensor
        gate_signal_bufs = [inp.buffer for inp in gate_graph.inputs[1:]]

        gate_broadcasts = ops.distributed_broadcast(
            gate_input, gate_signal_bufs
        )
        gate_outputs: list[TensorValue] = []
        for moe_gate_shard, inp in zip(
            moe_gate_shards, gate_broadcasts, strict=False
        ):
            gate_outputs.extend(moe_gate_shard(inp))
        gate_graph.output(*gate_outputs)

    gate_model = session.load(
        gate_graph, weights_registry=moe_gate.state_dict()
    )

    return CompiledAllreduceEPModels(
        moe_model=moe_model,
        gate_model=gate_model,
        ep_comm_init=ep_comm_init,
        signal_buffers=signals.buffers(),
        devices=devices,
    )


@pytest.fixture(scope="module")
def compiled_allreduce_ep_models(
    moe_weights: dict[str, torch.Tensor],
) -> CompiledAllreduceEPModels | None:
    """Compile allreduce EP MoE and MoEGate graphs once.

    Returns None when hardware requirements are not met.
    """
    return _build_compiled_allreduce_ep_models(moe_weights)
