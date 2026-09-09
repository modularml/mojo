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

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    Shape,
    ShardingStrategy,
    TensorType,
    TensorValue,
)
from max.graph.weights import WeightData
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_kernels import (
    ep_mxfp4_max_padded_m,
    uses_mx_ep_token_format,
)
from max.nn.moe import MoEGate, MoEQuantized
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_b100_b200

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64


def _dequantize_mxfp8_to_bf16(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize MXFP8 E4M3 weights to BF16 on the given device."""
    assert weight.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.float8_e8m0fnu
    scale = weight_scale.to(torch.float32).repeat_interleave(32, dim=1)
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


def _simulate_mxfp8_blockwise_quantize(x: torch.Tensor) -> torch.Tensor:
    """Simulate blockwise MXFP8 quantization on a BF16 tensor."""
    assert x.dtype == torch.bfloat16
    block_size = 32
    orig_shape = x.shape
    *batch, last = orig_shape
    assert last % block_size == 0

    x_blocks = x.reshape(*batch, last // block_size, block_size)
    abs_max = x_blocks.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_restored = (
        (abs_max / 448.0).to(torch.float8_e8m0fnu).to(torch.float32)
    )
    finfo = torch.finfo(torch.float8_e4m3fn)
    x_fp8 = (
        (x_blocks.float() / scale_restored)
        .clamp(finfo.min, finfo.max)
        .to(torch.float8_e4m3fn)
    )
    x_deq = (x_fp8.to(torch.float32) * scale_restored).to(torch.bfloat16)

    return x_deq.reshape(orig_shape)


def _torch_moe_mxfp8(
    input_token: torch.Tensor,
    moe_weights: dict[str, torch.Tensor],
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
) -> torch.Tensor:
    """Single-token MoE reference with MXFP8 quantization simulation."""
    assert input_token.shape[0] == 1

    input_token = _simulate_mxfp8_blockwise_quantize(input_token)
    top_k = topk_indices.shape[1]
    result = torch.zeros_like(input_token)

    for i in range(top_k):
        scores = topk_scores[0, i]
        expert_idx = topk_indices[0, i].item()
        gate_weight = moe_weights[f"experts.{expert_idx}.gate_proj.weight"]
        up_weight = moe_weights[f"experts.{expert_idx}.up_proj.weight"]
        down_weight = moe_weights[f"experts.{expert_idx}.down_proj.weight"]

        expert_gate = input_token @ gate_weight.T
        expert_up = input_token @ up_weight.T
        down_input = _simulate_mxfp8_blockwise_quantize(
            torch.nn.functional.silu(expert_gate) * expert_up
        )
        expert_output = down_input @ down_weight.T

        result += expert_output * scores

    shared_gate_weight = moe_weights["shared_experts.gate_proj.weight"]
    shared_up_weight = moe_weights["shared_experts.up_proj.weight"]
    shared_down_weight = moe_weights["shared_experts.down_proj.weight"]
    shared_expert_gate = input_token @ shared_gate_weight.T
    shared_expert_up = input_token @ shared_up_weight.T
    shared_down_input = _simulate_mxfp8_blockwise_quantize(
        torch.nn.functional.silu(shared_expert_gate) * shared_expert_up
    )
    shared_expert_output = shared_down_input @ shared_down_weight.T
    result += shared_expert_output

    return result


def _wrap_float8_weights(
    weights_cpu: dict[str, torch.Tensor],
) -> dict[str, WeightData | torch.Tensor]:
    """Wraps CPU float8 tensors as typed MAX weight buffers."""
    wrapped: dict[str, WeightData | torch.Tensor] = {}
    for key, value in weights_cpu.items():
        if value.dtype == torch.float8_e4m3fn:
            max_dtype = DType.float8_e4m3fn
        elif value.dtype == torch.float8_e8m0fnu:
            max_dtype = DType.float8_e8m0fnu
        else:
            wrapped[key] = value
            continue

        wrapped[key] = WeightData(
            Buffer.from_dlpack(value.view(torch.uint8)).view(max_dtype),
            key,
            max_dtype,
            Shape(value.shape),
        )
    return wrapped


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MXFP8 kernel only supports Nvidia GPUs"
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="MXFP8 kernel requires B100 or B200",
)
@pytest.mark.parametrize("n_devices", [2])
def test_ep_moe_mxfp8_nvidia(
    n_devices: int,
    moe_weights_mxfp8: dict[str, torch.Tensor],
) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    top_k = 8
    max_tokens_per_rank = 128
    dtype = DType.float8_e4m3fn

    moe_weights_cpu = {k: v.cpu() for k, v in moe_weights_mxfp8.items()}
    wrapped_moe_weights = _wrap_float8_weights(moe_weights_cpu)

    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    mxfp8_input_config = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, 32),
    )
    mxfp8_weight_config = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e8m0fnu,
        block_size=(1, 32),
    )
    mxfp8_config = QuantConfig(
        input_scale=mxfp8_input_config,
        weight_scale=mxfp8_weight_config,
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.MXFP8,
        can_use_fused_swiglu=True,
    )

    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=top_k,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
        dispatch_quant_config=mxfp8_config,
        fused_shared_expert=True,
    )

    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)

    moe = MoEQuantized(
        devices=[DeviceRef.CPU()] + devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=top_k,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=n_devices,
        dtype=dtype,
        apply_router_weight_first=False,
        ep_batch_manager=ep_batch_manager,
        quant_config=mxfp8_config,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)
    moe.load_state_dict(wrapped_moe_weights)

    ep_comm_init.ep_init(session)

    per_device_input_types: list[TensorType] = [
        TensorType(
            DType.bfloat16,
            (f"input_len_{i}", HIDDEN_DIM),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]
    input_lengths = torch.randint(1, max_tokens_per_rank, (n_devices,))
    per_device_inputs_torch = [
        torch.randn(
            input_lengths[i],
            HIDDEN_DIM,
            dtype=torch.bfloat16,
            device="cpu",
        )
        for i in range(n_devices)
    ]
    per_device_inputs = [
        Buffer.from_dlpack(input).to(devices[i])
        for i, input in enumerate(per_device_inputs_torch)
    ]

    with Graph(
        "EPMoE_MXFP8",
        input_types=[
            *per_device_input_types,
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        inputs_tensors = [x.tensor for x in graph.inputs[:n_devices]]
        ep_batch_manager.fetch_buffers(graph.inputs[n_devices:])
        outputs = forward_moe_sharded_layers(moe_shards, inputs_tensors)
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=moe.state_dict())
    result = compiled.execute(*per_device_inputs, *ep_comm_init.model_inputs())
    torch_result = [torch.from_dlpack(x).to("cpu") for x in result]
    all_outputs = torch.cat(torch_result, dim=0)
    assert not torch.any(torch.isnan(all_outputs)), (
        "MoE output should not contain NaN"
    )

    moe_gate = MoEGate(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=top_k,
        dtype=DType.bfloat16,
    )
    moe_gate.sharding_strategy = ShardingStrategy.replicate(n_devices)
    moe_gate_shards = moe_gate.shard(devices_ref)
    moe_gate.load_state_dict(
        {"gate_score.weight": moe_weights_cpu["gate.gate_score.weight"]}
    )

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

    gate_compiled = session.load(
        gate_graph, weights_registry=moe_gate.state_dict()
    )
    gate_result = gate_compiled.execute(*per_device_inputs)
    topk_idxs_weights = [torch.from_dlpack(x).to("cpu") for x in gate_result]

    gpu = moe_weights_mxfp8["gate.gate_score.weight"].device
    moe_weights_gpu: dict[str, torch.Tensor] = {}
    moe_weights_gpu["gate.gate_score.weight"] = moe_weights_mxfp8.pop(
        "gate.gate_score.weight"
    )
    weight_keys = [
        k
        for k in moe_weights_mxfp8
        if k.endswith(".weight")
        and moe_weights_mxfp8[k].dtype == torch.float8_e4m3fn
    ]
    for key in weight_keys:
        weight = moe_weights_mxfp8.pop(key)
        scale = moe_weights_mxfp8.pop(f"{key}_scale")
        moe_weights_gpu[key] = _dequantize_mxfp8_to_bf16(weight, scale)
        del weight, scale

    all_outputs = all_outputs.to(gpu)
    all_inputs = torch.cat(per_device_inputs_torch, dim=0).to(gpu)
    all_topk_idxs = torch.cat(topk_idxs_weights[::2], dim=0).to(gpu)
    all_topk_weights = torch.cat(topk_idxs_weights[1::2], dim=0).to(gpu)

    for tok_idx in range(all_inputs.shape[0]):
        torch_output = _torch_moe_mxfp8(
            all_inputs[tok_idx : tok_idx + 1],
            moe_weights_gpu,
            all_topk_idxs[tok_idx : tok_idx + 1],
            all_topk_weights[tok_idx : tok_idx + 1],
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            all_outputs[tok_idx : tok_idx + 1].float(),
            torch_output.float(),
            dim=-1,
        )
        assert cos_sim.min() > 0.98, (
            f"token {tok_idx}: cosine similarity {cos_sim.min().item():.6f}"
            " < 0.98"
        )


def _mx_quant_config(
    fmt: QuantFormat, *, preshuffled_b: bool = True
) -> QuantConfig:
    """Builds an MX (block-scaled, group 32) quant config for the fold gate."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=fmt,
        block_scaled_preshuffled_b=preshuffled_b,
        can_use_fused_swiglu=True,
    )


def _build_ep_moe(
    quant_config: QuantConfig, n_devices: int, max_tokens_per_rank: int
) -> tuple[MoEQuantized, EPConfig]:
    ep_config = EPConfig(
        dispatch_dtype=(
            DType.uint8 if quant_config.is_mxfp4 else DType.float8_e4m3fn
        ),
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=8,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
        dispatch_quant_config=quant_config,
        fused_shared_expert=True,
    )
    devices_ref = [DeviceRef.GPU(i) for i in range(n_devices)]
    moe = MoEQuantized(
        devices=[DeviceRef.CPU()] + devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=8,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=n_devices,
        dtype=ep_config.dispatch_dtype,
        apply_router_weight_first=False,
        ep_batch_manager=EPBatchManager(ep_config),
        quant_config=quant_config,
        use_swigluoai=True,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    return moe, ep_config


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="The MX EP A-scale dispatch fold is AMD-only",
)
@pytest.mark.parametrize("n_devices", [2])
def test_ep_mxfp8_scale_fold_gate_amd(n_devices: int) -> None:
    """The dispatch A-scale fold must engage for MXFP8, not just MXFP4.

    Pinned where the fold is decided -- before any weight is loaded.
    """
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )
    max_tokens_per_rank = 128
    # The fold's single source of truth for the per-expert slot stride.
    max_recv_per_expert = max_tokens_per_rank * n_devices
    expected_stride = (max_recv_per_expert + 31) // 32 * 32

    for fmt in (QuantFormat.MXFP8, QuantFormat.MXFP4):
        moe, ep_config = _build_ep_moe(
            _mx_quant_config(fmt), n_devices, max_tokens_per_rank
        )
        assert uses_mx_ep_token_format(ep_config), (
            f"{fmt} should dispatch through MXTokenFormat on AMD"
        )
        moe.configure_ep_scale_fusion(dispatch_supports_fold=True)
        assert ep_config.mxfp4_a_scales_preshuffled, (
            f"{fmt} EP A-scale fold did not engage"
        )
        assert ep_mxfp4_max_padded_m(ep_config) == expected_stride

    # Boundary: the fold still requires the preshuffled-B grouped matmul, so
    # widening the format check must not make it fire on the row-major kernel.
    moe, ep_config = _build_ep_moe(
        _mx_quant_config(QuantFormat.MXFP8, preshuffled_b=False),
        n_devices,
        max_tokens_per_rank,
    )
    moe.configure_ep_scale_fusion(dispatch_supports_fold=True)
    assert not ep_config.mxfp4_a_scales_preshuffled, (
        "the fold must stay off without a preshuffled-B matmul to read it"
    )
    assert ep_mxfp4_max_padded_m(ep_config) == 0
