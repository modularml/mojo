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

"""Graph-level test for the rowwise / per-token grouped FP8 matmul dispatch.

Target: NVIDIA SM100 (B200). This validates the *Python dispatch path* added to
``quantized_grouped_matmul`` for the compressed-tensors FP8-dynamic layout
(per-output-channel weight scale + per-token dynamic activation scale, the
``RedHatAI/Llama-4-Scout-...-FP8-dynamic`` layout). The kernel numerics
themselves are covered by ``test_grouped_matmul_rowwise_scaled_fp8.mojo``; here
we check that ``quantized_grouped_matmul`` quantizes the activation per token,
gets the weight/scale orientation right, and produces the expected output
against a NumPy fp8 reference.
"""

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.quant_ops import quantized_grouped_matmul
from torch.utils.dlpack import from_dlpack

# A small Llama-4-Scout-ish gate/up shape: K (hidden) -> N (2 * moe_dim).
K = 256
N = 512
NUM_EXPERTS = 4


def _quantize_fp8_e4m3(
    x: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (along ``axis``) dynamic fp8_e4m3 quantization, NumPy reference.

    Returns the fp8-cast values (as float32 for easy dequant) and the float32
    scales. Scale = amax / 448, value = round_to_e4m3(x / scale).
    """
    FP8_MAX = 448.0
    amax = np.maximum(np.abs(x).max(axis=axis, keepdims=True), 1e-12)
    scale = (amax / FP8_MAX).astype(np.float32)
    q = x / scale
    # Round-trip through torch's e4m3 to match the kernel's rounding.
    q_fp8 = (
        torch.from_numpy(q.astype(np.float32))
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .numpy()
    )
    return q_fp8, scale


def test_grouped_matmul_rowwise_fp8_dispatch() -> None:
    """Drive ``quantized_grouped_matmul`` through the rowwise FP8 branch."""
    rng = np.random.default_rng(0)

    quant_config = QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,  # per-token activation
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE,  # per-output-channel weight
            dtype=DType.float32,
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
    )
    # The rowwise branch requires no block_size on either spec.
    assert quant_config.input_scale.block_size is None
    assert quant_config.weight_scale.is_rowwise

    # Two active experts in a ragged batch: expert 1 gets 5 tokens, expert 3
    # gets 3 tokens (sparse expert ids, exercises the routing wiring).
    tokens_per_expert = [5, 3]
    active_expert_ids = [1, 3]
    total_tokens = sum(tokens_per_expert)

    x = (rng.standard_normal((total_tokens, K)) * 0.5).astype(np.float32)

    # Weight is stored [E, K, N] (in, out) -- the same storage layout
    # StackedMLP allocates; quantized_grouped_matmul transposes it to [E, N, K].
    w_bf16 = (rng.standard_normal((NUM_EXPERTS, K, N)) * 0.05).astype(
        np.float32
    )
    # Per-output-channel (rowwise) weight quant: one scale per (expert, N).
    # Quantize along the K axis (axis=1) so each output channel n gets a scale.
    w_fp8 = np.empty_like(w_bf16, dtype=np.float32)
    w_scale = np.empty((NUM_EXPERTS, N, 1), dtype=np.float32)
    for e in range(NUM_EXPERTS):
        # w_bf16[e] is [K, N]; we want a per-N scale -> quantize along K (axis 0).
        q, s = _quantize_fp8_e4m3(w_bf16[e], axis=0)  # s: [1, N]
        w_fp8[e] = q
        w_scale[e, :, 0] = s[0]

    # Reference per-token activation quant (matches kernel's per-token path).
    x_fp8_ref, x_scale_ref = _quantize_fp8_e4m3(x, axis=1)  # x_scale: [T, 1]

    # NumPy fp8 reference: out[t, n] = sum_k a_fp8[t,k]*b_fp8[e,n,k]
    #                                  * a_scale[t] * b_scale[e,n]
    expert_of_token = np.concatenate(
        [
            np.full(tokens_per_expert[i], active_expert_ids[i], dtype=np.int64)
            for i in range(len(active_expert_ids))
        ]
    )
    ref = np.empty((total_tokens, N), dtype=np.float32)
    for t in range(total_tokens):
        e = int(expert_of_token[t])
        # w_fp8[e] is [K, N]; matmul a_fp8[t] (K,) with it -> (N,)
        acc = x_fp8_ref[t] @ w_fp8[e]  # (N,)
        ref[t] = acc * x_scale_ref[t, 0] * w_scale[e, :, 0]

    # Graph: feed bf16 x + fp8 weight + fp8 weight-scale, let
    # quantized_grouped_matmul do the activation quant + dispatch.
    device = Accelerator()
    session = InferenceSession(devices=[device])

    x_type = TensorType(
        DType.float32, [total_tokens, K], device=DeviceRef.GPU()
    )
    w_type = TensorType(
        DType.float8_e4m3fn, [NUM_EXPERTS, K, N], device=DeviceRef.GPU()
    )
    w_scale_type = TensorType(
        DType.float32, [NUM_EXPERTS, N, 1], device=DeviceRef.GPU()
    )
    # In production these come from ``moe_create_indices``, which sizes
    # ``expert_ids`` to the full local expert count [num_experts] and
    # ``expert_start_indices`` to [num_experts + 1]; the first
    # ``num_active_experts`` group slots are active and packed at the front,
    # the rest are padding (empty token ranges). The kernel only iterates the
    # first ``num_active_experts`` groups.
    start_type = TensorType(
        DType.uint32, [NUM_EXPERTS + 1], device=DeviceRef.GPU()
    )
    ids_type = TensorType(DType.int32, [NUM_EXPERTS], device=DeviceRef.GPU())
    usage_type = TensorType(DType.uint32, [2], device=DeviceRef.GPU())

    with Graph(
        "rowwise_fp8_grouped_matmul",
        input_types=(
            x_type,
            w_type,
            w_scale_type,
            start_type,
            ids_type,
            usage_type,
        ),
    ) as graph:
        x_in, w_in, ws_in, start_in, ids_in, usage_in = (
            t.tensor for t in graph.inputs
        )
        out = quantized_grouped_matmul(
            x=x_in.cast(DType.bfloat16),
            weight=w_in,
            weight_scale=ws_in,
            expert_start_indices=start_in,
            expert_ids=ids_in,
            usage_stats=usage_in,
            quant_config=quant_config,
        )
        graph.output(out.cast(DType.float32))

    compiled = session.load(graph)

    # Group slots: [active group 0 = 5 tok, active group 1 = 3 tok, pad, pad].
    # expert_start_indices is cumulative; padding slots get zero-length ranges.
    expert_start = np.array([0, 5, 8, 8, 8], dtype=np.uint32)
    # First two ids are the real active experts; pad the rest with 0.
    expert_ids = np.array(active_expert_ids + [0, 0], dtype=np.int32)
    usage = np.array(
        [max(tokens_per_expert), len(active_expert_ids)], dtype=np.uint32
    )

    def _to_dev(arr: np.ndarray, dtype: torch.dtype) -> Buffer:
        return Buffer.from_dlpack(torch.from_numpy(arr).to(dtype).cuda()).to(
            device
        )

    result = compiled.execute(
        _to_dev(x, torch.float32),
        Buffer.from_dlpack(
            torch.from_numpy(w_fp8).to(torch.float8_e4m3fn).cuda()
        ).to(device),
        _to_dev(w_scale, torch.float32),
        _to_dev(expert_start, torch.uint32),
        _to_dev(expert_ids, torch.int32),
        _to_dev(usage, torch.uint32),
    )
    out_np = from_dlpack(result[0]).to(torch.float32).cpu().numpy()

    assert out_np.shape == (total_tokens, N)
    assert np.all(np.isfinite(out_np))

    # bf16-output + fp8 reductions: loose tolerance, plus cosine check to catch
    # any orientation transpose mistake (which would tank cosine).
    cos = np.sum(out_np * ref) / (
        np.linalg.norm(out_np) * np.linalg.norm(ref) + 1e-12
    )
    assert cos > 0.99, f"cosine {cos} too low -- likely an orientation bug"
    np.testing.assert_allclose(out_np, ref, rtol=0.05, atol=0.05)
