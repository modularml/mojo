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

"""Wan VAE AttentionBlock harness.

Compares the MAX ``AttentionBlock`` (from ``max.pipelines.architectures.
autoencoders.vae``) against a hand-rolled torch reference that mirrors
diffusers' ``WanAttentionBlock``: single-head attention over the full
channel dim, ``scale = 1/sqrt(dim)``, residual add. Random weights are
shared between both sides so the comparison is a pure math check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from max.driver import Accelerator, Buffer, DLPackArray
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.autoencoders.vae import AttentionBlock
from torch import nn

from testbed.dtypes import DTYPE_MAP
from testbed.harness import CompiledLayerBundle, LayerTestHarness
from testbed.registry import register_harness


@dataclass
class WanVaeAttentionStaticParams:
    """Static parameters for the Wan VAE AttentionBlock harness."""

    dim: int
    dtype: str = "bfloat16"


@dataclass
class WanVaeAttentionDynamicParams:
    """Per-shape parameters for the Wan VAE AttentionBlock harness."""

    batch_size: int
    num_frames: int
    height: int
    width: int


class _TorchWanVaeAttentionBlock(nn.Module):
    """Torch reference mirroring diffusers' ``WanAttentionBlock``.

    Single-head attention over ``[B*T, H*W, dim]`` with residual add.
    Uses the same weights as the MAX layer: FCRS Conv2d filters
    (``[out, in, 1, 1]``) + biases, and an RMSNorm gamma of shape
    ``[dim]`` applied over the channel axis.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        self.gamma = nn.Parameter(torch.empty(dim))
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        identity = x

        # [B, C, T, H, W] -> [B*T, C, H, W]
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # RMSNorm(images=True, channel_first=True, eps=1e-12): normalize
        # over channel axis, multiply by gamma broadcast across H, W.
        rms = x2d.pow(2).mean(dim=1, keepdim=True)
        inv = torch.rsqrt(rms + 1e-12)
        gamma = self.gamma.view(1, c, 1, 1)
        x2d = x2d * inv * gamma

        # QKV projection (kernel_size=1 Conv2d).
        qkv = self.to_qkv(x2d)  # [B*T, 3C, H, W]

        # Flatten spatial -> sequence.
        qkv = qkv.reshape(b * t, 3 * c, h * w).transpose(1, 2)
        q = qkv[:, :, :c]
        k = qkv[:, :, c : 2 * c]
        v = qkv[:, :, 2 * c : 3 * c]

        # Single-head: [B*T, 1, seq, dim]. Torch SDPA falls back to
        # the MATH backend for head_dim>256 (no mem-efficient /
        # flash kernel), which materialises the full [B*T, 1, seq,
        # seq] attention matrix. At 720p (seq=14400, B*T=4) that's
        # ~3.3 GiB and OOMs the test budget, so process one frame
        # at a time. Per-frame peak is ~800 MiB which fits.
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        outs: list[torch.Tensor] = []
        for i in range(q.shape[0]):
            outs.append(
                F.scaled_dot_product_attention(
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    scale=self.scale,
                )
            )
        out = torch.cat(outs, dim=0)

        # Back to [B*T, C, H, W].
        out = out.squeeze(1).transpose(1, 2).reshape(b * t, c, h, w)
        out = self.proj(out)

        # [B*T, C, H, W] -> [B, C, T, H, W] + residual.
        out = out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return out + identity


@register_harness("wan_vae_attention")
class WanVaeAttentionHarness(
    LayerTestHarness[
        WanVaeAttentionStaticParams, WanVaeAttentionDynamicParams, None
    ]
):
    """Harness for the Wan VAE mid-block ``AttentionBlock``."""

    @staticmethod
    def static_params_type() -> type:
        return WanVaeAttentionStaticParams

    @staticmethod
    def dynamic_params_type() -> type:
        return WanVaeAttentionDynamicParams

    def __init__(
        self,
        static_params: WanVaeAttentionStaticParams,
        session: InferenceSession,
        device: Accelerator,
    ) -> None:
        super().__init__(static_params, session, device)
        _, torch_dtype = DTYPE_MAP[static_params.dtype]
        dim = static_params.dim

        # Reference weights live in PyTorch's native FCRS / [dim]
        # layouts. The MAX side transposes Conv2d filters to RSCF
        # inside build_graph. w_scale=0.1 is the sweet spot: large
        # enough that softmax is non-uniform (so multi-head vs
        # single-head actually produces different outputs and the
        # harness catches that regression), small enough that
        # outputs don't saturate and bf16 drift stays bounded.
        gen = torch.Generator().manual_seed(0x7A7EA11)
        w_scale = 0.1
        self._torch_weights: dict[str, torch.Tensor] = {
            "gamma": torch.randn(dim, generator=gen).to(torch_dtype),
            "to_qkv.weight": (
                torch.randn(dim * 3, dim, 1, 1, generator=gen) * w_scale
            ).to(torch_dtype),
            "to_qkv.bias": (torch.randn(dim * 3, generator=gen) * w_scale).to(
                torch_dtype
            ),
            "proj.weight": (
                torch.randn(dim, dim, 1, 1, generator=gen) * w_scale
            ).to(torch_dtype),
            "proj.bias": (torch.randn(dim, generator=gen) * w_scale).to(
                torch_dtype
            ),
        }

    @property
    def name(self) -> str:
        return "wan_vae_attention"

    def _max_state_dict(self) -> dict[str, Any]:
        """Produce the state_dict for the MAX AttentionBlock.

        The MAX ``Conv2d`` uses RSCF filter layout
        (``[kh, kw, in, out]``), while PyTorch stores Conv2d weights
        in FCRS (``[out, in, kh, kw]``); transpose accordingly. The
        RMSNorm gamma is broadcast-shaped ``[dim, 1, 1]``.
        """
        w = self._torch_weights
        dim = self.static_params.dim
        return {
            # RMSNorm(images=True, channel_first=True) gamma shape.
            "norm.gamma": w["gamma"].reshape(dim, 1, 1).contiguous(),
            # Conv2d filter: FCRS [out, in, 1, 1] -> RSCF [1, 1, in, out].
            "to_qkv.weight": w["to_qkv.weight"]
            .permute(2, 3, 1, 0)
            .contiguous(),
            "to_qkv.bias": w["to_qkv.bias"],
            "proj.weight": w["proj.weight"].permute(2, 3, 1, 0).contiguous(),
            "proj.bias": w["proj.bias"],
        }

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        max_dtype, _ = DTYPE_MAP[p.dtype]
        device_ref = DeviceRef.GPU()

        layer = AttentionBlock(dim=p.dim, dtype=max_dtype, device=device_ref)
        layer.load_state_dict(self._max_state_dict())

        input_type = TensorType(
            dtype=max_dtype,
            shape=["batch", p.dim, "frames", "height", "width"],
            device=device_ref,
        )

        with Graph("WanVaeAttention", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            graph.output(layer(x.tensor))

        return graph, layer.state_dict()

    def build_and_compile(self) -> CompiledLayerBundle:
        graph, weights_registry = self.build_graph()
        compiled = self.session.load(graph, weights_registry=weights_registry)
        return CompiledLayerBundle(
            compiled_model=compiled,
            device=self.device,
            session=self.session,
        )

    def prepare_inputs(
        self,
        bundle: CompiledLayerBundle,
        dynamic_params: WanVaeAttentionDynamicParams,
    ) -> tuple[list[Buffer], None]:
        p = self.static_params
        _, torch_dtype = DTYPE_MAP[p.dtype]
        dp = dynamic_params

        gen = torch.Generator().manual_seed(
            hash((dp.batch_size, dp.num_frames, dp.height, dp.width))
            & 0x7FFFFFFF
        )
        torch_input = torch.randn(
            dp.batch_size,
            p.dim,
            dp.num_frames,
            dp.height,
            dp.width,
            generator=gen,
            dtype=torch_dtype,
        )
        input_buf = Buffer.from_dlpack(torch_input).to(bundle.device)
        return [input_buf], None

    def cleanup_inputs(
        self, bundle: CompiledLayerBundle, context: None
    ) -> None:
        pass

    def cuda_graph_eligible(
        self, dynamic_params: WanVaeAttentionDynamicParams
    ) -> bool:
        return True

    # -------------------------------------------------------------- #
    # Correctness support
    # -------------------------------------------------------------- #

    def torch_reference_layer(self, device: str = "cuda") -> torch.nn.Module:
        p = self.static_params
        _, torch_dtype = DTYPE_MAP[p.dtype]

        ref = _TorchWanVaeAttentionBlock(dim=p.dim).to(
            device=device, dtype=torch_dtype
        )
        w = self._torch_weights
        with torch.no_grad():
            ref.gamma.copy_(w["gamma"].to(device=device, dtype=torch_dtype))
            ref.to_qkv.weight.copy_(
                w["to_qkv.weight"].to(device=device, dtype=torch_dtype)
            )
            ref.to_qkv.bias.copy_(
                w["to_qkv.bias"].to(device=device, dtype=torch_dtype)
            )
            ref.proj.weight.copy_(
                w["proj.weight"].to(device=device, dtype=torch_dtype)
            )
            ref.proj.bias.copy_(
                w["proj.bias"].to(device=device, dtype=torch_dtype)
            )
        return ref

    def prepare_torch_inputs(
        self,
        execute_args: list[Buffer],
        dynamic_params: WanVaeAttentionDynamicParams,
        device: str = "cuda",
    ) -> list[torch.Tensor]:
        return [torch.from_dlpack(execute_args[0]).to(device=device)]
