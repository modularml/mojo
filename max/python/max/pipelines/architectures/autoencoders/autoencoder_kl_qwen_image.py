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

"""QwenImage VAE (encoder + decoder) for T=1 image generation.

The QwenImage VAE (Wan-2.1 architecture) uses 3D causal convolutions. For
single-image generation (T=1), these reduce to 2D convolutions: with causal
temporal padding on T=1 input, only the last temporal kernel slice contributes
non-zero values. This module implements the decoder using Conv2d with weights
extracted from the last temporal slice of the 3D kernels.

Weight transformations in load_model():
- 5D conv [O, I, D, H, W] -> 4D [O, I, H, W] (last temporal slice)
- Norm gamma [C, 1, 1, 1] or [C, 1, 1] -> [C]
- Fused to_qkv -> split into separate to_q, to_k, to_v
- time_conv weights -> skipped (irrelevant for T=1)
"""

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.weights import WeightData, Weights
from max.nn.conv import Conv2d
from max.nn.layer import LayerList, Module
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.modeling.base.component_model import ComponentModel

from .model_config import (
    AutoencoderKLQwenImageConfig,
    AutoencoderKLQwenImageConfigBase,
)


def _interpolate_2d_nearest(
    x: TensorValue,
    *,
    scale_factor: int = 2,
) -> TensorValue:
    """Upsample an NCHW tensor using nearest-neighbor expansion."""
    if x.rank != 4:
        raise ValueError(f"Input tensor must have rank 4, got {x.rank}")
    if scale_factor != 2:
        raise NotImplementedError(
            f"Only scale_factor=2 is supported, got {scale_factor}"
        )

    n, c, h, w = x.shape
    x_reshaped = ops.reshape(x, [n, c, h, 1, w, 1])
    ones = ops.broadcast_to(
        ops.constant(1.0, dtype=x.dtype, device=x.device),
        [1, 1, 1, scale_factor, 1, scale_factor],
    )
    x_expanded = x_reshaped * ones
    return ops.reshape(x_expanded, [n, c, h * scale_factor, w * scale_factor])


class NCHWRMSNorm(Module):
    """RMS normalization with learnable gamma for NCHW tensors.

    The Wan VAE RMS norm computes mean(x^2) over the channel dimension.
    We permute to channels-last, apply standard rms_norm (over last dim),
    and permute back.

    HF weight key produces: {name}.gamma
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = Weight(
            "gamma",
            dtype or DType.float32,
            [dim],
            device=device or DeviceRef.CPU(),
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x_perm = ops.permute(x, [0, 2, 3, 1])
        gamma = self.gamma.cast(x_perm.dtype)
        if x_perm.device:
            gamma = gamma.to(x_perm.device)
        x_normed = ops.rms_norm(
            x_perm,
            gamma,
            self.eps,
            weight_offset=0.0,
            multiply_before_cast=True,
        )
        return ops.permute(x_normed, [0, 3, 1, 2])


class ResBlock(Module):
    """Residual block with RMS norm and Conv2d.

    HF keys: norm1.gamma, conv1.{weight,bias}, norm2.gamma, conv2.{weight,bias}
    Optional: conv_shortcut.{weight,bias} when in_ch != out_ch.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        eps: float = 1e-6,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.norm1 = NCHWRMSNorm(in_ch, eps, dtype=dtype, device=device)
        self.conv1 = Conv2d(
            kernel_size=3,
            in_channels=in_ch,
            out_channels=out_ch,
            dtype=dtype,
            padding=1,
            device=device,
            has_bias=True,
            permute=True,
        )
        self.norm2 = NCHWRMSNorm(out_ch, eps, dtype=dtype, device=device)
        self.conv2 = Conv2d(
            kernel_size=3,
            in_channels=out_ch,
            out_channels=out_ch,
            dtype=dtype,
            padding=1,
            device=device,
            has_bias=True,
            permute=True,
        )
        self.conv_shortcut: Conv2d | None = None
        if in_ch != out_ch:
            self.conv_shortcut = Conv2d(
                kernel_size=1,
                in_channels=in_ch,
                out_channels=out_ch,
                dtype=dtype,
                padding=0,
                device=device,
                has_bias=True,
                permute=True,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        h = ops.silu(self.norm1(x))
        h = self.conv1(h)
        h = ops.silu(self.norm2(h))
        h = self.conv2(h)
        return h + shortcut


class Attention(Module):
    """Self-attention for VAE mid-block using 1x1 Conv2d.

    HF has fused to_qkv; we split into to_q/to_k/to_v during weight loading.
    HF keys: norm.gamma, to_qkv.{weight,bias} (split), proj.{weight,bias}
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self._dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        self.norm = NCHWRMSNorm(dim, eps, dtype=dtype, device=device)
        self.to_q = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            permute=True,
        )
        self.to_k = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            permute=True,
        )
        self.to_v = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            permute=True,
        )
        self.proj = Conv2d(
            kernel_size=1,
            in_channels=dim,
            out_channels=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            permute=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        residual = x
        x = self.norm(x)

        n, c, h, w = x.shape
        seq_len = h * w

        # Apply 1x1 convs for Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape [B, C, H, W] -> [B, H*W, C] for attention
        q = ops.permute(ops.reshape(q, [n, c, seq_len]), [0, 2, 1])
        k = ops.permute(ops.reshape(k, [n, c, seq_len]), [0, 2, 1])
        v = ops.permute(ops.reshape(v, [n, c, seq_len]), [0, 2, 1])

        # Scaled dot-product attention (single-head)
        attn = q @ ops.permute(k, [0, 2, 1]) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back [B, H*W, C] -> [B, C, H, W]
        out = ops.reshape(ops.permute(out, [0, 2, 1]), [n, c, h, w])

        # Output projection
        out = self.proj(out)

        return residual + out


class Interpolate2D(Module):
    """2x nearest-neighbor interpolation with no learnable parameters.

    Used as index 0 of the upsampler's resample layer list (Conv2d at index 1).
    This ensures weight keys match HF: resample.1.{weight,bias}.
    """

    def __call__(self, x: TensorValue) -> TensorValue:
        return _interpolate_2d_nearest(x, scale_factor=2)


class ZeroPadBottomRight2D(Module):
    """Pad right and bottom by 1 pixel (matches HF ZeroPad2d((0,1,0,1)))."""

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.pad(x, paddings=[0, 0, 0, 0, 0, 1, 0, 1], value=0)


class Upsampler(Module):
    """Spatial upsampler: 2x interpolation then Conv2d.

    HF keys: resample.0 (no weights), resample.1.{weight,bias}
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.resample = LayerList(
            [
                Interpolate2D(),
                Conv2d(
                    kernel_size=3,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    dtype=dtype,
                    padding=1,
                    device=device,
                    has_bias=True,
                    permute=True,
                ),
            ]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.resample[0](x)
        x = self.resample[1](x)
        return x


class Downsampler(Module):
    """Spatial downsampler with key layout matching HF resample.1.* weights."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.resample = LayerList(
            [
                ZeroPadBottomRight2D(),
                Conv2d(
                    kernel_size=3,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    dtype=dtype,
                    stride=2,
                    padding=0,
                    device=device,
                    has_bias=True,
                    permute=True,
                ),
            ]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.resample[0](x)
        x = self.resample[1](x)
        return x


class MidBlock(Module):
    """Mid block: ResBlock -> Attention -> ResBlock.

    HF keys: resnets.{0,1}.*, attentions.0.*
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.resnets = LayerList(
            [
                ResBlock(dim, dim, eps, dtype=dtype, device=device),
                ResBlock(dim, dim, eps, dtype=dtype, device=device),
            ]
        )
        self.attentions = LayerList(
            [Attention(dim, eps, dtype=dtype, device=device)]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class UpBlock(Module):
    """Up block: ResBlocks then optional upsampler.

    HF keys: resnets.{0,1,2}.*, upsamplers.0.resample.1.*
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_resnets: int,
        upsample_out_ch: int | None = None,
        eps: float = 1e-6,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        resnets = []
        for i in range(num_resnets):
            res_in = in_ch if i == 0 else out_ch
            resnets.append(
                ResBlock(res_in, out_ch, eps, dtype=dtype, device=device)
            )
        self.resnets = LayerList(resnets)

        self.upsamplers: LayerList | None = None
        if upsample_out_ch is not None:
            self.upsamplers = LayerList(
                [
                    Upsampler(
                        out_ch,
                        upsample_out_ch,
                        dtype=dtype,
                        device=device,
                    )
                ]
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class DownBlock(Module):
    """Down block: ResBlocks then optional downsampler."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_resnets: int,
        add_downsample: bool,
        eps: float = 1e-6,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        resnets = []
        for i in range(num_resnets):
            res_in = in_ch if i == 0 else out_ch
            resnets.append(
                ResBlock(res_in, out_ch, eps, dtype=dtype, device=device)
            )
        self.resnets = LayerList(resnets)

        self.downsamplers: LayerList | None = None
        if add_downsample:
            self.downsamplers = LayerList(
                [Downsampler(out_ch, out_ch, dtype=dtype, device=device)]
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class QwenImageEncoder3d(Module):
    """QwenImage VAE encoder for T=1 image conditioning."""

    def __init__(self, config: AutoencoderKLQwenImageConfigBase):
        dims = [config.base_dim * m for m in config.dim_mult]
        num_levels = len(dims)

        self._z_dim = config.z_dim
        self._dtype = config.dtype
        self._device = DeviceRef.from_device(config.device)

        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=3,
            out_channels=dims[0],
            dtype=self._dtype,
            padding=1,
            device=self._device,
            has_bias=True,
            permute=True,
        )

        down_blocks: list[DownBlock] = []
        prev_ch = dims[0]
        for i, block_ch in enumerate(dims):
            down_blocks.append(
                DownBlock(
                    in_ch=prev_ch,
                    out_ch=block_ch,
                    num_resnets=config.num_res_blocks,
                    add_downsample=i < (num_levels - 1),
                    dtype=self._dtype,
                    device=self._device,
                )
            )
            prev_ch = block_ch

        # Flatten to match HF key structure:
        # encoder.down_blocks.{i} where each entry is either a resblock or downsampler.
        flat_down_blocks: list[Module] = []
        for block in down_blocks:
            assert isinstance(block, DownBlock)
            for resnet in block.resnets:
                flat_down_blocks.append(resnet)
            if block.downsamplers is not None:
                flat_down_blocks.append(cast(Module, block.downsamplers[0]))

        self.down_blocks = LayerList(flat_down_blocks)

        deepest = dims[-1]
        self.mid_block = MidBlock(
            deepest, dtype=self._dtype, device=self._device
        )
        self.norm_out = NCHWRMSNorm(
            deepest, dtype=self._dtype, device=self._device
        )
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=deepest,
            out_channels=2 * config.z_dim,
            dtype=self._dtype,
            padding=1,
            device=self._device,
            has_bias=True,
            permute=True,
        )
        self.quant_conv = Conv2d(
            kernel_size=1,
            in_channels=2 * config.z_dim,
            out_channels=2 * config.z_dim,
            dtype=self._dtype,
            device=self._device,
            has_bias=True,
            permute=True,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", 3, "height", "width"],
                device=self._device,
            ),
        )

    def __call__(self, image: TensorValue) -> TensorValue:
        h = self.conv_in(image)
        for down_block in self.down_blocks:
            h = down_block(h)
        h = self.mid_block(h)
        h = self.norm_out(h)
        h = ops.silu(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class QwenImageDecoder3d(Module):
    """QwenImage VAE decoder for T=1 image generation.

    Converts latent [B, 16, H, W] to image [B, 3, 8H, 8W] via:
    post_quant_conv -> conv_in -> mid_block -> 4 up_blocks -> norm_out -> conv_out

    For base_dim=96, dim_mult=[1,2,4,4], the channel flow is:
    16 -> 384 (conv_in) -> 384 (mid_block) -> up_blocks:
      [384->384, upsample->192] -> [192->384, upsample->192] ->
      [192->192, upsample->96] -> [96->96] -> 3 (conv_out)
    """

    def __init__(self, config: AutoencoderKLQwenImageConfigBase):
        dims = [config.base_dim * m for m in config.dim_mult]
        num_levels = len(dims)
        num_resnets = config.num_res_blocks + 1
        deepest = dims[-1]

        self._z_dim = config.z_dim
        self._dtype = config.dtype
        self._device = DeviceRef.from_device(config.device)

        # Post-quantization 1x1 conv (from top-level VAE, routed to decoder)
        self.post_quant_conv = Conv2d(
            kernel_size=1,
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            dtype=self._dtype,
            device=self._device,
            has_bias=True,
            permute=True,
        )

        # Input projection: z_dim -> deepest
        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=config.z_dim,
            out_channels=deepest,
            dtype=self._dtype,
            padding=1,
            device=self._device,
            has_bias=True,
            permute=True,
        )

        # Mid block
        self.mid_block = MidBlock(
            deepest, dtype=self._dtype, device=self._device
        )

        # Up blocks: compute channel configs from dim_mult
        dims_reversed = list(reversed(dims))
        up_blocks = []
        prev_ch = deepest

        for i in range(num_levels):
            block_ch = dims_reversed[i]

            # Determine upsample output channels (skip to next different dim)
            upsample_out = None
            if i < num_levels - 1:
                for k in range(i + 1, num_levels):
                    if dims_reversed[k] != block_ch:
                        upsample_out = dims_reversed[k]
                        break
                if upsample_out is None:
                    upsample_out = dims_reversed[-1]

            up_blocks.append(
                UpBlock(
                    in_ch=prev_ch,
                    out_ch=block_ch,
                    num_resnets=num_resnets,
                    upsample_out_ch=upsample_out,
                    dtype=self._dtype,
                    device=self._device,
                )
            )

            prev_ch = upsample_out if upsample_out is not None else block_ch

        self.up_blocks = LayerList(up_blocks)

        # Output
        shallowest = dims[0]
        self.norm_out = NCHWRMSNorm(
            shallowest, dtype=self._dtype, device=self._device
        )
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=shallowest,
            out_channels=3,
            dtype=self._dtype,
            padding=1,
            device=self._device,
            has_bias=True,
            permute=True,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", self._z_dim, "height", "width"],
                device=self._device,
            ),
        )

    def __call__(self, z: TensorValue) -> TensorValue:
        z = self.post_quant_conv(z)
        h = self.conv_in(z)
        h = self.mid_block(h)
        for up_block in self.up_blocks:
            h = up_block(h)
        h = self.norm_out(h)
        h = ops.silu(h)
        h = self.conv_out(h)
        return h


class AutoencoderKLQwenImage(Module):
    """QwenImage VAE wrapper for encoder + decoder."""

    def __init__(self, config: AutoencoderKLQwenImageConfigBase) -> None:
        super().__init__()
        self.encoder = QwenImageEncoder3d(config)
        self.decoder = QwenImageDecoder3d(config)

    def encode(self, image: TensorValue) -> TensorValue:
        return self.encoder(image)

    def __call__(self, z: TensorValue) -> TensorValue:
        return self.decoder(z)


def _transform_decoder_weights(
    raw_weights: dict[str, Any],
    target_dtype: DType,
) -> dict[str, Any]:
    """Transform 3D VAE weights to 2D for T=1 image generation.

    Transformations applied:
    1. 5D conv [O, I, D, H, W] -> 4D [O, I, H, W] (last temporal slice for
       causal conv, or squeeze for 1x1x1 conv)
    2. Norm gamma [C, 1, 1, 1] or [C, 1, 1] -> [C]
    3. Fused to_qkv weight/bias -> split to to_q/to_k/to_v
    4. time_conv weights -> skipped (irrelevant for T=1)
    5. Float weights -> cast to target dtype
    """
    result: dict[str, Any] = {}

    def _to_numpy(wd: WeightData) -> np.ndarray:
        """Convert WeightData to numpy, casting bfloat16 to float32."""
        if wd.dtype == DType.bfloat16:
            wd = wd.astype(DType.float32)
        return np.from_dlpack(wd.data)

    def _to_weight_data(arr: np.ndarray, name: str, dtype: DType) -> WeightData:
        """Convert numpy array to WeightData at target dtype."""
        wd = WeightData.from_numpy(np.ascontiguousarray(arr), name)
        if dtype != DType.float32:
            wd = wd.astype(dtype)
        return wd

    for key, raw_data in raw_weights.items():
        # Skip temporal convolution (irrelevant for T=1)
        if "time_conv" in key:
            continue

        # Check if this weight needs numpy transformation
        shape = tuple(raw_data.shape)
        needs_transform = (
            ".to_qkv." in key  # QKV split
            or len(shape) == 5  # 5D conv -> 4D
            or (len(shape) == 4 and shape[1:] == (1, 1, 1))  # norm gamma
            or (len(shape) == 3 and shape[1:] == (1, 1))  # norm gamma
        )

        if not needs_transform:
            # No transformation needed - pass WeightData directly
            if (
                raw_data.dtype != target_dtype
                and raw_data.dtype.is_float()
                and target_dtype.is_float()
            ):
                result[key] = raw_data.astype(target_dtype)
            else:
                result[key] = raw_data
            continue

        # Need numpy for transformation
        data = _to_numpy(raw_data)

        # Split fused QKV into separate Q, K, V
        if ".to_qkv.weight" in key:
            if data.ndim == 5:
                data = (
                    data[:, :, -1, :, :]
                    if data.shape[2] > 1
                    else data[:, :, 0, :, :]
                )
            C = data.shape[0] // 3
            prefix = key.replace(".to_qkv.weight", "")
            result[f"{prefix}.to_q.weight"] = _to_weight_data(
                data[:C], f"{prefix}.to_q.weight", target_dtype
            )
            result[f"{prefix}.to_k.weight"] = _to_weight_data(
                data[C : 2 * C], f"{prefix}.to_k.weight", target_dtype
            )
            result[f"{prefix}.to_v.weight"] = _to_weight_data(
                data[2 * C :], f"{prefix}.to_v.weight", target_dtype
            )
            continue
        if ".to_qkv.bias" in key:
            C = data.shape[0] // 3
            prefix = key.replace(".to_qkv.bias", "")
            result[f"{prefix}.to_q.bias"] = _to_weight_data(
                data[:C], f"{prefix}.to_q.bias", target_dtype
            )
            result[f"{prefix}.to_k.bias"] = _to_weight_data(
                data[C : 2 * C], f"{prefix}.to_k.bias", target_dtype
            )
            result[f"{prefix}.to_v.bias"] = _to_weight_data(
                data[2 * C :], f"{prefix}.to_v.bias", target_dtype
            )
            continue

        # Transform 5D conv weights to 4D
        if data.ndim == 5:
            if data.shape[2] == 1:
                data = data[:, :, 0, :, :]  # 1x1x1 conv -> squeeze
            else:
                data = data[:, :, -1, :, :]  # Causal conv -> last slice
        # Squeeze norm gamma
        elif (
            data.ndim == 4
            and data.shape[1] == 1
            and data.shape[2] == 1
            and data.shape[3] == 1
        ):
            data = data.reshape(data.shape[0])  # [C, 1, 1, 1] -> [C]
        elif data.ndim == 3 and data.shape[1] == 1 and data.shape[2] == 1:
            data = data.reshape(data.shape[0])  # [C, 1, 1] -> [C]

        result[key] = _to_weight_data(data, key, target_dtype)

    return result


class AutoencoderKLQwenImageModel(ComponentModel):
    """ComponentModel wrapper for QwenImage VAE.

    Handles:
    - 3D to 2D weight transformation for T=1 image generation
    - Latent normalization via latents_mean/latents_std
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
    ) -> None:
        self.latents_mean_tensor: Buffer | None = None
        self.latents_std_tensor: Buffer | None = None

        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderKLQwenImageConfig.generate(
            config, encoding, devices
        )
        self.session = session
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        target_dtype = self.config.dtype

        # Collect VAE weights by component.
        raw_encoder_weights: dict[str, Any] = {}
        raw_decoder_weights: dict[str, Any] = {}
        for key, value in self.weights.items():
            weight_data = value.data()
            if key.startswith("encoder."):
                raw_encoder_weights[key.removeprefix("encoder.")] = weight_data
            elif key.startswith("decoder."):
                raw_decoder_weights[key.removeprefix("decoder.")] = weight_data
            elif key.startswith("post_quant_conv."):
                raw_decoder_weights[key] = weight_data
            elif key.startswith("quant_conv."):
                raw_encoder_weights[key] = weight_data

        # Transform 3D weights to 2D for T=1 image generation
        encoder_state_dict = _transform_decoder_weights(
            raw_encoder_weights, target_dtype
        )
        decoder_state_dict = _transform_decoder_weights(
            raw_decoder_weights, target_dtype
        )

        autoencoder = AutoencoderKLQwenImage(self.config)
        autoencoder.encoder.load_state_dict(
            encoder_state_dict, weight_alignment=1, strict=True
        )
        autoencoder.decoder.load_state_dict(
            decoder_state_dict, weight_alignment=1, strict=True
        )

        session = self.session
        if session is None:
            session = InferenceSession(devices=self.devices)

        with Graph(
            "qwen_image_vae_encoder",
            input_types=autoencoder.encoder.input_types(),
        ) as graph:
            encoded = autoencoder.encoder(graph.inputs[0].tensor)
            graph.output(encoded[:, : self.config.z_dim, :, :])
        self.encoder: Model = session.load(
            graph,
            weights_registry=autoencoder.encoder.state_dict(),
        )

        with Graph(
            "qwen_image_vae_decoder",
            input_types=autoencoder.decoder.input_types(),
        ) as graph:
            decoded = autoencoder.decoder(graph.inputs[0].tensor)
            graph.output(decoded)
        self.model: Model = session.load(
            graph,
            weights_registry=autoencoder.decoder.state_dict(),
        )

        # Store latents_mean and latents_std as tensors on device
        if self.config.latents_mean:
            self.latents_mean_tensor = self._buffer_from_float_array(
                self.config.latents_mean,
                target_dtype=target_dtype,
            )

        if self.config.latents_std:
            self.latents_std_tensor = self._buffer_from_float_array(
                self.config.latents_std,
                target_dtype=target_dtype,
            )

        return self.model.execute

    def _buffer_from_float_array(
        self,
        values: list[float],
        *,
        target_dtype: DType,
    ) -> Buffer:
        arr = np.asarray(values, dtype=np.float32)
        if target_dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(arr)
            return (
                Buffer.from_numpy(u16)
                .to(self.devices[0])
                .view(dtype=DType.bfloat16, shape=arr.shape)
            )
        if target_dtype == DType.float16:
            arr = arr.astype(np.float16)
        return Buffer.from_dlpack(arr).to(self.devices[0])

    def encode(self, image: Buffer) -> Buffer:
        result = self.encoder.execute(image)
        return result[0] if isinstance(result, (list, tuple)) else result

    def decode(self, z: Buffer) -> Buffer:
        result = self.model.execute(z)
        return result[0] if isinstance(result, (list, tuple)) else result

    def __call__(self, z: Buffer) -> Buffer:
        return self.decode(z)
