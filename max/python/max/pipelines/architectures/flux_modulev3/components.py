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

"""VAE encoder/decoder building blocks for the standalone ``flux_modulev3``.

Encoder, Decoder, and their interior blocks (``DownEncoderBlock2D``,
``UpDecoderBlock2D``, ``MidBlock2D``) used by the unified
:class:`Vae` Module.  Mirrors the equivalent block library in
``autoencoders_modulev3.vae`` but lives entirely inside this package
so the demonstration of the
single-Module-with-two-forward-methods-and-shared-weights pattern is
self-contained.
"""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Conv2d, GroupNorm, Module, ModuleList
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType

from .layers import Downsample2D, ResnetBlock2D, Upsample2D, VAEAttention


class DownEncoderBlock2D(Module[[Tensor], Tensor]):
    """Downsampling encoder block for 2D VAE.

    This module consists of multiple ResNet blocks followed by an optional
    downsampling layer. It progressively decreases spatial resolution while
    processing features through residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize DownEncoderBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_layers: Number of ResNet blocks in this encoder block.
            resnet_eps: Epsilon value for ResNet GroupNorm layers.
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            add_downsample: Whether to add downsampling layer after ResNet blocks.
            downsample_padding: Padding for the downsampling layer.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        resnets_list = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnet = ResnetBlock2D(
                in_channels=input_channels,
                out_channels=out_channels,
                groups=resnet_groups,
                groups_out=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                use_conv_shortcut=False,
                conv_shortcut_bias=True,
                device=device,
                dtype=dtype,
            )
            resnets_list.append(resnet)

        self.resnets = ModuleList(resnets_list)

        self.downsamplers: ModuleList[Downsample2D] | None = None
        if add_downsample:
            downsampler = Downsample2D(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                kernel_size=3,
                norm_type=None,
                bias=True,
                device=device,
                dtype=dtype,
            )
            self.downsamplers = ModuleList([downsampler])

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply DownEncoderBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C_in, H, W].

        Returns:
            Output tensor of shape [N, C_out, H//2, W//2] (if downsampling) or
            [N, C_out, H, W] (if no downsampling).
        """
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)

        return hidden_states


class UpDecoderBlock2D(Module[[Tensor], Tensor]):
    """Upsampling decoder block for 2D VAE.

    This module consists of multiple ResNet blocks followed by an optional
    upsampling layer. It progressively increases spatial resolution while
    processing features through residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize UpDecoderBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_layers: Number of ResNet blocks in this decoder block.
            resnet_eps: Epsilon value for ResNet GroupNorm layers.
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            add_upsample: Whether to add upsampling layer after ResNet blocks.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        resnets_list = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnet = ResnetBlock2D(
                in_channels=input_channels,
                out_channels=out_channels,
                groups=resnet_groups,
                groups_out=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                use_conv_shortcut=False,
                conv_shortcut_bias=True,
                device=device,
                dtype=dtype,
            )
            resnets_list.append(resnet)
        self.resnets = ModuleList(resnets_list)

        if add_upsample:
            upsampler = Upsample2D(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                interpolate=True,
                device=device,
                dtype=dtype,
            )
            self.upsamplers: ModuleList[Upsample2D] | None = ModuleList(
                [upsampler]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply UpDecoderBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C_in, H, W].

        Returns:
            Output tensor of shape [N, C_out, H*2, W*2] (if upsampling) or
            [N, C_out, H, W] (if no upsampling).
        """
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class MidBlock2D(Module[[Tensor], Tensor]):
    """Middle block for 2D VAE.

    This module processes features at the middle of the VAE architecture,
    applying ResNet blocks with optional spatial attention mechanisms.
    It maintains spatial dimensions while processing features through
    residual connections and self-attention.
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize MidBlock2D module.

        Args:
            in_channels: Number of input channels.
            num_layers: Number of ResNet/attention layer pairs.
            resnet_eps: Epsilon value for ResNet GroupNorm layers.
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            add_attention: Whether to interleave attention between ResNet blocks.
            attention_head_dim: Dimension of each attention head.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """

        def _resnet() -> ResnetBlock2D:
            return ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=resnet_groups,
                groups_out=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                use_conv_shortcut=False,
                conv_shortcut_bias=True,
                device=device,
                dtype=dtype,
            )

        # Structure: one leading ResNet, then ``num_layers`` (attention,
        # ResNet) pairs.  Attention is all-or-nothing per ``add_attention``.
        self.resnets = ModuleList([_resnet() for _ in range(num_layers + 1)])
        self.attentions: ModuleList[VAEAttention] | None = (
            ModuleList(
                [
                    VAEAttention(
                        query_dim=in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        num_groups=resnet_groups,
                        eps=resnet_eps,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(num_layers)
                ]
            )
            if add_attention
            else None
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply MidBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C, H, W].

        Returns:
            Output tensor of shape [N, C, H, W] with same spatial dimensions.
        """
        hidden_states = self.resnets[0](hidden_states)

        for i in range(len(self.resnets) - 1):
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states)
            hidden_states = self.resnets[i + 1](hidden_states)

        return hidden_states


class Encoder(Module[[Tensor], Tensor]):
    r"""The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    This module progressively downsamples the input through multiple encoder blocks,
    applies a middle block for feature processing, and outputs encoded latents.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        down_block_types: The types of down blocks to use. Currently only supports "DownEncoderBlock2D".
        block_out_channels: The number of output channels for each block.
        layers_per_block: The number of layers per block.
        norm_num_groups: The number of groups for normalization.
        act_fn: The activation function to use (e.g., "silu").
        double_z: Whether to double the number of output channels for the last block.
        mid_block_add_attention: Whether to add attention in the middle block.
        device: Device reference for module placement.
        dtype: Data type for module parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
        use_quant_conv: bool = False,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize Encoder module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            down_block_types: Tuple of down block types (currently only "DownEncoderBlock2D").
            block_out_channels: Tuple of block output channels.
            layers_per_block: Number of layers per block.
            norm_num_groups: Number of groups for normalization.
            act_fn: Activation function name (e.g., "silu").
            double_z: Whether to double output channels for the last block.
            mid_block_add_attention: Whether to add attention in the middle block.
            use_quant_conv: Whether to add 1x1 conv after conv_out (encoder output -> latent moments).
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype

        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.down_blocks = ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type != "DownEncoderBlock2D":
                raise ValueError(
                    f"Unsupported down_block_type: {down_block_type}. "
                    "Currently only 'DownEncoderBlock2D' is supported."
                )

            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_downsample=not is_final_block,
                downsample_padding=0,
                device=device,
                dtype=dtype,
            )
            self.down_blocks.append(down_block)

        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            device=device,
            dtype=dtype,
        )

        self.conv_norm_out = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=block_out_channels[-1],
            eps=1e-6,
            affine=True,
        )

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[-1],
            out_channels=conv_out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.quant_conv: Conv2d | None = None
        if use_quant_conv:
            self.quant_conv = Conv2d(
                kernel_size=1,
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=True,
                device=device,
                permute=True,
            )

    def forward(self, sample: Tensor) -> Tensor:
        r"""The forward method of the `Encoder` class.

        Args:
            sample: Input tensor of shape [N, C_in, H, W].

        Returns:
            Output tensor of shape [N, C_out, H_latent, W_latent] (downsampled).
        """
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.mid_block(sample)

        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)

        if self.quant_conv is not None:
            sample = self.quant_conv(sample)

        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the encoder model.

        Returns:
            Tuple of TensorType specifications for encoder input.
        """
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")
        image_type = TensorType(
            self.dtype,
            shape=[
                "batch_size",
                self.in_channels,
                "image_height",
                "image_width",
            ],
            device=self.device,
        )

        return (image_type,)


class Decoder(Module[[Tensor], Tensor]):
    """VAE decoder for generating images from latent representations.

    This decoder progressively upsamples latent features through multiple
    decoder blocks, applying ResNet layers and attention mechanisms to
    reconstruct high-resolution images from compressed latent codes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
        use_post_quant_conv: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize Decoder module.

        Args:
            in_channels: Number of input channels (latent channels).
            out_channels: Number of output channels (image channels).
            up_block_types: Tuple of upsampling block types.
            block_out_channels: Tuple of channel counts for each decoder block.
            layers_per_block: Number of ResNet layers per decoder block.
            norm_num_groups: Number of groups for GroupNorm layers.
            act_fn: Activation function name (e.g., "silu").
            mid_block_add_attention: Whether to add attention in the middle block.
            use_post_quant_conv: Whether to use post-quantization convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype

        self.post_quant_conv: Conv2d | None = None
        if use_post_quant_conv:
            self.post_quant_conv = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=in_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=True,
                device=device,
                permute=True,
            )

        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[-1],
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            device=device,
            dtype=dtype,
        )

        up_blocks_list = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "UpDecoderBlock2D":
                up_block = UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=self.layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_upsample=not is_final_block,
                    device=device,
                    dtype=dtype,
                )
                up_blocks_list.append(up_block)
            else:
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

        self.up_blocks = ModuleList(up_blocks_list)

        self.conv_norm_out = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=block_out_channels[0],
            eps=1e-6,
            affine=True,
        )

        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

    def forward(self, z: Tensor) -> Tensor:
        """Apply Decoder forward pass.

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].

        Returns:
            Decoded image tensor of shape [N, C_out, H, W] where H and W are
            upsampled from H_latent and W_latent.
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        sample = self.conv_in(z)
        sample = self.mid_block(sample)

        for up_block in self.up_blocks:
            sample = up_block(sample)

        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)

        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the decoder model.

        Returns:
            Tuple of TensorType specifications for decoder input.
        """
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")
        latent_type = TensorType(
            self.dtype,
            shape=[
                "batch_size",
                self.in_channels,
                "latent_height",
                "latent_width",
            ],
            device=self.device,
        )

        return (latent_type,)
