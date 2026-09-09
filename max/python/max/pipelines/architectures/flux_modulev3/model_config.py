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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from pydantic import Field


class AutoencoderKLConfigBase(MAXModelConfigBase):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: list[str] = Field(default_factory=list, max_length=4)
    up_block_types: list[str] = Field(default_factory=list, max_length=4)
    block_out_channels: list[int] = Field(default_factory=list, max_length=4)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    shift_factor: float | None = None
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    device: DeviceRef = Field(default_factory=DeviceRef.CPU)
    dtype: DType = DType.bfloat16


class AutoencoderKLFlux2Config(AutoencoderKLConfigBase):
    patch_size: tuple[int, int] = (2, 2)
    batch_norm_eps: float = 1e-4
    batch_norm_momentum: float = 0.1
    latent_channels: int = 32  # Flux2 uses 32 channels

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLFlux2Config":
        """Generate AutoencoderKLFlux2Config from dictionary.

        Args:
            config_dict: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.

        Returns:
            AutoencoderKLFlux2Config instance.
        """
        field_names = AutoencoderKLFlux2Config.model_fields.keys()
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in field_names
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )

        if init_dict.get("shift_factor") is None:
            init_dict["shift_factor"] = 0.0
        if (
            "scaling_factor" in init_dict
            and float(init_dict["scaling_factor"]) == 0.0
        ):
            raise ValueError("`scaling_factor` must be non-zero.")

        return AutoencoderKLFlux2Config(**init_dict)
