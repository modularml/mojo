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

"""Standardized configuration for Pipeline Inference."""

from __future__ import annotations

import os
from typing import Literal

from max.driver import DeviceSpec
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding

RepoType = Literal["online", "local"]
"""Specifies the source location type of a model repository.

``"online"`` indicates an online repository (typically Hugging Face Hub).
``"local"`` indicates a repository stored on the local filesystem.
"""


# Reference: https://github.com/ggerganov/llama.cpp/blob/eb5c3dc64bd967f2e23c87d9dec195f45468de60/src/llama.cpp#L20778
RopeType = Literal["none", "normal", "neox", "longrope", "yarn"]


PipelineRole = Literal["prefill_and_decode", "prefill_only", "decode_only"]
"""Indicates whether the pipeline should do prefill and/or decode."""


SupportedEncoding = Literal[
    "float32",
    "float16",
    "bfloat16",
    "q4_k",
    "q4_0",
    "q6_k",
    "float8_e4m3fn",
    "float4_e2m1fnx2",
    "float6_e2m3fn",
    "gptq",
]
"""All possible encodings which may be supported by a particular model."""


_SUPPORTED_ENCODING_TO_DTYPE: dict[SupportedEncoding, DType] = {
    "float32": DType.float32,
    "float16": DType.float16,
    "bfloat16": DType.bfloat16,
    "float8_e4m3fn": DType.float8_e4m3fn,
    "float4_e2m1fnx2": DType.uint8,
    "float6_e2m3fn": DType.uint8,
    "q4_k": DType.uint8,
    "q4_0": DType.uint8,
    "q6_k": DType.uint8,
    "gptq": DType.uint8,
}

_SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING: dict[
    SupportedEncoding, QuantizationEncoding | None
] = {
    "float32": None,
    "float16": None,
    "bfloat16": None,
    "float8_e4m3fn": None,
    "float4_e2m1fnx2": None,
    "float6_e2m3fn": None,
    "q4_k": QuantizationEncoding.Q4_K,
    "q4_0": QuantizationEncoding.Q4_0,
    "q6_k": QuantizationEncoding.Q6_K,
    "gptq": QuantizationEncoding.GPTQ,
}

# Basic validation for supported devices for each type of encoding.
_SUPPORTED_DEVICES: dict[SupportedEncoding, tuple[str, ...]] = {
    "float32": ("cpu", "gpu"),
    "float16": ("gpu",),
    "bfloat16": ("gpu",),
    "float8_e4m3fn": ("gpu",),
    "float4_e2m1fnx2": ("gpu",),
    "float6_e2m3fn": ("gpu",),
    "q4_k": ("cpu",),
    "q4_0": ("cpu",),
    "q6_k": ("cpu",),
    "gptq": ("gpu",),
}


def supported_encoding_dtype(encoding: SupportedEncoding) -> DType:
    """Returns the underlying model dtype for the given encoding."""
    if encoding not in _SUPPORTED_ENCODING_TO_DTYPE:
        raise ValueError(
            f"SupportedEncoding '{encoding}' does not have corresponding dtype."
        )
    return _SUPPORTED_ENCODING_TO_DTYPE[encoding]


def supported_encoding_quantization(
    encoding: SupportedEncoding,
) -> QuantizationEncoding | None:
    """Returns the :class:`QuantizationEncoding` for the given encoding."""
    if encoding not in _SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING:
        raise ValueError(
            f"SupportedEncoding '{encoding}' does not have corresponding"
            " QuantizationEncoding."
        )
    return _SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING[encoding]


def parse_supported_encoding_from_file_name(
    name: str,
) -> SupportedEncoding | None:
    """Infers a SupportedEncoding from a file name string.

    Args:
        name: The file name to parse.

    Returns:
        The :obj:`SupportedEncoding` value inferred from the file name, or
        ``None`` if no known encoding pattern is found.
    """
    # TODO(AITLIB-127): Robustify detection of quantization encoding
    # Strip directory components so that parent path segments (e.g. HF
    # cache revision hashes) don't cause false-positive matches.
    name = os.path.basename(name).lower()
    if "f32" in name or "fp32" in name or "float32" in name:
        return "float32"
    elif "bf16" in name or "bfloat16" in name:
        # Check bf16 before f16 so "bf16" doesn't match the float16 patterns.
        return "bfloat16"
    elif "fp16" in name or "float16" in name:
        return "float16"
    elif "q4_k_m" in name:
        return "q4_k"
    elif "q4_0" in name:
        return "q4_0"
    elif "q6_k" in name:
        return "q6_k"
    elif "gptq" in name:
        return "gptq"
    elif "f8" in name or "fp8" in name or "float8" in name:
        # For now, default float8 to e4m3. It is the dtype used for inference.
        return "float8_e4m3fn"
    elif "fp6" in name or "float6" in name:
        return "float6_e2m3fn"
    elif "fp4" in name or "f4" in name or "float4" in name or "nvfp4" in name:
        return "float4_e2m1fnx2"
    else:
        return None


def supported_encoding_supported_on(
    encoding: SupportedEncoding, device_spec: DeviceSpec
) -> bool:
    """Returns whether the given encoding is supported on a device."""
    return device_spec.device_type in supported_encoding_supported_devices(
        encoding
    )


def supported_encoding_supported_devices(
    encoding: SupportedEncoding,
) -> tuple[str, ...]:
    """Returns the devices that the given encoding is supported on."""
    return _SUPPORTED_DEVICES[encoding]


def is_float4_encoding(encoding: SupportedEncoding) -> bool:
    """Returns whether the given encoding is a float4 type."""
    return encoding == "float4_e2m1fnx2"


def is_float6_encoding(encoding: SupportedEncoding) -> bool:
    """Returns whether the given encoding is a float6 type."""
    return encoding == "float6_e2m3fn"
