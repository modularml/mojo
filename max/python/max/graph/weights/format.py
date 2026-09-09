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

"""Defines supported weight file formats and format detection for model checkpoints."""

from enum import Enum
from pathlib import Path


class WeightsFormat(str, Enum):
    """Enumeration of supported weight file formats.

    MAX supports multiple weight formats to accommodate different model sources
    and use cases.
    """

    gguf = "gguf"
    """GGUF (GPT-Generated Unified Format) for quantized models.

    File extension: ``.gguf``

    Optimized for quantized large language models, particularly those from the
    llama.cpp ecosystem. Supports multiple quantization schemes (``Q4_K``,
    ``Q5_K``, ``Q8_0``, etc.) and includes model metadata in the file.
    """

    safetensors = "safetensors"
    """Safetensors format for secure and efficient tensor storage.

    File extension: ``.safetensors``

    Designed by Hugging Face for safe serialization that prevents
    arbitrary code execution. Uses memory-mapped files for fast loading
    and supports sharding across multiple files.
    """


def weights_format(weight_paths: list[Path]) -> WeightsFormat:
    """Detect the format of weight files based on their extensions.

    This function examines the file extensions of all provided paths to
    determine the weight format. All files must have the same format;
    mixed formats are not supported.

    .. code-block:: python

        from pathlib import Path
        from max.graph.weights import weights_format

        # Detect format for safetensor files
        paths = [Path("model-00001.safetensors"), Path("model-00002.safetensors")]
        format = weights_format(paths)
        print(format)  # WeightsFormat.safetensors

    Args:
        weight_paths: List of file paths containing model weights. All files
            must have the same extension/format.

    Returns:
        The detected WeightsFormat enum value.

    Raises:
        ValueError: If weight_paths is empty, contains mixed formats, or
            has unsupported file extensions.
    """
    if not weight_paths:
        raise ValueError(
            "no weight_paths provided cannot infer weights format."
        )

    if all(weight_path.suffix == ".gguf" for weight_path in weight_paths):
        return WeightsFormat.gguf
    elif all(
        weight_path.suffix == ".safetensors" for weight_path in weight_paths
    ):
        return WeightsFormat.safetensors
    else:
        raise ValueError(f"weights type cannot be inferred from {weight_paths}")
