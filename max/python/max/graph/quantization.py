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

"""Defines quantization encodings and configuration types for MAX graph weights."""

import enum
from dataclasses import dataclass


# Can't put this directly on the enum because it then becomes unhashable.
@dataclass(frozen=True)
class BlockParameters:
    """Parameters describing the structure of a quantization block.

    Block-based quantization stores elements in fixed-size blocks.
    Each block contains a specific number of elements in a compressed format.
    """

    elements_per_block: int
    """The number of original tensor elements grouped into one quantization block."""

    block_size: int
    """The size in bytes of the encoded representation of one quantization block."""


# TODO: BlockParameters should be integrated into this class
@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for specifying quantization parameters that affect inference.

    These parameters control how tensor values are quantized, including the method,
    bit precision, grouping, and other characteristics that affect the trade-off
    between model size, inference speed, and accuracy.
    """

    quant_method: str
    """The quantization method name (for example, ``gptq`` or ``awq``)."""

    bits: int
    """The number of bits used to represent each quantized weight element."""

    group_size: int
    """The number of weight elements that share a single set of quantization parameters."""

    desc_act: bool = False
    """Whether to use activation ordering (descending activation order). Defaults to ``False``."""

    sym: bool = False
    """Whether to use symmetric quantization. Defaults to ``False``."""


class QuantizationEncoding(enum.Enum):
    """Quantization encodings supported by MAX Graph.

    Quantization reduces the precision of neural network weights to decrease
    memory usage and potentially improve inference speed. Each encoding represents
    a different compression method with specific trade-offs between model size,
    accuracy, and computational efficiency.
    These encodings are commonly used with pre-quantized model checkpoints
    (especially `GGUF` format) or can be applied during weight allocation.

    The following example shows how to create a quantized weight using the `Q4_K` encoding:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Weight
        from max.graph.quantization import QuantizationEncoding

        # Create a quantized weight using Q4_K encoding
        encoding = QuantizationEncoding.Q4_K
        quantized_weight = Weight(
            name="linear.weight",
            dtype=DType.uint8,
            shape=[4096, 4096],
            device=DeviceRef.GPU(0),
            quantization_encoding=encoding
        )

    MAX supports several quantization formats optimized for different use cases.
    """

    #: Basic 4-bit quantization with 32 elements per block.
    Q4_0 = "Q4_0"
    #: 4-bit K-quantization with 256 elements per block.
    Q4_K = "Q4_K"
    #: 5-bit K-quantization with 256 elements per block.
    Q5_K = "Q5_K"
    #: 6-bit K-quantization with 256 elements per block.
    Q6_K = "Q6_K"
    #: Group-wise Post-Training Quantization for large language models.
    GPTQ = "GPTQ"

    @property
    def block_parameters(self) -> BlockParameters:
        """Gets the block parameters for this quantization encoding.

        Returns:
            BlockParameters: The parameters describing how elements are organized
                             and encoded in blocks for this quantization encoding.
        """
        return _BLOCK_PARAMETERS[self]

    @property
    def elements_per_block(self) -> int:
        """Number of elements per block.

        All quantization types currently supported by MAX Graph are
        block-based: groups of a fixed number of elements are formed, and each
        group is quantized together into a fixed-size output block.  This value
        is the number of elements gathered into a block.

        Returns:
            int: Number of original tensor elements in each quantized block.
        """
        return self.block_parameters.elements_per_block

    @property
    def block_size(self) -> int:
        """Number of bytes in encoded representation of block.

        All quantization types currently supported by MAX Graph are
        block-based: groups of a fixed number of elements are formed, and each
        group is quantized together into a fixed-size output block.  This value
        is the number of bytes resulting after encoding a single block.

        Returns:
            int: Size in bytes of each encoded quantization block.
        """
        return self.block_parameters.block_size

    @property
    def name(self) -> str:
        """Gets the lowercase name of the quantization encoding.

        Returns:
            str: Lowercase string representation of the quantization encoding.
        """
        return self.value.lower()

    @property
    def is_gguf(self) -> bool:
        """Checks if this quantization encoding is compatible with GGUF format.

        GGUF is a format for storing large language models and compatible
        quantized weights.

        Returns:
            bool: True if this encoding is compatible with GGUF, False otherwise.
        """
        return self in [
            QuantizationEncoding.Q4_0,
            QuantizationEncoding.Q4_K,
            QuantizationEncoding.Q5_K,
            QuantizationEncoding.Q6_K,
        ]


_BLOCK_PARAMETERS: dict[QuantizationEncoding, BlockParameters] = {
    # Block: d, q (4 bits)
    QuantizationEncoding.Q4_0: BlockParameters(32, 2 + 32 // 2),
    # Block: d, dmin, scales, q (4 bits)
    QuantizationEncoding.Q4_K: BlockParameters(256, 2 + 2 + 12 + 256 // 2),
    # Block: d, dmin, scales, qh (4 bits), qs (1 bit)
    QuantizationEncoding.Q5_K: BlockParameters(
        256, 2 + 2 + 12 + 256 // 2 + 256 // 8
    ),
    # Block: ql, qh, scales, d
    QuantizationEncoding.Q6_K: BlockParameters(
        256, 256 // 2 + 256 // 4 + 256 // 16 + 2
    ),
}
