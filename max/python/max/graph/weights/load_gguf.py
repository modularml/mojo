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
"""Function for importing a GGUF checkpoint into a MAX Graph."""

from __future__ import annotations

from os import PathLike

# This is only imported internally if gguf is available
import gguf  # type: ignore
from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight
from ._gguf_reader import TokenSkippingGGUFReader
from .weights import WeightData, Weights

_GGML_TO_DTYPE = {
    gguf.GGMLQuantizationType.I8: DType.int8,
    gguf.GGMLQuantizationType.I16: DType.int16,
    gguf.GGMLQuantizationType.I32: DType.int32,
    gguf.GGMLQuantizationType.I64: DType.int64,
    gguf.GGMLQuantizationType.F16: DType.float16,
    gguf.GGMLQuantizationType.F32: DType.float32,
    gguf.GGMLQuantizationType.F64: DType.float64,
    gguf.GGMLQuantizationType.BF16: DType.bfloat16,
}

_FROM_QUANTIZED_GGML_DTYPES = {
    gguf.GGMLQuantizationType.Q4_0: QuantizationEncoding.Q4_0,
    # gguf.GGMLQuantizationType.Q4_1,
    # gguf.GGMLQuantizationType.Q5_0,
    # gguf.GGMLQuantizationType.Q5_1,
    # gguf.GGMLQuantizationType.Q8_0,
    # gguf.GGMLQuantizationType.Q8_1,
    # gguf.GGMLQuantizationType.Q2_K,
    # gguf.GGMLQuantizationType.Q3_K,
    gguf.GGMLQuantizationType.Q4_K: QuantizationEncoding.Q4_K,
    gguf.GGMLQuantizationType.Q5_K: QuantizationEncoding.Q5_K,
    gguf.GGMLQuantizationType.Q6_K: QuantizationEncoding.Q6_K,
    # gguf.GGMLQuantizationType.Q8_K,
    # gguf.GGMLQuantizationType.IQ2_XXS,
    # gguf.GGMLQuantizationType.IQ2_XS,
    # gguf.GGMLQuantizationType.IQ3_XXS,
    # gguf.GGMLQuantizationType.IQ1_S,
    # gguf.GGMLQuantizationType.IQ4_NL,
    # gguf.GGMLQuantizationType.IQ3_S,
    # gguf.GGMLQuantizationType.IQ2_S,
    # gguf.GGMLQuantizationType.IQ4_XS,
    # gguf.GGMLQuantizationType.IQ1_M,
}

_TO_QUANTIZED_GGML_DTYPES = {
    value: key for key, value in _FROM_QUANTIZED_GGML_DTYPES.items()
}


class GGUFWeights(Weights):
    """Implementation for loading weights from GGUF (GPT-Generated Unified Format) files.

    ``GGUFWeights`` provides an interface to load model weights from GGUF files,
    which are optimized for quantized large language models. GGUF is the
    successor to GGML format and is commonly used in the ``llama.cpp`` ecosystem
    for efficient storage and loading of quantized models.

    .. code-block:: python

        from pathlib import Path
        from max.graph.weights import GGUFWeights
        from max.dtype import DType
        from max.graph.quantization import QuantizationEncoding

        # Load weights from GGUF file
        gguf_path = Path("model-q4_k.gguf")
        weights = GGUFWeights(gguf_path)

        # Check if a weight exists
        if weights.model.layers[0].attention.wq.exists():
            # Allocate quantized attention weight
            wq_weight = weights.model.layers[0].attention.wq.allocate(
                dtype=DType.uint8,  # GGUF quantized weights use uint8
                device=DeviceRef.CPU()
            )

        # Access weight data with quantization info
        weight_data = weights.model.layers[0].attention.wq.data()
        print(f"Quantization: {weight_data.quantization_encoding}")
        print(f"Shape: {weight_data.shape}")

        # Allocate with quantization validation
        ffn_weight = weights.model.layers[0].feed_forward.w1.allocate(
            quantization_encoding=QuantizationEncoding.Q4_K,
            device=DeviceRef.GPU(0)
        )

        # Iterate through all weights in a layer
        for name, weight in weights.model.layers[0].items():
            if weight.exists():
                print(f"Found weight: {name}")
    """

    _reader: gguf.GGUFReader
    _tensors: dict[str, gguf.ReaderTensor]
    _prefix: str
    _allocated: dict[str, DLPackArray]

    def __init__(
        self,
        source: PathLike[str] | gguf.GGUFReader,
        tensors: dict[str, gguf.ReaderTensor] | None = None,
        prefix: str = "",
        allocated: dict[str, DLPackArray] | None = None,
    ) -> None:
        """Creates a GGUF weights reader.

        Args:
            source: Path to a GGUF file or a GGUFReader object.
            tensors: List of tensors in the GGUF checkpoint.
            prefix: Weight name or prefix.
            allocated: Dictionary of allocated values.
        """
        self._reader = (
            source
            if isinstance(source, gguf.GGUFReader)
            else TokenSkippingGGUFReader(source)
        )
        self._tensors = tensors or {t.name: t for t in self._reader.tensors}
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        return self._prefix

    def items(self):  # noqa: ANN201
        """Iterate through all allocable weights that start with the prefix."""
        for name in self._tensors:
            if name.startswith(self._prefix):
                yield (
                    name,
                    GGUFWeights(
                        self._reader,
                        self._tensors,
                        prefix=name,
                        allocated=self._allocated,
                    ),
                )

    def __getattr__(self, attr: str) -> GGUFWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        return GGUFWeights(
            self._reader,
            self._tensors,
            prefix=full_path,
            allocated=self._allocated,
        )

    def __getitem__(self, idx: int | str) -> GGUFWeights:
        return self.__getattr__(str(idx))

    def data(self) -> WeightData:
        """Loads and returns the weight data for this tensor."""
        tensor = self._raw_tensor()

        try:
            dtype = _GGML_TO_DTYPE[tensor.tensor_type]
        except KeyError:
            if tensor.tensor_type in _FROM_QUANTIZED_GGML_DTYPES:
                dtype = DType.uint8
            else:
                raise

        # Dims are reversed for some reason:
        # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
        # We have to un-reverse them here.
        shape_list = list(reversed(tensor.shape.tolist()))
        if tensor.tensor_type in _FROM_QUANTIZED_GGML_DTYPES:
            shape = Shape(
                gguf.quant_shape_to_byte_shape(shape_list, tensor.tensor_type)
            )
        else:
            shape = Shape(shape_list)
        quantization_encoding = _FROM_QUANTIZED_GGML_DTYPES.get(
            tensor.tensor_type
        )
        return WeightData(
            tensor.data, self.name, dtype, shape, quantization_encoding
        )

    def _raw_tensor(self) -> gguf.ReaderTensor:
        """Returns the GGUF tensor corresponding to this weights object.

        Raises:
            KeyError if this weights object isn't a tensor.
        """
        if self._prefix not in self._tensors:
            raise KeyError(
                f"Could not find weight named {self._prefix}. Please check that"
                " the name is correct."
            )

        return self._tensors[self._prefix]

    def exists(self) -> bool:
        """Returns True if a tensor exists for the current prefix."""
        return self._prefix in self._tensors

    def _parse_weight(
        self, tensor: gguf.ReaderTensor, device: DeviceRef
    ) -> Weight:
        # Dims are reversed for some reason:
        # https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py#L277
        # We have to un-reverse them here.
        shape = list(reversed(tensor.shape.tolist()))
        encoding = None
        if (dtype := _GGML_TO_DTYPE.get(tensor.tensor_type)) is None:
            if encoding := _FROM_QUANTIZED_GGML_DTYPES.get(tensor.tensor_type):
                # Quantized dtypes are treated as uint8 values.
                dtype = DType.uint8
                shape = gguf.quant_shape_to_byte_shape(
                    shape, tensor.tensor_type
                )
            else:
                raise ValueError(f"Unknown GGML DType: {tensor.tensor_type}.")

        return Weight(
            name=tensor.name,
            dtype=dtype,
            shape=shape,
            quantization_encoding=encoding,
            align=self._reader.alignment,
            device=device,
        )

    def allocate(
        self,
        dtype: DType | None = None,
        shape: ShapeLike | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> Weight:
        """Creates and optionally validates a new Weight."""
        tensor = self._raw_tensor()
        weight = self._parse_weight(tensor, device)
        self._allocated[self._prefix] = tensor.data

        # Validate the loaded weight.
        if shape is not None:
            expected_shape = tuple(shape)
            weight_unpacked_shape = tuple(int(dim) for dim in weight.shape)
            if weight.quantization_encoding:
                # Get the unpacked weight.
                ggml_dtype = _TO_QUANTIZED_GGML_DTYPES[
                    weight.quantization_encoding
                ]
                weight_unpacked_shape = gguf.quant_shape_from_byte_shape(
                    weight_unpacked_shape, ggml_dtype
                )

        if shape is not None and weight_unpacked_shape != expected_shape:
            raise ValueError(
                f"Value provided to weight '{self.name}' had different shape"
                f" (expected={expected_shape}, actual={weight_unpacked_shape})"
            )
        if dtype is not None and weight.dtype != dtype:
            raise ValueError(
                f"Value provided to weight '{self.name}' had different dtype"
                f" (expected={dtype}, actual={weight.dtype})"
            )

        # GGUF checkpoints can be mixed precision, so we don't check if the
        # quantization encoding matches exactly.
        if quantization_encoding and not weight.quantization_encoding:
            raise ValueError(
                "Expected quantized weight but checkpoint contained"
                f" unquantized weight for {self._prefix}"
            )

        return weight

    @property
    def allocated_weights(self) -> dict[str, DLPackArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
