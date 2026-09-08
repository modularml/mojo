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

"""Defines the :class:`Weights` protocol and :class:`WeightData` container for model weight management."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy.typing as npt
from max.driver import CPU, Buffer, DLPackArray, is_virtual_device_mode
from max.dtype import DType

from ..buffer_utils import cast_dlpack_to
from ..quantization import QuantizationEncoding
from ..type import DeviceRef, Shape, ShapeLike
from ..weight import Weight

_Self = TypeVar("_Self", bound="Weights")


@runtime_checkable
class Weights(Protocol):
    """Protocol for managing and accessing model weights hierarchically.

    The Weights protocol provides a convenient interface for loading and organizing
    neural network weights. It supports hierarchical naming through attribute and
    index access, making it easy to work with complex model architectures.

    Weights in MAX are tensors backed by external memory (buffers or memory-mapped
    files) that remain separate from the compiled graph.

    .. code-block:: python

        import json
        import struct
        import tempfile
        from pathlib import Path

        import numpy as np
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.graph.weights import load_weights

        def write_safetensors(path, tensors):
            header, buffers, offset = {}, [], 0
            for name, arr in tensors.items():
                arr = np.ascontiguousarray(arr)
                header[name] = {
                    "dtype": "F32",
                    "shape": list(arr.shape),
                    "data_offsets": [offset, offset + arr.nbytes],
                }
                buffers.append(arr.tobytes())
                offset += arr.nbytes
            blob = json.dumps(header).encode()
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(blob)))
                f.write(blob)
                for b in buffers:
                    f.write(b)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            write_safetensors(
                path,
                {
                    "transformer.layers.0.attention.weight": np.ones(
                        (8, 8), dtype=np.float32
                    ),
                },
            )
            weights = load_weights([path])

            attn_weight = weights.transformer.layers[0].attention.weight.allocate(
                dtype=DType.float32,
                device=DeviceRef.CPU(),
            )
            # Creates weight named "transformer.layers.0.attention.weight".

    .. invisible-code-block: python

        assert attn_weight.name == "transformer.layers.0.attention.weight"
    """

    @property
    def name(self) -> str:
        """The current weight name or prefix.

        Returns:
            The hierarchical name built from attribute and index access.
            For example, if accessed as ``weights.model.layers[0]``,
            returns ``model.layers.0``.
        """
        ...

    def __getattr__(self: _Self, attr: str) -> _Self: ...

    def __getitem__(self: _Self, idx: int | str) -> _Self: ...

    def exists(self) -> bool:
        """Checks if a weight with this exact name exists.

        .. code-block:: python

            import json
            import struct
            import tempfile
            from pathlib import Path

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef
            from max.graph.weights import load_weights

            def write_safetensors(path, tensors):
                header, buffers, offset = {}, [], 0
                for name, arr in tensors.items():
                    arr = np.ascontiguousarray(arr)
                    header[name] = {
                        "dtype": "F32",
                        "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes],
                    }
                    buffers.append(arr.tobytes())
                    offset += arr.nbytes
                blob = json.dumps(header).encode()
                with open(path, "wb") as f:
                    f.write(struct.pack("<Q", len(blob)))
                    f.write(blob)
                    for b in buffers:
                        f.write(b)

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.safetensors"
                write_safetensors(
                    path, {"model.head.weight": np.ones((4, 4), dtype=np.float32)}
                )
                weights = load_weights([path])

                if weights.model.head.weight.exists():
                    head = weights.model.head.weight.allocate(
                        dtype=DType.float32, device=DeviceRef.CPU()
                    )
                else:
                    head = None

        .. invisible-code-block: python

            assert head is not None

        Returns:
            ``True`` if a weight with the current hierarchical name exists
            in the loaded weights, ``False`` otherwise.
        """
        ...

    def items(self: _Self) -> Iterator[tuple[str, _Self]]:
        """Iterates through all weights that start with the current prefix.

        .. code-block:: python

            import json
            import struct
            import tempfile
            from pathlib import Path

            import numpy as np
            from max.graph.weights import load_weights

            def write_safetensors(path, tensors):
                header, buffers, offset = {}, [], 0
                for name, arr in tensors.items():
                    arr = np.ascontiguousarray(arr)
                    header[name] = {
                        "dtype": "F32",
                        "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes],
                    }
                    buffers.append(arr.tobytes())
                    offset += arr.nbytes
                blob = json.dumps(header).encode()
                with open(path, "wb") as f:
                    f.write(struct.pack("<Q", len(blob)))
                    f.write(blob)
                    for b in buffers:
                        f.write(b)

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.safetensors"
                write_safetensors(
                    path,
                    {
                        "transformer.layers.0.q_proj.weight": np.ones(
                            (4, 4), dtype=np.float32
                        ),
                        "transformer.layers.0.k_proj.weight": np.ones(
                            (4, 4), dtype=np.float32
                        ),
                    },
                )
                weights = load_weights([path])

                found = [
                    name
                    for name, weight in weights.transformer.layers[0].items()
                ]

        .. invisible-code-block: python

            assert len(found) == 2

        Yields:
            Tuples of (name, weight_accessor) for each weight under the
            current prefix. The name is relative to the current prefix.
        """
        ...

    def data(self) -> WeightData:
        """Returns weight data with metadata.

        .. code-block:: python

            import json
            import struct
            import tempfile
            from pathlib import Path

            import numpy as np
            from max.dtype import DType
            from max.graph.weights import load_weights

            def write_safetensors(path, tensors):
                header, buffers, offset = {}, [], 0
                for name, arr in tensors.items():
                    arr = np.ascontiguousarray(arr)
                    header[name] = {
                        "dtype": "F32",
                        "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes],
                    }
                    buffers.append(arr.tobytes())
                    offset += arr.nbytes
                blob = json.dumps(header).encode()
                with open(path, "wb") as f:
                    f.write(struct.pack("<Q", len(blob)))
                    f.write(blob)
                    for b in buffers:
                        f.write(b)

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.safetensors"
                write_safetensors(
                    path,
                    {
                        "model.embeddings.weight": np.ones(
                            (4, 4), dtype=np.float32
                        )
                    },
                )
                weights = load_weights([path])

                weight_data = weights.model.embeddings.weight.data()
                # weight_data.shape and weight_data.dtype hold the metadata.

                fp16_data = weight_data.astype(DType.float16)

        .. invisible-code-block: python

            assert fp16_data.dtype == DType.float16

        Returns:
            A :class:`WeightData` object containing the tensor data along with
            metadata like name, dtype, shape, and quantization encoding.

        Raises:
            KeyError: If no weight exists at the current hierarchical name.
        """
        ...

    def allocate(
        self,
        dtype: DType | None = None,
        shape: ShapeLike | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> Weight:
        """Creates a :class:`Weight` object for this tensor.

        .. code-block:: python

            import json
            import struct
            import tempfile
            from pathlib import Path

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph
            from max.graph.weights import load_weights

            def write_safetensors(path, tensors):
                header, buffers, offset = {}, [], 0
                for name, arr in tensors.items():
                    arr = np.ascontiguousarray(arr)
                    header[name] = {
                        "dtype": "F32",
                        "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes],
                    }
                    buffers.append(arr.tobytes())
                    offset += arr.nbytes
                blob = json.dumps(header).encode()
                with open(path, "wb") as f:
                    f.write(struct.pack("<Q", len(blob)))
                    f.write(blob)
                    for b in buffers:
                        f.write(b)

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.safetensors"
                write_safetensors(
                    path,
                    {
                        "model.layers.0.weight": np.ones(
                            (8, 8), dtype=np.float32
                        )
                    },
                )
                weights = load_weights([path])

                weight = weights.model.layers[0].weight.allocate(
                    dtype=DType.float32,
                    shape=(8, 8),
                    device=DeviceRef.CPU(),
                )

                with Graph("allocate_example", input_types=[]) as graph:
                    weight_tensor = graph.add_weight(weight)

        .. invisible-code-block: python

            assert weight_tensor.shape == [8, 8]

        Args:
            dtype: Data type for the weight. If ``None``, uses the original dtype.
            shape: Shape of the weight tensor. If ``None``, uses the original shape.
            quantization_encoding: Quantization scheme to apply (for example, ``Q4_K``, ``Q8_0``).
            device: Target device for the weight (CPU or GPU).

        Returns:
            A :class:`Weight` object that can be added to a graph using
            ``graph.add_weight()``.
        """
        ...

    @property
    def allocated_weights(self) -> dict[str, DLPackArray]:
        """Returns all previously allocated weights.

        This only includes weights that were explicitly allocated using
        :meth:`Weights.allocate`, not all available weights.

        Returns:
            A dictionary mapping weight names to their numpy arrays for
            all weights that have been allocated through this interface.
        """
        ...


@dataclasses.dataclass
class WeightData(DLPackArray):
    """Container for weight tensor data with metadata.

    ``WeightData`` encapsulates a weight tensor along with its metadata,
    providing utilities for type conversion and format compatibility.
    It supports the DLPack protocol for efficient tensor sharing between
    frameworks.
    """

    data: DLPackArray
    """The weight tensor as a DLPack array."""
    name: str
    """Hierarchical name of the weight (for example, ``model.layers.0.weight``)."""

    dtype: DType
    """Data type of the tensor (for example, ``DType.float32``, ``DType.uint8``)."""

    shape: Shape
    """Shape of the tensor as a Shape object."""

    quantization_encoding: QuantizationEncoding | None = None
    """Optional quantization scheme applied to the weight."""

    def __dlpack__(self, *, stream: None = None) -> Any:
        return self.data.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> Any:
        return self.data.__dlpack_device__()

    def to_buffer(self) -> Buffer:
        """Mutates the data into a Buffer."""
        # We store the result of Buffer.from_dlpack because it may copy the
        # data.
        if not isinstance(self.data, Buffer):
            self.data = Buffer.from_dlpack(self.data)
        return self.data

    @classmethod
    def from_numpy(cls, arr: npt.NDArray[Any], name: str) -> WeightData:
        """Create WeightData from a numpy array.

        Args:
            arr: Numpy array containing the weight data.
            name: Name to assign to this weight.

        Returns:
            A new WeightData instance with dtype and shape inferred
            from the numpy array.
        """
        return cls(arr, name, DType.from_numpy(arr.dtype), Shape(arr.shape))

    def astype(self, dtype: DType) -> WeightData:
        """Convert the weight data to a different dtype.

        This method performs actual data conversion of the underlying tensor
        data. Special handling is provided for bfloat16 conversions using
        PyTorch when available.

        During cross-compilation (warm-cache) scenarios there is no device to
        run the conversion on, so the result holds an uninitialized buffer of
        the target dtype. The values are never read in that mode, and
        ``dtype`` still describes ``data``, which consumers that read the
        DLPack payload rather than this metadata rely on.

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph.weights import WeightData

            # Convert float32 weights to float16 for reduced memory.
            weight_data = WeightData.from_numpy(
                np.ones((4, 4), dtype=np.float32), "model.layer.weight"
            )
            fp16_data = weight_data.astype(DType.float16)

        .. invisible-code-block: python

            assert fp16_data.dtype == DType.float16

        Args:
            dtype: Target data type for conversion.

        Returns:
            A new WeightData instance with the converted data.
        """
        if self.dtype == dtype:
            return self
        # Uninitialized, but at the target dtype: nothing reads the values
        # here, while `dtype` describes `data` everywhere.
        if is_virtual_device_mode():
            return WeightData(
                data=Buffer(dtype, [int(dim) for dim in self.shape]),
                name=self.name,
                dtype=dtype,
                shape=self.shape,
                quantization_encoding=self.quantization_encoding,
            )
        data = cast_dlpack_to(self.data, self.dtype, dtype, CPU())
        return WeightData(
            data=data,
            name=self.name,
            dtype=dtype,
            shape=Shape(data.shape),
        )

    def __repr__(self) -> str:
        return f"WeightData({self.dtype}, {self.shape})"


WeightsAdapter = Callable[..., dict[str, WeightData]]
"""Type alias for functions that adapt weight formats to WeightData dictionaries.

WeightsAdapter functions are used by pipeline architectures to convert between
different checkpoint formats (for example, HuggingFace, PyTorch) and MAX's internal
format. They take model configuration and return a dictionary mapping weight
names to WeightData objects.
"""
