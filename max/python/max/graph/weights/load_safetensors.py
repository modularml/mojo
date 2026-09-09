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

"""Implements :class:`~max.graph.weights.Weights` loading from safetensors checkpoint files."""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from os import PathLike

from max._core.safetensors import SafeTensor, safe_open
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph import DeviceRef

from ..quantization import QuantizationEncoding
from ..type import Shape, ShapeLike
from ..weight import Weight
from .weights import WeightData, Weights


class SafetensorWeights(Weights):
    """Implementation for loading weights from safetensors files.

    SafetensorWeights provides a secure and efficient way to load model weights
    from safetensors format files. Safetensors is designed by Hugging Face for
    safe serialization that prevents arbitrary code execution and supports
    memory-mapped loading for fast access.

    .. code-block:: python

        import json
        import struct
        import tempfile
        from pathlib import Path

        import numpy as np
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.graph.weights import SafetensorWeights

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
                    ),
                    "transformer.layers.0.attention.weight": np.ones(
                        (4, 4), dtype=np.float32
                    ),
                },
            )

            weights = SafetensorWeights([path])

            if weights.model.embeddings.weight.exists():
                embedding_weight = weights.model.embeddings.weight.allocate(
                    dtype=DType.float32,
                    device=DeviceRef.CPU(),
                )

            attn_weight = weights.transformer.layers[0].attention.weight.allocate(
                dtype=DType.float32,
                device=DeviceRef.CPU(),
            )

    .. invisible-code-block: python

        assert attn_weight.shape == [4, 4]
    """

    _filepaths: Sequence[PathLike[str]]
    _tensors: AbstractSet[str]
    _tensors_to_file_idx: Mapping[str, int]
    _allocated: dict[str, DLPackArray]
    _st_weight_map: dict[str, Buffer]
    # This is a mapping of filepaths to SafeTensor handles. This is used to
    # avoid opening and mapping the same file to virtual memory multiple times,
    # which can use up all virtual memory.
    _st_file_handles: dict[PathLike[str], SafeTensor]

    def __init__(
        self,
        filepaths: Sequence[PathLike[str]],
        *,
        tensors: AbstractSet[str] | None = None,
        tensors_to_file_idx: Mapping[str, int] | None = None,
        prefix: str = "",
        allocated: dict[str, DLPackArray] | None = None,
        _st_weight_map: dict[str, Buffer] | None = None,
        _st_file_handles: dict[PathLike[str], SafeTensor] | None = None,
    ) -> None:
        self._filepaths = filepaths
        if tensors is not None:
            self._tensors = tensors
            assert tensors_to_file_idx is not None
            self._tensors_to_file_idx = tensors_to_file_idx
        else:
            self._tensors_to_file_idx = {}
            self._tensors = set()
            for idx, filepath in enumerate(self._filepaths):
                with safe_open(filepath) as f:
                    self._tensors |= set(f.keys())
                    self._tensors_to_file_idx |= {k: idx for k in f.keys()}  # noqa: SIM118
        self._prefix = prefix
        self._allocated = {} if allocated is None else allocated
        self._st_weight_map = {} if _st_weight_map is None else _st_weight_map
        if _st_file_handles is not None:
            self._st_file_handles = _st_file_handles
        else:
            file_handles: dict[PathLike[str], SafeTensor] = {}
            for filepath in self._filepaths:
                file_handles[filepath] = safe_open(filepath)
            self._st_file_handles = file_handles

    @property
    def name(self) -> str:
        """The current weight name or prefix."""
        return self._prefix

    def items(self):  # noqa: ANN201
        """Iterates through all allocable weights that start with the prefix."""
        for name in self._tensors:
            if name.startswith(self.name):
                yield (
                    name,
                    SafetensorWeights(
                        self._filepaths,
                        tensors=self._tensors,
                        tensors_to_file_idx=self._tensors_to_file_idx,
                        prefix=name,
                        allocated=self._allocated,
                        _st_weight_map=self._st_weight_map,
                        _st_file_handles=self._st_file_handles,
                    ),
                )

    def __getattr__(self, attr: str) -> SafetensorWeights:
        if self._prefix:
            full_path = f"{self._prefix}.{attr}"
        else:
            full_path = str(attr)
        return SafetensorWeights(
            self._filepaths,
            tensors=self._tensors,
            tensors_to_file_idx=self._tensors_to_file_idx,
            prefix=full_path,
            allocated=self._allocated,
            _st_weight_map=self._st_weight_map,
            _st_file_handles=self._st_file_handles,
        )

    def __getitem__(self, idx: int | str) -> SafetensorWeights:
        return self.__getattr__(str(idx))

    def _load_tensor(self, dtype: DType | None = None) -> Buffer:
        if self._prefix in self._st_weight_map:
            return self._st_weight_map[self._prefix]

        if self.name not in self._tensors_to_file_idx:
            possible_match = difflib.get_close_matches(
                self.name, self._tensors_to_file_idx.keys(), n=1
            )
            hint = (
                f" Did you mean '{possible_match[0]}'?"
                if possible_match
                else ""
            )
            raise KeyError(
                f"'{self.name}' is not a weight in the Safetensor checkpoint.{hint}"
            )

        filepath = self._filepaths[self._tensors_to_file_idx[self.name]]
        tensor = self._st_file_handles[filepath].get_buffer(self.name)

        self._st_weight_map[self._prefix] = tensor
        return tensor

    def data(self) -> WeightData:
        """Loads and returns the weight data for this tensor."""
        tensor = self._load_tensor()
        return WeightData(
            tensor,
            self.name,
            tensor.dtype,
            Shape(tensor.shape),
        )

    def close(self) -> None:
        """Releases the checkpoint file mappings and cached tensors.

        Drops the file handles and cached tensors (shared with prefix
        clones), letting the mappings unmap once nothing else references
        them. Only call after the weights are on their execution device;
        later reads through this object raise.
        """
        self._st_file_handles.clear()
        self._st_weight_map.clear()
        self._allocated.clear()

    def exists(self) -> bool:
        """Returns True if a tensor exists for the current name."""
        return self.name in self._tensors_to_file_idx

    def allocate(
        self,
        dtype: DType | None = None,
        shape: ShapeLike | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> Weight:
        """Creates a Weight that can be added to a graph."""
        if quantization_encoding is not None:
            raise ValueError(
                "Quantization encodings are not supported in safetensor"
                f" format. Got: {quantization_encoding}"
            )
        tensor = self._load_tensor(dtype)
        weight_dtype = tensor.dtype

        weight = Weight(
            name=self._prefix,
            dtype=weight_dtype,
            shape=tensor.shape,
            # Set align=1 because safetensors loads the data in as uint8 and
            # not the tensor dtype. This has no effect on GPU because once the
            # data is copied on GPU, the tensor will be properly aligned.
            align=1,
            device=device,
        )
        self._allocated[self._prefix] = tensor

        # Validate the loaded weight.
        weight_shape = tuple(dim for dim in weight.shape)
        if shape is not None and weight_shape != tuple(shape):
            raise ValueError(
                f"Value provided to weight '{self.name}' had different shape "
                f"(expected={shape}, actual={weight_shape})"
            )
        if dtype is not None and weight_dtype != dtype:
            raise ValueError(
                f"Value provided to weight '{self.name}' had different dtype "
                f"(expected={dtype}, actual={weight_dtype})"
            )

        return weight

    def allocate_as_bytes(self, dtype: DType | None = None) -> Weight:
        """Creates a Weight that can be added to the graph with uint8 representation.

        The last dimension is scaled by the number of bytes of the original
        dtype (for example, ``[512, 256]`` ``float32`` becomes ``[512, 1024]`` ``uint8``). Scalars
        are interpreted as shape ``[1]``.
        """
        tensor = self._load_tensor(dtype)
        if len(tensor.shape) == 0:
            tensor = tensor.view(tensor.dtype, [1])
        tensor = tensor.view(DType.uint8)

        weight = Weight(
            name=self._prefix,
            dtype=DType.uint8,
            shape=tensor.shape,
            device=DeviceRef.CPU(),
        )
        self._allocated[self._prefix] = tensor
        return weight

    @property
    def allocated_weights(self) -> dict[str, DLPackArray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
