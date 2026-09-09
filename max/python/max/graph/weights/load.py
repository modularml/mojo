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
"""Function for loading paths as Weights."""

import os
from pathlib import Path

from .format import WeightsFormat, weights_format
from .load_safetensors import SafetensorWeights
from .loader_wrappers import GGUFWeights
from .weights import Weights


def load_weights(paths: list[Path]) -> Weights:
    """Loads neural network weights from checkpoint files.

    Automatically detects checkpoint formats based on file extensions and returns
    the appropriate :class:`~max.graph.weights.Weights` implementation. Supported formats:

    - ``.safetensors`` (Safetensors)
    - ``.gguf`` (GGUF)

    The following example shows how to load weights from a Safetensors file:

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
                    "model.layers.23.mlp.gate_proj.weight": np.ones(
                        (16, 8), dtype=np.float32
                    )
                },
            )

            # load_weights also accepts multiple paths for sharded checkpoints.
            weights = load_weights([path])
            layer_weight = weights.model.layers[23].mlp.gate_proj.weight.allocate(
                dtype=DType.float32,
                shape=[16, 8],
                device=DeviceRef.CPU(),
            )

    .. invisible-code-block: python

        assert layer_weight.shape == [16, 8]

    Args:
        paths: List of :class:`pathlib.Path` objects pointing to checkpoint files.
            For multi-file checkpoints (for example, sharded Safetensors), provide
            all file paths in the list. For single-file checkpoints, provide
            a list with one path.
    """
    # Check that paths is not empty.
    if not paths:
        raise ValueError("no paths provided, cannot load weights.")

    # Check that all paths exist
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(
                f"file path ({path}) does not exist, cannot load weights."
            )

    _weights_format = weights_format(paths)

    if _weights_format == WeightsFormat.gguf:
        if len(paths) > 1:
            raise ValueError("loading multiple gguf files is not supported.")

        return GGUFWeights(paths[0])
    elif _weights_format == WeightsFormat.safetensors:
        return SafetensorWeights(paths)
    else:
        raise ValueError(
            f"loading weights format '{_weights_format}' not supported."
        )
