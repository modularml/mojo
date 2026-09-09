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
"""Helpers for writing minimal fake safetensors files in tests."""

import json
import os
import struct


def write_fake_safetensors(
    path: str | os.PathLike[str], dtype: str = "BF16"
) -> None:
    """Writes a minimal safetensors file with a single tensor of the given dtype.

    Args:
        path: File path to write.
        dtype: Safetensors dtype string for the single ``weight`` tensor.
    """
    header = {"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00\x00")


def write_mixed_safetensors(
    path: str | os.PathLike[str], tensors: dict[str, str]
) -> None:
    """Writes a safetensors file with multiple tensors of different dtypes.

    Args:
        path: File path to write.
        tensors: Mapping of tensor name to safetensors dtype string,
            e.g. ``{"model.layers.0.weight": "U8", "model.norm.weight": "BF16"}``.
    """
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for name, dtype in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + 2],
        }
        offset += 2
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
