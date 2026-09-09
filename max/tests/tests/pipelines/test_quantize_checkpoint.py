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
"""Tests the offline MXFP6 requantizer against a bfloat16 source checkpoint.

Real checkpoints are bfloat16, which numpy cannot represent at all -- so a
fixture written in float32 exercises none of what the tool actually meets on
disk. Every case here builds a bf16 shard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from max.pipelines.weights.fp6_quantization import MX_BLOCK_SIZE, FP6Format
from max.pipelines.weights.quantize_checkpoint import quantize_checkpoint
from safetensors.torch import safe_open, save_file

_EXPERT = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
_N, _K = 64, 128


def _write_source(
    src: Path, extra: dict[str, torch.Tensor] | None = None
) -> None:
    """Writes a one-shard bf16 checkpoint holding a single expert weight."""
    tensors = {
        _EXPERT: torch.randn(_N, _K, dtype=torch.float32).to(torch.bfloat16)
    }
    tensors.update(extra or {})
    src.mkdir(parents=True, exist_ok=True)
    save_file(tensors, src / "model-00001-of-00001.safetensors")
    (src / "config.json").write_text(json.dumps({"model_type": "test"}))


def _read_output(dst: Path) -> dict[str, torch.Tensor]:
    with safe_open(
        dst / "model-00001-of-00001.safetensors", framework="pt"
    ) as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118


def test_quantizes_a_bfloat16_expert(tmp_path: Path) -> None:
    """A bf16 source is readable, and its expert lands packed with scales."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_source(src)

    stats = quantize_checkpoint(src, dst, FP6Format.E2M3)

    assert stats["quantized"] == 1
    out = _read_output(dst)
    assert out[_EXPERT].shape == (_N, _K * 3 // 4)
    assert out[_EXPERT].dtype == torch.uint8
    assert out[f"{_EXPERT}_scale"].shape == (_N, _K // MX_BLOCK_SIZE)
    assert out[f"{_EXPERT}_scale"].dtype == torch.uint8


def test_non_target_tensors_keep_their_source_dtype(tmp_path: Path) -> None:
    """Pass-through must not silently widen bf16, which would inflate the copy.

    The whole non-quantized remainder of a checkpoint takes this path, so a
    dtype change here is a change to most of the bytes on disk.
    """
    src, dst = tmp_path / "src", tmp_path / "dst"
    norm = torch.randn(_K, dtype=torch.float32).to(torch.bfloat16)
    _write_source(src, {"model.layers.0.input_layernorm.weight": norm})

    quantize_checkpoint(src, dst, FP6Format.E2M3)

    out = _read_output(dst)
    copied = out["model.layers.0.input_layernorm.weight"]
    assert copied.dtype == torch.bfloat16
    assert torch.equal(copied, norm)


def test_router_gate_is_never_quantized(tmp_path: Path) -> None:
    """Quantizing a router changes which experts fire -- it must be skipped."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    gate = torch.randn(8, _K, dtype=torch.float32).to(torch.bfloat16)
    _write_source(src, {"model.layers.0.block_sparse_moe.gate.weight": gate})

    quantize_checkpoint(src, dst, FP6Format.E2M3)

    out = _read_output(dst)
    assert "model.layers.0.block_sparse_moe.gate.weight_scale" not in out
    assert out["model.layers.0.block_sparse_moe.gate.weight"].dtype == (
        torch.bfloat16
    )


@pytest.mark.parametrize("fmt", [FP6Format.E2M3, FP6Format.E3M2])
def test_config_records_the_element_encoding(
    tmp_path: Path, fmt: FP6Format
) -> None:
    """Shapes cannot tell the two encodings apart, so the config must say."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_source(src)

    quantize_checkpoint(src, dst, fmt)

    config = json.loads((dst / "config.json").read_text())
    assert config["quantization_config"]["fp6_format"] == fmt.value
    assert config["quantization_config"]["quant_method"] == "mxfp6"
