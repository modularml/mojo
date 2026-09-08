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
"""Requantizes a HuggingFace checkpoint to MXFP6 or NVFP4.

Weight quantization is round-to-nearest over fixed-size blocks -- a pure
function of the weight tensor, with no calibration set and no activation
statistics -- so this is a batch weight transform, not a data pipeline.

MXFP6 encoding lives in :mod:`fp6_quantization`; NVFP4 encoding lives in
:mod:`fp4_quantization`. Reads one shard at a time and writes one shard at a
time, so peak memory is a couple of shards rather than the whole model. See
``max/docs/internal/MXFP6Checkpoints.md`` and
``max/docs/internal/NVFP4Checkpoints.md`` for the operator manuals.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch  # type: ignore
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    nvfp4_quantization_config,
    quantize_nvfp4,
)
from max.pipelines.weights.fp6_quantization import (
    FP6Format,
    mxfp6_quantization_config,
    quantize_mxfp6,
)

# The torch backend, not the numpy one: numpy has no bfloat16, and safetensors
# refuses a BF16 tensor on both read and write through it. Source checkpoints
# are overwhelmingly BF16, and every tensor this tool does not quantize has to
# survive the round trip in its original dtype.
from safetensors.torch import safe_open, save_file

logger = logging.getLogger("max.pipelines")

QuantFormat = FP4Format | FP6Format

_WEIGHT_INDEX = "model.safetensors.index.json"

_ALL_FORMATS: tuple[QuantFormat, ...] = (
    *FP6Format,
    *FP4Format,
)

_DEFAULT_TARGETS = (
    r"\.block_sparse_moe\.experts\.\d+\.w[123]\.weight$",
    r"\.block_sparse_moe\.shared_experts\.(gate|up|down)_proj\.weight$",
)
"""MoE expert projections, the weights that dominate an M3-shaped checkpoint.

Routed and shared experts both, matching the coverage of the shipping MXFP4
checkpoints exactly (22059 tensors for M3: 57 MoE layers x (128 routed x 3 + 3
shared)). Leaving the shared expert unquantized would diverge from MXFP4 in a
way the model config has to compensate for, since the per-region dtype probe
then finds it at a different width than the routed experts.

Attention stays in its source precision, also matching those checkpoints
(`attn_quantized_layers=set()` in `_parse_mxfp4_config`).
"""

_QKV_TARGETS = (r"\.self_attn\.(q|k|v)_proj\.weight$",)

_ALWAYS_SKIP = (
    r"embed_tokens",
    r"lm_head",
    r"\.gate\.weight$",
    r"e_score_correction_bias",
    r"norm\.weight$",
)
"""Never quantize: embeddings, the LM head, router gates, and norms.

Router gates and norms are tiny but numerically load-bearing -- quantizing a
router changes which experts fire, which is a far larger error than anything
the weights themselves contribute.
"""
_ALREADY_QUANTIZED_FLOATS = frozenset(
    dt
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if (dt := getattr(torch, name, None)) is not None
)
"""Narrow float dtypes that only ever appear as an already-quantized payload."""


def _load_shards(src: Path) -> list[Path]:
    """Returns the checkpoint's safetensors shards in index order."""
    index_path = src / _WEIGHT_INDEX
    if index_path.exists():
        index = json.loads(index_path.read_text())
        names = dict.fromkeys(index["weight_map"].values())
        return [src / name for name in names]

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found under {src}")
    return shards


def _should_quantize(
    name: str, targets: Sequence[re.Pattern[str]], ignored: Sequence[str]
) -> bool:
    """Decides whether one tensor should be quantized."""
    if any(re.search(p, name) for p in _ALWAYS_SKIP):
        return False
    if any(module in name for module in ignored):
        return False
    return any(p.search(name) for p in targets)


def _parse_format(value: str) -> QuantFormat:
    """Resolves a CLI ``--format`` string to an FP4 or FP6 encoding."""
    for enum in (FP4Format, FP6Format):
        try:
            return enum(value)
        except ValueError:
            continue
    raise ValueError(f"unknown quantization format {value!r}")


def _quantize_tensor(
    name: str, tensor: torch.Tensor, fmt: QuantFormat
) -> dict[str, torch.Tensor] | None:
    """Quantizes one 2D weight onto checkpoint keys, or returns ``None``.

    An already-quantized or integer tensor is rejected rather than reinterpreted
    as floats, which is what quantizing a source checkpoint twice would look
    like.
    """
    if not tensor.dtype.is_floating_point:
        return None
    if tensor.dtype in _ALREADY_QUANTIZED_FLOATS:
        return None
    if tensor.ndim != 2 or tensor.shape[-1] % fmt.block_size:
        return None
    # float32 first: the encoder is numpy, which cannot represent bfloat16, and
    # the widening is exact.
    values = tensor.to(torch.float32).numpy()
    if isinstance(fmt, FP4Format):
        packed, block_scales, decode_scale = quantize_nvfp4(values, fmt)
        # ModelOpt stores E4M3 block scales as float8_e4m3fn, not uint8.
        # The bits are the same; serving keys off the dtype.
        scales = torch.from_numpy(
            np.ascontiguousarray(np.squeeze(block_scales, axis=-1))
        ).view(torch.float8_e4m3fn)
        return {
            name: torch.from_numpy(np.ascontiguousarray(packed)),
            f"{name}_scale": scales,
            f"{name}_scale_2": torch.tensor(
                [float(decode_scale)], dtype=torch.float32
            ),
            # Sibling of `.weight`, not `{name}_scale`. Offline encode has no
            # activation stats; 1.0 is the identity ModelOpt default.
            f"{name.removesuffix('.weight')}.input_scale": torch.tensor(
                1.0, dtype=torch.float32
            ),
        }

    # MXFP6: packed E2M3/E3M2 codes plus one E8M0 scale per 32-element block.
    packed, scales = quantize_mxfp6(values, fmt)
    return {
        name: torch.from_numpy(packed),
        f"{name}_scale": torch.from_numpy(scales),
    }


def _quantization_config(
    fmt: QuantFormat, ignored: Sequence[str]
) -> dict[str, object]:
    """Builds the ``quantization_config`` block for the output config.json."""
    if isinstance(fmt, FP4Format):
        return nvfp4_quantization_config(fmt, ignored)
    return mxfp6_quantization_config(fmt, ignored)


def _copy_auxiliary_files(src: Path, dst: Path) -> None:
    """Copies tokenizer, generation, and other non-weight files."""
    for path in src.iterdir():
        if path.is_dir() or path.suffix == ".safetensors":
            continue
        if path.name in (_WEIGHT_INDEX, "config.json"):
            continue
        shutil.copy2(path, dst / path.name)


def _iter_shard_tensors(
    shard: Path,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yields every tensor in a shard, one at a time."""
    with safe_open(shard, framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118
            yield name, handle.get_tensor(name)


def quantize_checkpoint(
    src: Path,
    dst: Path,
    fmt: QuantFormat = FP6Format.E2M3,
    *,
    include_qkv: bool = False,
    extra_targets: Sequence[str] = (),
    dry_run: bool = False,
) -> dict[str, int]:
    """Requantizes a checkpoint to MXFP6 or NVFP4, shard by shard.

    Args:
        src: The source checkpoint directory. bf16 is strongly preferred;
            requantizing an already-quantized checkpoint would only re-encode
            information that is already gone.
        dst: The output directory, created if absent.
        fmt: The element encoding to write (``FP6Format`` or ``FP4Format``).
        include_qkv: Also quantize the attention Q/K/V projections.
        extra_targets: Additional regexes matching tensors to quantize.
        dry_run: Report what would be written without writing weights.

    Returns:
        Counts keyed ``quantized``, ``copied``, and ``bytes_saved``.

    Raises:
        FileNotFoundError: If the source has no safetensors shards.
    """
    patterns = [
        re.compile(p)
        for p in (
            *_DEFAULT_TARGETS,
            *(_QKV_TARGETS if include_qkv else ()),
            *extra_targets,
        )
    ]

    config_path = src / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    ignored = list(
        config.get("quantization_config", {}).get("ignored_layers", [])
    )

    shards = _load_shards(src)
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    stats = {"quantized": 0, "copied": 0, "bytes_saved": 0}
    weight_map: dict[str, str] = {}
    total_size = 0

    for shard_idx, shard in enumerate(shards, start=1):
        out_name = f"model-{shard_idx:05d}-of-{len(shards):05d}.safetensors"
        out_tensors: dict[str, torch.Tensor] = {}

        for name, array in _iter_shard_tensors(shard):
            written = None
            if _should_quantize(name, patterns, ignored):
                written = _quantize_tensor(name, array, fmt)
                if written is None:
                    logger.warning(
                        "%s matched a target pattern but its shape %s is not "
                        "%s-quantizable; copying verbatim",
                        name,
                        tuple(array.shape),
                        fmt.value,
                    )

            if written is None:
                out_tensors[name] = array
                stats["copied"] += 1
                continue

            out_tensors.update(written)
            stats["quantized"] += 1
            stats["bytes_saved"] += array.nbytes - sum(
                tensor.nbytes for tensor in written.values()
            )

        for name, array in out_tensors.items():
            weight_map[name] = out_name
            total_size += array.nbytes

        if dry_run:
            logger.info(
                "[dry run] %s -> %s (%d tensors)",
                shard.name,
                out_name,
                len(out_tensors),
            )
            continue

        save_file(out_tensors, dst / out_name)
        logger.info(
            "wrote %s (%d/%d), %d quantized so far",
            out_name,
            shard_idx,
            len(shards),
            stats["quantized"],
        )

    if dry_run:
        return stats

    (dst / _WEIGHT_INDEX).write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
            indent=2,
        )
    )

    config["quantization_config"] = _quantization_config(fmt, ignored)
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    _copy_auxiliary_files(src, dst)

    return stats


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the requantizer from the command line."""
    parser = argparse.ArgumentParser(
        prog="quantize_checkpoint",
        description="Requantize a HuggingFace checkpoint to MXFP6 or NVFP4.",
    )
    parser.add_argument("src", type=Path, help="source checkpoint directory")
    parser.add_argument("dst", type=Path, help="output checkpoint directory")
    parser.add_argument(
        "--format",
        choices=[f.value for f in _ALL_FORMATS],
        default=FP6Format.E2M3.value,
        help=(
            "Element encoding to write. e2m3/e3m2 select MXFP6; nvfp4 "
            "selects NVFP4. e2m3 (default) is the right MXFP6 choice for "
            "weights."
        ),
    )
    parser.add_argument(
        "--include-qkv",
        action="store_true",
        help="also quantize attention Q/K/V projections",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="REGEX",
        help="additional tensor-name regex to quantize (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without writing weights",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    fmt = _parse_format(args.format)
    stats = quantize_checkpoint(
        args.src,
        args.dst,
        fmt,
        include_qkv=args.include_qkv,
        extra_targets=args.target,
        dry_run=args.dry_run,
    )

    label = "NVFP4" if isinstance(fmt, FP4Format) else f"MXFP6 ({fmt.value})"
    logger.info(
        "%s: %d tensors quantized, %d copied, %.1f GiB saved",
        label,
        stats["quantized"],
        stats["copied"],
        stats["bytes_saved"] / 2**30,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
