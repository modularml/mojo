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

"""Tool to compare max and torch tensors.

Example usage:
    Compare two tensor files
        bazel run //max/tests/integration/tools:compare_tensors -- \
            --torch-tensor ref.pt --max-tensor out.max --rtol 1e-5 --atol 1e-8

    Auto-match tensors in directories
        bazel run //max/tests/integration/tools:compare_tensors -- \
            --torch-tensor torch_dir/ --max-tensor max_dir/
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from max.driver.buffer import load_max_buffer


@dataclass
class ComparisonMetrics:
    """Metrics computed when comparing two tensors."""

    max_abs_diff: float
    max_rel_diff: float
    max_abs_diff_index: tuple[int, ...]
    max_rel_diff_index: tuple[int, ...]
    num_failing_elements: int | None
    total_elements: int


@dataclass
class TensorComparisonResult:
    """Result of comparing two tensors."""

    torch_tensor_name: str
    max_tensor_name: str
    all_close: bool | None
    metrics: ComparisonMetrics
    torch_shape: tuple[int, ...]
    max_shape: tuple[int, ...]
    was_reshaped: bool
    original_max_shape: tuple[int, ...]


def load_torch_tensor(file_path: Path) -> torch.Tensor:
    """Load a torch tensor from a .pt file and move to CPU.

    Args:
        file_path: Path to the .pt file.

    Returns:
        Loaded torch tensor on CPU.
    """
    tensor = torch.load(file_path, weights_only=True).squeeze(0)
    return tensor.cpu() if tensor.is_cuda else tensor


def load_max_tensor_as_torch(file_path: Path) -> torch.Tensor:
    """Load a max tensor and convert it to torch tensor.

    Args:
        file_path: Path to the .max file.

    Returns:
        Torch tensor converted from max tensor.
    """
    buffer = load_max_buffer(file_path)
    torch_tensor = torch.from_dlpack(buffer).cpu()
    return torch_tensor


def compute_comparison_metrics(
    torch_tensor: torch.Tensor,
    max_tensor: torch.Tensor,
    rtol: float | None,
    atol: float | None,
) -> ComparisonMetrics:
    """Compute comparison metrics between two tensors.

    Args:
        torch_tensor: Reference tensor from torch.
        max_tensor: Tensor to compare from max.
        rtol: Relative tolerance for tolerance check. If None, relative tolerance is not checked.
        atol: Absolute tolerance for tolerance check. If None, absolute tolerance is not checked.

    Returns:
        ComparisonMetrics containing max_abs_diff, max_rel_diff, max_abs_diff_index,
        max_rel_diff_index, num_failing_elements, and total_elements.
        num_failing_elements will be None if both rtol and atol are None. If only one tolerance
        is provided, only that tolerance is checked (the other is treated as infinite).
    """
    abs_diff = torch.abs(torch_tensor - max_tensor)

    max_abs_diff = float(torch.max(abs_diff).item())
    max_abs_diff_idx = tuple(
        int(i.item())
        for i in torch.unravel_index(torch.argmax(abs_diff), abs_diff.shape)
    )

    rel_diff = abs_diff / torch.maximum(
        torch.abs(torch_tensor), torch.tensor(1e-10)
    )
    max_rel_diff = float(torch.max(rel_diff).item())
    max_rel_diff_idx = tuple(
        int(i.item())
        for i in torch.unravel_index(torch.argmax(rel_diff), rel_diff.shape)
    )

    num_failing: int | None = None
    if rtol is not None or atol is not None:
        effective_rtol = rtol if rtol is not None else float("inf")
        effective_atol = atol if atol is not None else float("inf")
        is_close = torch.isclose(
            torch_tensor, max_tensor, rtol=effective_rtol, atol=effective_atol
        )
        num_failing = int((~is_close).sum().item())

    total_elements = int(torch.numel(torch_tensor))

    return ComparisonMetrics(
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        max_abs_diff_index=max_abs_diff_idx,
        max_rel_diff_index=max_rel_diff_idx,
        num_failing_elements=num_failing,
        total_elements=total_elements,
    )


def compare_tensors(
    torch_tensor: torch.Tensor,
    max_tensor: torch.Tensor,
    torch_name: str,
    max_name: str,
    rtol: float | None = None,
    atol: float | None = None,
    allow_reshape: bool = False,
) -> TensorComparisonResult:
    """Compare two tensors and return comparison results.

    Args:
        torch_tensor: Reference tensor from torch.
        max_tensor: Tensor to compare from max.
        torch_name: Name of the torch tensor.
        max_name: Name of the max tensor.
        rtol: Relative tolerance. If None, relative tolerance is not checked.
        atol: Absolute tolerance. If None, absolute tolerance is not checked.
               If both are None, only reports metrics without pass/fail.
               If only one is provided, only that tolerance is checked.
        allow_reshape: If True, reshape tensors to match when they have the same number of elements.

    Returns:
        TensorComparisonResult containing comparison metrics.
    """
    torch_cpu = torch_tensor.cpu().to(torch.float32)
    max_cpu = max_tensor.cpu().to(torch.float32)

    original_torch_shape = tuple(torch_tensor.shape)
    original_max_shape = tuple(max_tensor.shape)
    reshaped = False

    if torch_cpu.shape != max_cpu.shape:
        if allow_reshape and torch_cpu.numel() == max_cpu.numel():
            max_cpu = max_cpu.reshape(torch_cpu.shape)
            reshaped = True
        else:
            return TensorComparisonResult(
                torch_tensor_name=torch_name,
                max_tensor_name=max_name,
                all_close=None,
                metrics=ComparisonMetrics(
                    max_abs_diff=float("nan"),
                    max_rel_diff=float("nan"),
                    max_abs_diff_index=(),
                    max_rel_diff_index=(),
                    num_failing_elements=None,
                    total_elements=0,
                ),
                torch_shape=original_torch_shape,
                max_shape=original_max_shape,
                was_reshaped=False,
                original_max_shape=original_max_shape,
            )

    metrics = compute_comparison_metrics(torch_cpu, max_cpu, rtol, atol)

    all_close: bool | None = None
    if metrics.num_failing_elements is not None:
        all_close = metrics.num_failing_elements == 0

    return TensorComparisonResult(
        torch_tensor_name=torch_name,
        max_tensor_name=max_name,
        all_close=all_close,
        metrics=metrics,
        torch_shape=tuple(torch_cpu.shape),
        max_shape=tuple(max_cpu.shape),
        was_reshaped=reshaped,
        original_max_shape=original_max_shape,
    )


def find_matching_tensors(
    torch_dir: Path, max_dir: Path
) -> list[tuple[Path, Path]]:
    """Find all pairs of matching tensors between torch and max directories.

    Args:
        torch_dir: Directory containing torch tensors (.pt files).
        max_dir: Directory containing max tensors (.max files).

    Returns:
        List of tuples (torch_path, max_path) for matching tensor names.
    """
    torch_tensors = {f.stem: f for f in torch_dir.rglob("*.pt")}

    max_tensors = {f.stem: f for f in max_dir.rglob("*.max")}

    matching_pairs: list[tuple[Path, Path]] = []
    for name in torch_tensors:  # noqa: PLC0206
        if name in max_tensors:
            matching_pairs.append((torch_tensors[name], max_tensors[name]))

    return matching_pairs


@click.command()
@click.option(
    "--torch-tensor",
    "torch_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to torch tensor file (.pt) or directory.",
)
@click.option(
    "--max-tensor",
    "max_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to max tensor file (.max) or directory.",
)
@click.option(
    "--rtol",
    type=float,
    default=None,
    help="Relative tolerance for pass/fail check. Can be used alone or with --atol.",
)
@click.option(
    "--atol",
    type=float,
    default=None,
    help="Absolute tolerance for pass/fail check. Can be used alone or with --rtol.",
)
@click.option(
    "--allow-reshape",
    is_flag=True,
    default=False,
    help="Allow reshaping tensors when they have the same number of elements but different shapes.",
)
def main(
    torch_path: Path,
    max_path: Path,
    rtol: float | None,
    atol: float | None,
    allow_reshape: bool,
) -> None:
    """Compare tensors from torch and max paths.

    Paths can be either specific tensor files or directories containing tensors.
    """
    results: list[TensorComparisonResult] = []

    if torch_path.is_file() and max_path.is_file():
        click.echo(
            f"Comparing tensors:\n  Torch: {torch_path}\n  Max:   {max_path}\n"
        )
        results.append(
            compare_tensors(
                load_torch_tensor(torch_path),
                load_max_tensor_as_torch(max_path),
                torch_path.stem,
                max_path.stem,
                rtol=rtol,
                atol=atol,
                allow_reshape=allow_reshape,
            )
        )
    elif torch_path.is_dir() and max_path.is_dir():
        matching_pairs = find_matching_tensors(torch_path, max_path)
        if not matching_pairs:
            click.echo("No matching tensors found.", err=True)
            sys.exit(1)

        click.echo(
            f"Found {len(matching_pairs)} matching tensor pair(s) in:\n  Torch: {torch_path}\n  Max:   {max_path}\n"
        )

        for torch_file, max_file in matching_pairs:
            results.append(
                compare_tensors(
                    load_torch_tensor(torch_file),
                    load_max_tensor_as_torch(max_file),
                    torch_file.stem,
                    max_file.stem,
                    rtol=rtol,
                    atol=atol,
                    allow_reshape=allow_reshape,
                )
            )
    else:
        torch_type = "file" if torch_path.is_file() else "directory"
        max_type = "file" if max_path.is_file() else "directory"
        raise click.UsageError(
            f"Both paths must be either files or directories.\n"
            f"  --torch-tensor: {torch_type}\n"
            f"  --max-tensor: {max_type}"
        )

    tolerances_provided = rtol is not None or atol is not None

    if tolerances_provided:
        tolerance_parts = []
        if rtol is not None:
            tolerance_parts.append(f"rtol: {rtol}")
        if atol is not None:
            tolerance_parts.append(f"atol: {atol}")
        click.echo(f"\nUsing {', '.join(tolerance_parts)}")
    else:
        click.echo(
            "\nNo tolerances provided - reporting metrics only (no pass/fail)\n"
        )

    # Track if any tensor has large relative diff for a single warning at the end
    has_large_rel_diff = False

    for result in results:
        click.echo(
            f"\nTensor: {result.torch_tensor_name} vs {result.max_tensor_name}"
        )
        if result.was_reshaped:
            click.echo(
                f"Shapes: torch={result.torch_shape}, max={result.original_max_shape} → {result.max_shape} (reshaped)"
            )
        else:
            click.echo(
                f"Shapes: torch={result.torch_shape}, max={result.max_shape}"
            )

        if result.torch_shape != result.max_shape:
            click.echo("\n⚠️  SHAPE MISMATCH - tensors cannot be compared!")
            continue

        metrics = result.metrics
        click.echo(
            f"Greatest absolute difference: {metrics.max_abs_diff} at index {metrics.max_abs_diff_index}"
        )
        click.echo(
            f"Greatest relative difference: {metrics.max_rel_diff} at index {metrics.max_rel_diff_index}"
        )
        if metrics.max_rel_diff > 100:
            has_large_rel_diff = True

        if tolerances_provided and result.all_close is not None:
            click.echo()
            if result.all_close:
                click.echo("✓ Tensors are close within specified tolerances!")
            else:
                assert metrics.num_failing_elements is not None
                failure_pct = (
                    metrics.num_failing_elements / metrics.total_elements
                ) * 100
                click.echo("✗ Tensors exceed specified tolerances!")
                click.echo(
                    f"Mismatched elements: {metrics.num_failing_elements} / {metrics.total_elements} ({failure_pct:.1f}%)"
                )

    # Print warning once if any tensor had large relative diff
    if has_large_rel_diff:
        click.echo(
            "\n⚠️  Note: Large relative diff often indicates near-zero values in reference tensor"
        )

    if tolerances_provided:
        num_close = sum(1 for r in results if r.all_close is True)
        click.echo(
            f"\n{'=' * 80}\nSummary: {num_close}/{len(results)} tensors close\n{'=' * 80}"
        )
        if num_close < len(results):
            sys.exit(1)
    else:
        click.echo(
            f"\n{'=' * 80}\nComparison complete for {len(results)} tensor(s)\n{'=' * 80}"
        )


if __name__ == "__main__":
    main()
