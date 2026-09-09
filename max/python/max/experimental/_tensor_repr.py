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
"""Tensor string representation using the matrix-of-matrices format.

Based on https://gist.github.com/ezyang/15791ae363900f42c704c09ca34346e3

The matrix-of-matrices format displays N-dimensional tensors by treating
higher dimensions as nested grids of 2D matrices. It alternates between
horizontal and vertical stacking for dimensions beyond 2D:

- **0D (scalar)**: Single value, e.g., ``42.5``
- **1D (vector)**: Space-separated row, e.g., ``[1 2 3]``
- **2D (matrix)**: Rows on separate lines::

    [1 2
     3 4]

- **3D**: Matrices stacked **horizontally** with ``|`` separator::

    [1 2 | 5 6
     3 4 | 7 8]

  This shows two 2x2 matrices side-by-side (shape [2, 2, 2]).

- **4D**: 3D blocks stacked **vertically** with blank lines between them.
- **5D**: 4D blocks stacked **horizontally** with ``:`` separator.
- **6D+**: Continues alternating vertical/horizontal with more separators.

The stacking direction follows the rule::

    dim_offset = ndim - 2
    horizontal if dim_offset % 2 == 1 else vertical

This approach prioritizes visual clarity over matching Python literal syntax
(unlike NumPy's nested brackets). It makes spatial relationships in tensors
easier to understand for ML/tensor work.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.experimental.tensor import Tensor

# Module-level configuration
THRESHOLD: int = 1000  # Elements before summarization
EDGEITEMS: int = 3  # Elements at edges when summarizing
PRECISION: int = 4  # Significant digits for floats


def render(tensor: Tensor) -> str:
    """Renders tensor with PyTorch-like output format.

    Args:
        tensor: The tensor to render.

    Returns:
        A string representation of the tensor.
    """
    width = _compute_cell_width(tensor)
    data = _render_data(tensor, width)
    formatted = _add_brackets(data, len(tensor.shape))

    meta = [
        f"dtype={tensor.dtype}",
        f"device={tensor.device}",
    ]

    return f"Tensor({formatted}, {', '.join(meta)})"


def _format_value(val: float | bool) -> str:
    """Format a single element value.

    Args:
        val: The value to format.

    Returns:
        String representation of the value.
    """
    # Format floats with precision; handle special values (NaN, inf)
    # Note: bool is a subclass of int (not float), so isinstance(val, float)
    # correctly excludes booleans
    if isinstance(val, float):
        if val != val:  # noqa: PLR0124 # NaN check
            return "nan"
        if val == float("inf"):
            return "inf"
        if val == float("-inf"):
            return "-inf"
        return f"{val:.{PRECISION}g}"

    # bool, int, and other types use their native str representation
    return str(val)


def _compute_cell_width(tensor: Tensor) -> int:
    """Calculate the maximum cell width for alignment.

    Samples elements from the tensor to determine the widest formatted value.

    Args:
        tensor: The tensor to analyze.

    Returns:
        The maximum width needed for any cell.
    """
    limit = min(THRESHOLD, EDGEITEMS * 10)
    summarize = tensor.num_elements() > THRESHOLD

    def sampled_values() -> Iterator[str]:
        for i, v in enumerate(tensor._values()):
            yield _format_value(v)
            if summarize and i >= limit - 1:
                break

    return max((len(s) for s in sampled_values()), default=1)


def _add_brackets(data: str, ndim: int) -> str:
    """Add brackets around tensor data based on dimensionality.

    Args:
        data: The formatted tensor data.
        ndim: Number of dimensions.

    Returns:
        Data with appropriate bracket wrapping.
    """
    if ndim == 0:
        return data
    if ndim == 1:
        return f"[{data}]"

    # Multi-dimensional: wrap with [ ] and indent continuation lines
    lines = data.split("\n")
    result = [f"[{lines[0]}"]
    result.extend(" " + line for line in lines[1:])
    result[-1] = result[-1] + "]"
    return "\n".join(result)


def _render_data(tensor: Tensor, width: int) -> str:
    """Recursively render tensor data using matrix-of-matrices format.

    Args:
        tensor: The tensor to render.
        width: Cell width for alignment.

    Returns:
        Formatted string representation of the tensor data.
    """
    shape = tensor.shape
    ndim = len(shape)
    summarize = tensor.num_elements() > THRESHOLD

    if ndim == 0:
        return _format_value(next(tensor._values())).rjust(width)

    if ndim == 1:
        return _render_1d(tensor, width, summarize)

    if ndim == 2:
        return _render_2d(tensor, width, summarize)

    return _render_nd(tensor, width, ndim, summarize)


def _render_1d(tensor: Tensor, width: int, summarize: bool) -> str:
    """Render a 1D tensor as a space-separated row.

    Args:
        tensor: 1D tensor to render.
        width: Cell width for alignment.
        summarize: Whether to truncate with '...'.

    Returns:
        Space-separated string of values.
    """
    n = int(tensor.shape[0])
    vals = list(tensor._values())

    if summarize and n > 2 * EDGEITEMS:
        parts = [_format_value(vals[i]).rjust(width) for i in range(EDGEITEMS)]
        parts.append("...")
        parts.extend(
            _format_value(vals[i]).rjust(width) for i in range(n - EDGEITEMS, n)
        )
    else:
        parts = [_format_value(v).rjust(width) for v in vals]

    return " ".join(parts)


def _render_2d(tensor: Tensor, width: int, summarize: bool) -> str:
    """Render a 2D tensor as rows on separate lines.

    Args:
        tensor: 2D tensor to render.
        width: Cell width for alignment.
        summarize: Whether to truncate with '...'.

    Returns:
        Newline-separated string of rows.
    """
    rows = int(tensor.shape[0])

    if summarize and rows > 2 * EDGEITEMS:
        indices = list(range(EDGEITEMS)) + list(range(rows - EDGEITEMS, rows))
        lines = []
        for i, idx in enumerate(indices):
            if i == EDGEITEMS:
                lines.append("...")
            lines.append(_render_data(tensor[idx], width))
    else:
        lines = [_render_data(tensor[i], width) for i in range(rows)]

    return "\n".join(lines)


def _render_nd(tensor: Tensor, width: int, ndim: int, summarize: bool) -> str:
    """Render N-dimensional tensor (N >= 3) with alternating stacking.

    3D, 5D, 7D... -> horizontal stacking
    4D, 6D, 8D... -> vertical stacking

    Args:
        tensor: N-dimensional tensor to render.
        width: Cell width for alignment.
        ndim: Number of dimensions.
        summarize: Whether to truncate with '...'.

    Returns:
        Formatted string with alternating horizontal/vertical stacking.
    """
    outer = int(tensor.shape[0])

    if summarize and outer > 2 * EDGEITEMS:
        indices = list(range(EDGEITEMS)) + list(range(outer - EDGEITEMS, outer))
        subs = []
        for i, idx in enumerate(indices):
            if i == EDGEITEMS:
                subs.append("...")
            subs.append(_render_data(tensor[idx], width))
    else:
        subs = [_render_data(tensor[i], width) for i in range(outer)]

    # Alternate: 3D,5D,7D -> horizontal; 4D,6D,8D -> vertical
    if (ndim - 2) % 2 == 1:
        sep = ":" * ((ndim - 3) // 2) if ndim >= 5 else "|"
        return _join_horizontal(subs, sep)

    sep_lines = (ndim - 4) // 2 if ndim >= 6 else 0
    return _join_vertical(subs, sep_lines)


def _join_horizontal(blocks: list[str], sep: str = "") -> str:
    """Join blocks horizontally side-by-side.

    Args:
        blocks: List of multi-line block strings.
        sep: Separator character(s) between blocks.

    Returns:
        Horizontally joined string.
    """
    if not blocks:
        return ""

    lines_per_block = [b.split("\n") for b in blocks]
    max_height = max(len(lines) for lines in lines_per_block)
    widths = [
        max((len(line) for line in lines), default=0)
        for lines in lines_per_block
    ]

    # Pad each block to have the same height
    def pad_block(lines: list[str], width: int) -> list[str]:
        return [
            lines[i].ljust(width) if i < len(lines) else " " * width
            for i in range(max_height)
        ]

    padded = [
        pad_block(lines, width)
        for lines, width in zip(lines_per_block, widths, strict=True)
    ]

    joiner = f" {sep} " if sep else " "
    return "\n".join(
        joiner.join(block[i] for block in padded) for i in range(max_height)
    )


def _join_vertical(blocks: list[str], sep_lines: int = 0) -> str:
    """Join blocks vertically with optional separator lines.

    Args:
        blocks: List of multi-line block strings.
        sep_lines: Number of separator lines (0 for blank line separation).

    Returns:
        Vertically joined string.
    """
    if not blocks:
        return ""

    if sep_lines == 0:
        return "\n\n".join(blocks)

    # Use horizontal rule separators for higher dimensions
    all_lines = [line for block in blocks for line in block.split("\n")]
    max_width = max((len(line) for line in all_lines), default=0)
    separator = "\n".join("-" * max_width for _ in range(sep_lines))

    return ("\n" + separator + "\n").join(blocks)
