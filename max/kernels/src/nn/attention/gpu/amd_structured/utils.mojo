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
"""Shared helpers for gfx950 attention kernels.

Warp + fragment layout types are constructed inline in the structs that
need them (`Softmax`, `MaskTileOp`), using TileLayout / `Coord` Layout
Algebra. Selecting concrete Layout specializations across an `Int` struct
parameter at the struct field level (rather than a ternary between two
different `Layout[...]` specializations) avoids the Mojo conditional-type
unification failure: see `feedback_mojo_conditional_field_type`.
"""

from max.gpu import warp_id as get_warp_id
from std.math.uutils import udivmod


@always_inline
def get_warp_coords[BN: Int, WN: Int]() -> Tuple[Int, Int]:
    """Return `(warp_row, warp_col)` for the current warp given a BN×WN grid.

    Python-style: `var row, col = get_warp_coords[BN, WN]()`.

    Parameters:
        BN: Block extent along the N dimension of the tile grid.
        WN: Warp extent along the N dimension of the tile grid.
    """
    comptime num_warps_n = BN // WN
    return udivmod(get_warp_id(), num_warps_n)
