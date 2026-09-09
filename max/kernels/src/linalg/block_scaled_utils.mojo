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

"""Provides helpers shared by every MX block-scaled format (MXFP4 and MXFP8)."""

from std.math import isfinite, recip
from std.memory import bitcast

from .fp4_utils import E4M3_MAXABS_RECIP


@always_inline
def compute_mxfp8_block_scale[
    scales_dtype: DType
](group_max: Float32) -> Tuple[Scalar[scales_dtype], Float32, Bool]:
    """Derives an MXFP8 E8M0 block scale and the reciprocal the data path applies.

    Parameters:
        scales_dtype: E8M0 scale type to emit (`float8_e8m0fnu`).

    Args:
        group_max: Largest absolute value in the 32-element scale block.

    Returns:
        `(e8m0_scale, data_multiplier, block_is_dead)`. Multiply the block's
        data by `data_multiplier` and store `e8m0_scale`. When `block_is_dead`
        the caller MUST also zero the data: the block has no E4M3
        representation, and a non-finite lane would otherwise store
        `inf * 0.0 = NaN`.
    """
    var e8m0 = (group_max * E4M3_MAXABS_RECIP).cast[scales_dtype]()
    var multiplier = Float32(0.0)
    if group_max != 0:
        multiplier = recip(e8m0.cast[.float32]())

    # `recip` is non-finite at both ends of E8M0's range: the 2^-127 floor is an
    # fp32 subnormal that V_RCP_F32 flushes to inf, and a `group_max` that
    # overflowed fp32 casts to E8M0 NaN. Neither has an MX encoding, so the
    # block is dead and the caller must zero its data too.
    if not isfinite(multiplier):
        return (bitcast[scales_dtype](UInt8(0)), Float32(0.0), True)

    return (e8m0, multiplier, False)
