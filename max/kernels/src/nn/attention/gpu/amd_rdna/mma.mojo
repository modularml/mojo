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
"""RDNA Wave32 WMMA helper for attention kernels.

Wraps the raw `mma` intrinsic in a parametric per-fragment loop. RDNA
WMMA always uses 16-element A/B fragments and 8-element C/D fragments
per lane (16x16x16, group_size=1).
"""

from max.gpu.compute.mma import mma as _mma_intrinsic
from std.memory import AddressSpace
from layout import TileTensor

from .buffers import RDNA_AB_FRAG_SIZE, RDNA_CD_FRAG_SIZE


@always_inline
def rdna_mma(
    a_reg: TileTensor[_, _, address_space=.LOCAL, ...],
    b_reg: TileTensor[_, _, address_space=.LOCAL, ...],
    c_reg: TileTensor[mut=True, _, _, address_space=.LOCAL, ...],
):
    """Per-fragment WMMA loop. Derives MMA counts from operand shapes;
    accumulator indexing is col-major over (M, N): c_idx = m + n*num_m.

    Args:
        a_reg: A operand tile [num_m_mmas, RDNA_AB_FRAG_SIZE].
        b_reg: B operand tile [num_n_mmas, RDNA_AB_FRAG_SIZE].
        c_reg: Accumulator tile [num_m_mmas * num_n_mmas, RDNA_CD_FRAG_SIZE],
            modified in-place.
    """
    comptime num_m_mmas = type_of(a_reg).static_shape[0]
    comptime num_n_mmas = type_of(b_reg).static_shape[0]

    var a_vec = a_reg.vectorize[1, RDNA_AB_FRAG_SIZE]()
    var b_vec = b_reg.vectorize[1, RDNA_AB_FRAG_SIZE]()
    var c_vec = c_reg.vectorize[1, RDNA_CD_FRAG_SIZE]()
    comptime assert a_vec.flat_rank == 2
    comptime assert b_vec.flat_rank == 2
    comptime assert c_vec.flat_rank == 2

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime c_idx = m_mma + n_mma * num_m_mmas
            _mma_intrinsic(
                c_vec[c_idx, 0],
                b_vec[n_mma, 0],
                a_vec[m_mma, 0],
                c_vec[c_idx, 0],
            )
