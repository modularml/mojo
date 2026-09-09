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
"""Provides Ampere (SM80) matmul dispatch table lookup and config creation."""

from std.hashlib import default_comp_time_hasher


from ....utils_gpu import MatmulConfig


def create_matmul_configs_ampere[
    key: String, a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    """Returns the Ampere matmul config registered under the given key, falling back to a default config when the key is absent.

    Parameters:
        key: Tile configuration key to look up in the dispatch table.
        a_type: Element type of the left-hand input matrix.
        b_type: Element type of the right-hand input matrix.
        c_type: Element type of the output matrix.
        transpose_b: Whether the right-hand input matrix is transposed.
    """
    var dict = get_dispatch_table[a_type, b_type, c_type, transpose_b]()
    try:
        return dict[key]
    except error:
        return MatmulConfig[a_type, b_type, c_type, transpose_b](
            num_pipeline_stages=0,
        )  # 128x128_4


def get_dispatch_table[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]() -> Dict[
    String,
    MatmulConfig[a_type, b_type, c_type, transpose_b],
    default_comp_time_hasher,
]:
    """Returns the dispatch table of named Ampere matmul tile configurations for the given dtypes and transpose setting.

    Parameters:
        a_type: Element type of the left-hand input matrix.
        b_type: Element type of the right-hand input matrix.
        c_type: Element type of the output matrix.
        transpose_b: Whether the right-hand input matrix is transposed.
    """
    var tile_configs = Dict[
        String,
        MatmulConfig[a_type, b_type, c_type, transpose_b],
        default_comp_time_hasher,
    ]()

    # TODO(PAQ-1284):
    #   The configs below were optimal than the
    #   default config, leading to performance regressions that were previously
    #   masked by a bug. Retune and update before using them again. Return an
    #   empty dict in the meantime.
    #
    # The old code is in https://github.com/modularml/modular/pull/69274
    #
    return tile_configs^
