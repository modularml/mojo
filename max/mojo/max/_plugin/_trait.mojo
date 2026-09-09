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

from std.collections.string.string_span import _get_kgen_string
from std.utils.index import IndexList
from std._plugin.selector import Plugin

from max.gpu import PDLLevel
from max.gpu.host import DeviceContext


trait MaxPluginHooks(Plugin):
    """Defines compile-time hooks for target-specific MAX behavior."""

    comptime name: __mlir_type.`!kgen.string`
    """The stable plugin identifier used by the target selector."""

    @staticmethod
    def elementwise_fn[
        rank: Int,
        simd_width: Int,
        *,
        pdl_level: PDLLevel = PDLLevel.ON,
    ](
        func: Some[
            def[
                width: Int, rank: Int, alignment: Int = 1
            ](IndexList[rank]) -> None
        ],
        shape: IndexList[rank, element_type=_],
        ctx: DeviceContext,
    ) raises:
        """Per-target plugin hook for `elementwise[...]`.

        Parameters:
            rank: The rank of the work domain.
            simd_width: The SIMD lane count for bulk invocations.
            pdl_level: PDL level for overlap control.

        Args:
            func: The body closure to invoke per index.
            shape: The shape of the work domain.
            ctx: The device context to dispatch on.

        Only invoked when `_handles_elementwise` is `True`; the default
        is never called and is a no-op.
        """
        pass

    comptime _handles_elementwise: Bool = False
    """Whether this plugin overrides MAX elementwise dispatch."""


struct DefaultMaxPlugin(MaxPluginHooks):
    """Provides the default MAX plugin behavior."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
