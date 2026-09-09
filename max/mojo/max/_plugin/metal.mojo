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

from ._trait import MaxPluginHooks


struct MetalMaxPlugin(MaxPluginHooks):
    """Provides default MAX behavior for Metal targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()
