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
"""Selects the active `PluginHooks` used by the stdlib.

`CurrentPlugin` is resolved from the registered `STD_PLUGINS`, keyed on the
target's `stdlib_plugin` field.
The default build, whose field is `"default"`, resolves to `DefaultPlugin` and
leaves every hook at its default value.
"""

from ._trait import PluginHooks
from ._overlay import STD_PLUGINS
from std.sys.info import _TargetType

comptime CurrentPlugin: PluginHooks = STD_PLUGINS.current
"""The active `PluginHooks`."""

comptime PluginForTarget[Target: _TargetType] = STD_PLUGINS.for_target[Target]
"""The `PluginHooks` to use for the specified kgen.target."""
