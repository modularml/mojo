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
"""Pluggable backend hooks for vendor-specific stdlib behavior.

This package defines the `PluginHooks` trait — a compile-time interface that
can be used to override selected stdlib dispatch sites without
forking base stdlib files.

The stdlib ships a default `CurrentPlugin` alias (set in `std._plugin._impl`)
that points at `DefaultPlugin`, which leaves every hook at its default value
and preserves the built-in code paths. A plugin can overlay
its own `std/_plugin/_impl.mojo` that re-aliases `CurrentPlugin` to
a specific struct. Because each hook is a `comptime Optional[...]`
field and call sites guard the vendor path with
`comptime if CurrentPlugin.xxx_fn:`, the default build pays no runtime or
code-size cost for hooks the active backend does not set.
"""

from ._impl import CurrentPlugin, PluginForTarget
from ._trait import DefaultPlugin, PluginHooks
