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

from ._overlay import MAX_PLUGINS
from ._trait import MaxPluginHooks

from std.sys.info import _TargetType


comptime CurrentMaxPlugin: MaxPluginHooks = MAX_PLUGINS.current
"""The active MAX plugin."""

comptime MaxPluginForTarget[Target: _TargetType] = MAX_PLUGINS.for_target[
    Target
]
"""The MAX plugin selected for the specified target."""
