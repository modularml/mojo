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

from std._plugin.selector import PluginSelector

from ._trait import DefaultMaxPlugin, MaxPluginHooks
from .cuda import CudaMaxPlugin
from .hip import HipMaxPlugin
from .metal import MetalMaxPlugin


comptime MAX_PLUGINS = PluginSelector[
    DefaultMaxPlugin,
    MetalMaxPlugin,
    CudaMaxPlugin,
    HipMaxPlugin,
]
