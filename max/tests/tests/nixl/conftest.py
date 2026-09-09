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
"""Configure NIXL plugin discovery before any test creates a nixlAgent.

Delegates to the shared nixl_plugin_env module: importing it (via max._core)
resolves NIXL_PLUGIN_DIR to the per-GPU-vendor plugin directory in the Bazel
runfiles, and configure() fails loudly if a GPU host resolved no flavor and
pre-loads the GPU runtime libraries the UCX plugin needs at dlopen time.
"""

from __future__ import annotations

import nixl_plugin_env

# Run at import time — before any test or fixture creates a nixlAgent — so
# NIXL_PLUGIN_DIR is set and GPU libs are loaded before the plugin manager
# singleton is constructed.
nixl_plugin_env.configure()
