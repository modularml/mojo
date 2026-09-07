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
"""Manage temporary files and directories: create, locate, and cleanup.

The `tempfile` package provides utilities for creating and managing temporary
files and directories. It handles platform-specific temporary storage locations
and ensures proper cleanup of temporary resources. Temporary files are useful
for intermediate data, testing, and operations that need scratch space without
polluting permanent storage.

Use this package when you need scratch space for intermediate computations,
temporary storage during testing, or any operation requiring files that
shouldn't persist after program execution.
"""

from .tempfile import (
    NamedTemporaryFile,
    TemporaryDirectory,
    gettempdir,
    mkdtemp,
)
