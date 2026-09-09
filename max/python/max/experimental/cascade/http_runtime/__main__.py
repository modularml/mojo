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
"""Entry point for the ``http_worker`` binary.

Kept as a separate file from :py:mod:`.server` so that ``server.py`` lives
in exactly one Bazel rule (the :py:obj:`http_runtime` library) and can be
imported as a module by :py:mod:`.client` without also being a binary
``srcs`` entry. See ``bazel/internal/find_duplicate_srcs.py``.
"""

from __future__ import annotations

from max.experimental.cascade.http_runtime.server import cli

if __name__ == "__main__":
    cli()
