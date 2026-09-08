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
"""Bazel entry point for lsp-smoke-one.

Kept separate from lsp_smoke_all.py (see lsp_smoke_lib) so that script has
exactly one owning rule; this binary and lsp-parse-smoke-stdlib each get their
own thin `main` wrapper instead of both listing the same file as `main`. See
bazel/internal/find_duplicate_srcs.py.
"""

import sys

from lsp_smoke_all import main

if __name__ == "__main__":
    sys.exit(main())
