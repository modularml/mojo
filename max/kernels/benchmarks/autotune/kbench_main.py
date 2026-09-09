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

"""Entry point for the `kbench` binary.

The kbench Python harness itself lives in `:kbench_lib` (which now
includes `kbench.py`) so it can be reused by out-of-tree consumers.
This file exists purely so the `kbench` `modular_py_binary` has a
unique `srcs` entry — listing `kbench.py` in both the binary and the
library would trip the "Lint files in multiple targets" check.
"""

import os

from kbench import main

if __name__ == "__main__":
    # Mirror kbench.py's own __main__ chdir so relative paths supplied
    # on the CLI are resolved from where the user ran `bazel run`, not
    # the runfiles dir.
    if directory := os.environ.get("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)
    main()
