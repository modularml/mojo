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

import os
import sys

from mojo._package_root import get_package_root


def exec_gpu_query() -> None:
    root = get_package_root()

    # We shouldn't run this through Bazel
    assert root

    os.execv(str(root / "bin/gpu-query"), sys.argv)


if __name__ == "__main__":
    exec_gpu_query()
