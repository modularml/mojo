# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

import sys
from pathlib import Path

# We can't check much more than this at the moment, because the license year
# changes and the language is not mature enough to do regex yet.
alias LICENSE = String(
    """
# ===----------------------------------------------------------------------=== #
# Copyright (c)
"""
).strip()


def main():
    target_paths = sys.argv()
    if len(target_paths) < 2:
        raise Error("A file path must be given as a command line argument.")

    one_file_failed = False
    for i in range(len(target_paths)):
        if i == 0:
            # this is the current file
            continue
        file_path = Path(target_paths[i])
        if not file_path.read_text().startswith(LICENSE):
            print(
                "The license has been forgotten at the top of the file `"
                + str(file_path)
                + "`, please add it before commiting. "
            )
            one_file_failed = True

    if one_file_failed:
        print("At least one file is missing the license 💥 💔 💥")
        sys.exit(1)
