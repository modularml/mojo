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
var LICENSE = String(
    """
# ===----------------------------------------------------------------------=== #
# Copyright (c)
"""
).strip()


def main():
    target_paths = sys.argv()
    if len(target_paths) < 2:
        raise Error("A file path must be given as a command line argument.")

    files_without_license = List[Path]()
    for i in range(len(target_paths)):
        if i == 0:
            # this is the current file
            continue
        file_path = Path(target_paths[i])
        if not file_path.read_text().startswith(LICENSE):
            files_without_license.append(file_path)

    if len(files_without_license) > 0:
        print("The following files have missing licences 💥 💔 💥")
        for file in files_without_license:
            print(file[])
        print("Please add the license to each file before committing.")
        sys.exit(1)
