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

import subprocess
import sys

# TODO: Use the "mojo doc" directly when there is an option to
# fail if warnings are present (something like -Werror for gcc).


def main():
    # This is actually faster than running "mojo doc" on each file since
    # "mojo doc" only accept a single file/path as argument
    command = [
        "mojo",
        "doc",
        "--diagnose-missing-doc-strings",
        "-o",
        "/dev/null",
        "./stdlib/src",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.stderr or result.returncode != 0:
        print(f"Docstring issue found in the stdlib: ")
        print(result.stderr.decode())
        sys.exit(1)


if __name__ == "__main__":
    main()
