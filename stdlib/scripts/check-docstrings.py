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
from pathlib import Path
import sys

# TODO: Convert this script to Mojo when Mojo has a subprocess module

def main():
    # The paths to analyse are given as arguments to the script
    files_to_analyse = sys.argv[1:]

    files_with_docstrings_issues = []
    for file in files_to_analyse:
        file = Path(file)
        # We run "mojo doc" and if stderr is not empty, we consider that there is an issue
        # with the docstrings
        result = subprocess.run(["mojo", "doc", file], capture_output=True)
        if result.stderr or result.returncode != 0:
            files_with_docstrings_issues.append((file, result.stderr))
    
    for file, error in files_with_docstrings_issues:
        print(f"Docstring issue in {file}: ")
        print(error.decode())
    
    if files_with_docstrings_issues:
        sys.exit(1)

if __name__ == "__main__":
    main()
