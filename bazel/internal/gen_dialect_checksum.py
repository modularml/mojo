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
"""Generate a C header containing a SHA256 checksum of MLIR dialect .td files.

This script is invoked at build time by the dialect_checksum Bazel rule.
It reads the provided .td file paths, sorts them for determinism, computes a
SHA256 hash of their concatenated contents, and writes a C header defining
the checksum as a macro.
"""

import argparse
import hashlib


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dialect checksum header"
    )
    parser.add_argument(
        "td_files", nargs="+", help="Paths to .td files to checksum"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output header file path"
    )
    args = parser.parse_args()

    hasher = hashlib.sha256()
    for path in sorted(args.td_files):
        with open(path, "rb") as f:
            hasher.update(f.read())

    checksum = hasher.hexdigest()
    header_guard = "GEN_KGEN_GENERATEDDIALECTCHECKSUM_H"

    with open(args.output, "w") as f:
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")
        f.write(f'#define MOJO_DIALECT_CHECKSUM "{checksum}"\n\n')
        f.write(f"#endif // {header_guard}\n")


if __name__ == "__main__":
    main()
