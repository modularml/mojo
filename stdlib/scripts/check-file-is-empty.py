#!/usr/bin/env python3
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

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Exits successfully if the file at the given path is empty or does"
            " not exist. Otherwise, prints the file's contents, then exits"
            " unsuccessfully."
        )
    )
    parser.add_argument("path")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        return

    with open(args.path, "r") as f:
        content = f.read().strip()
        if content:
            print(
                f"error: '{args.path}' is not empty:\n{content}",
                file=sys.stderr,
            )
            exit(1)


if __name__ == "__main__":
    main()
