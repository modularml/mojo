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

# RUN: %mojo %s | FileCheck %s --check-prefix=NO_ARGS
# RUN: %mojo %s --arg1 --arg2=10 --arg3="arg3" | FileCheck %s --check-prefix=ARGS

from std.sys import argv


def main() raises:
    # NO_ARGS: exec_argv.mojo

    # ARGS: exec_argv.mojo
    # ARGS: --arg1
    # ARGS: --arg2=10
    # ARGS: --arg3=arg3

    for i in range(argv().__len__()):
        print(argv()[i])
