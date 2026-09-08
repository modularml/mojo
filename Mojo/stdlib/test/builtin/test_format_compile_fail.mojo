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

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s

from std.sys import get_defined_int


def main() raises:
    comptime if get_defined_int["test"]() == 1:
        # CHECK_1: note: constraint failed: Conversion flag "invalid" not recognized.
        _ = "{!invalid}".format(42)
    elif get_defined_int["test"]() == 2:
        # CHECK_2: note: constraint failed: Index 1 not in *args
        _ = "{1}".format(42)
