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

from std.ffi import c_int, external_call
from std.sys.defines import get_defined_int


def main() raises:
    comptime if get_defined_int["test"]() == 1:
        # CHECK_1: cannot be negative
        _ = external_call["puts", c_int, num_fixed_args=-1](c_int(0))
    elif get_defined_int["test"]() == 2:
        # A callee cannot have more fixed arguments than it is passed.
        # CHECK_2: 'numFixedArgs' must not exceed the number of call operands
        _ = external_call["puts", c_int, num_fixed_args=2](c_int(0))
