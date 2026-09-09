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

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_INT %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_BOOL %s

# Regression test: `copy_to_numpy_array` must reject non-fixed-width dtypes with an
# actionable diagnostic that names the offending dtype and points `Int` users
# at a fixed-width scalar such as `Int64`.

from std.python.numpy import copy_to_numpy_array
from std.sys import get_defined_int


def main() raises:
    comptime if get_defined_int["test"]() == 1:
        # `Int` is a machine-word integer, not a fixed-width dtype. This is the
        # case a plain list comprehension (`[i for i in range(n)]`) produces.
        # CHECK_INT: unsupported dtype
        # CHECK_INT: use a fixed-width scalar such as `Int64`
        var nums = [i for i in range(8)]
        _ = copy_to_numpy_array(nums)
    elif get_defined_int["test"]() == 2:
        # CHECK_BOOL: unsupported dtype 'bool'
        var flags = [Scalar[.bool](True), Scalar[.bool](False)]
        _ = copy_to_numpy_array(flags)
