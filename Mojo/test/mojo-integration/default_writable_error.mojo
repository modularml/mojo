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

# Test that the default Writable implementation produces a clear error message
# when a field does not implement Writable.

from std.sys import get_defined_int


@fieldwise_init
struct NotWritable(ImplicitlyCopyable):
    var x: Int


@fieldwise_init
struct HasBadField(Writable):
    var field: NotWritable


def main():
    var value = HasBadField(NotWritable(1))
    var string = String()

    comptime if get_defined_int["test"]() == 1:
        # CHECK_1: constraint failed: Could not derive Writable for HasBadField
        value.write_to(string)
    elif get_defined_int["test"]() == 2:
        # CHECK_2: constraint failed: Could not derive Writable for HasBadField
        value.write_repr_to(string)
