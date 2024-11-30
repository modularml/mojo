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
# REQUIRES: has_not
# RUN: not --crash mojo -D ASSERT=all %s 2>&1

from testing import assert_equal


def test_range_uint_bad_step_size():
    # Ensure constructing a range with a "-1" step size (i.e. reverse range)
    # with UInt is rejected and aborts now via `debug_assert` handler.
    var r = range(UInt(0), UInt(10), UInt(Int(-1)))


def main():
    test_range_uint_bad_step_size()
