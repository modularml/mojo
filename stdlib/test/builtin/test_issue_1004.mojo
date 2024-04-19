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
# RUN: %mojo %s
# Test for https://github.com/modularml/mojo/issues/1004

from testing import assert_equal


fn foo(x: String) raises:
    raise Error("Failed on: " + x)


def main():
    try:
        foo("Hello")
    except e:
        assert_equal(str(e), "Failed on: Hello")
