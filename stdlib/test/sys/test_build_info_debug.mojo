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
# REQUIRES: is_debug
# RUN: %mojo %s

from sys._build import is_debug_build, is_release_build

from testing import assert_false, assert_true


fn test_is_debug():
    assert_true(is_debug_build())
    assert_false(is_release_build())


fn main():
    test_is_debug()
