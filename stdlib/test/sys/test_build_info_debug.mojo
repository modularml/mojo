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
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys._build import is_debug_build, is_release_build


# CHECK-OK-LABEL: test_is_debug
fn test_is_debug():
    print("== test_is_debug")

    # CHECK: True
    print(is_debug_build())

    # CHECK: False
    print(is_release_build())


fn main():
    test_is_debug()
