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
#
# This file is only run on linux targets with amx_tile
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# REQUIRES: amx_tile
# RUN: %mojo -debug-level full %s | FileCheck %s


from sys.info import has_intel_amx, os_is_linux

from LinAlg.intel_amx import init_intel_amx


# CHECK-LABEL: test_has_intel_amx
fn test_has_intel_amx():
    print("== test_intel_amx_amx")
    # CHECK: True
    print(os_is_linux())
    # CHECK: True
    print(has_intel_amx())
    # CHECK: True
    print(init_intel_amx())


fn main():
    test_has_intel_amx()
