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

# RUN: %mojo-no-debug -debug-level full %s

from sys import has_intel_amx, os_is_linux
from testing import assert_false, assert_true
from LinAlg.intel_amx import init_intel_amx


fn test_has_intel_amx():
    assert_true(os_is_linux())
    assert_true(has_intel_amx())
    assert_true(init_intel_amx())


fn main():
    test_has_intel_amx()
