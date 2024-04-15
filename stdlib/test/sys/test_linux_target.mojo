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
# This file is only run on linux targets.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# RUN: %mojo -debug-level full %s | FileCheck %s


from sys import os_is_linux, os_is_macos


# CHECK-LABEL: test_os_query
fn test_os_query():
    print("== test_os_query")

    # CHECK: False
    print(os_is_macos())

    # CHECK: True
    print(os_is_linux())


fn main():
    test_os_query()
