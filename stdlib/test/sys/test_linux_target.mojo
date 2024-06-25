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
# RUN: %mojo %s

from sys import os_is_linux, os_is_macos

from testing import assert_false, assert_true


def test_os_query():
    assert_false(os_is_macos())
    assert_true(os_is_linux())


def main():
    test_os_query()
