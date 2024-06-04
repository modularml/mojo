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
# TODO(#33762): This is causing recursive dirs to be created.
# REQUIRES: DISABLED
# RUN: rm -rf %t && mkdir -p %t
# RUN: ln -s %S %t/tmp
# RUN: %mojo  -D TEMP_DIR=%t/tmp %s

from os.path import isdir, islink
from pathlib import Path
from sys import env_get_string

from testing import assert_false, assert_true

alias TEMP_DIR = env_get_string["TEMP_DIR"]()


def main():
    assert_true(isdir(Path(TEMP_DIR)))
    assert_true(isdir(TEMP_DIR))
    assert_true(islink(TEMP_DIR))
    assert_false(islink(str(Path(TEMP_DIR) / "nonexistant")))
