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
"""Provides a set of operating-system independent functions for manipulating
file system paths."""

from .path import (
    basename,
    dirname,
    exists,
    expanduser,
    expandvars,
    getsize,
    is_absolute,
    isdir,
    isfile,
    islink,
    join,
    lexists,
    realpath,
    split,
    split_extension,
    splitroot,
)
