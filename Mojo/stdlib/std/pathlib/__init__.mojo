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
"""Filesystem path manipulation and navigation.

The `pathlib` package provides object-oriented filesystem path handling with
platform-independent path operations. It offers the `Path` type for
representing and manipulating filesystem paths, along with utilities for path
joining, expansion, and directory operations. This package makes working with
file paths safer and more intuitive than manual string manipulation.

Use this package for any filesystem path operations including constructing
paths, navigating directories, checking file existence, or performing
path-related queries in a platform-independent way.
"""

from .path import DIR_SEPARATOR, Path, _dir_of_current_file, cwd
