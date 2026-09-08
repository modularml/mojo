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

# Tests that you can import modules and packages through (nested) directories
# using the same syntax as through packages.

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

import import_through_dir.module

# // -----

from import_through_dir import module

# // -----

from import_through_dir.module import foo

# // -----

# Nested directories

import import_through_dir.nested_dir.module

# // -----

from import_through_dir.nested_dir import module

# // -----

from import_through_dir.nested_dir.module import bar

# // -----

# Packages inside nested directories

import import_through_dir.nested_dir.nested_package.module

# // -----

from import_through_dir.nested_dir.nested_package import module

# // -----

from import_through_dir.nested_dir.nested_package import baz

# // -----

from import_through_dir.nested_dir.nested_package.module import baz2
