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

# Modules and packages are importable through plain source directories (no
# __init__.mojo), so their API surface must show up in generated docs too.
# `dir` here is a plain directory inside the package.

# RUN: kgen-doc %S/test_dir_package | FileCheck %s

# CHECK:  "kind": "package"
# CHECK:  "modules":
# CHECK:    "name": "__init__"
# CHECK:  "name": "test_dir_package"
# CHECK:  "packages":
# CHECK:    "name": "dir_fn"
# CHECK:    "summary": "Does directory things."
# CHECK:    "name": "module"
# CHECK:    "summary": "A module inside a plain directory."
# CHECK:    "name": "dir"
# CHECK:  "summary": "This is a package with a plain source directory."
