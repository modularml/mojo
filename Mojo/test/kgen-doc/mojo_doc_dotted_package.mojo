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

# RUN: kgen-doc %S/dotted.pkg | FileCheck %s

# A package directory (and sub-package/module) whose name contains periods
# keeps its whole name in the generated docs: the periods are part of the
# name, not an extension to strip.

# CHECK:  "kind": "package"
# CHECK:  "modules":
# CHECK:    "name": "__init__"
# CHECK:    "name": "module.with.dots"
# CHECK:  "name": "dotted.pkg"
# CHECK:  "packages":
# CHECK:    "name": "sub.pkg"
# CHECK:  "summary": "This is a dotted package."
