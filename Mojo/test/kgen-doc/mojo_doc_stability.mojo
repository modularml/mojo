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

# Test that isStabilityTracked reflects whether a declaration is in a package
# opted into stability tracking ("std" or "test_std_mock"), and that isStable
# reflects the presence of the @stable decorator.
#
# RUN 1: kgen-doc on test_std_mock (an opted-in package).
# RUN 2: kgen-doc on this file (outside any opted-in package).

# RUN: kgen-doc %S/test_std_mock | FileCheck %s --check-prefix TRACKED
# RUN: kgen-doc %s | FileCheck %s --check-prefix UNTRACKED

# ---- Stability-tracked package: test_std_mock ----

# @stable(since="1.0") populates sinceVersion in the function overload.
# Check this before the "structs": label since functions precede structs
# alphabetically in the JSON output.
# TRACKED: "sinceVersion": "1.0"

# Each struct has an "aliases" array (containing the synthesized
# __del__is_trivial alias) that appears before "isStabilityTracked" in the
# alphabetically-sorted JSON. Use "functions": [] as an anchor to reach
# the struct-level isStabilityTracked / isStable fields.
# Structs appear alphabetically: StableStruct before UnstableStruct.

# TRACKED-LABEL: "structs":
# TRACKED: "functions": [],
# TRACKED-NEXT: "isStabilityTracked": true,
# TRACKED-NEXT: "isStable": true,
# TRACKED: "name": "StableStruct",
# TRACKED: "functions": [],
# TRACKED-NEXT: "isStabilityTracked": true,
# TRACKED-NEXT: "isStable": false,
# TRACKED: "name": "UnstableStruct",

# ---- Non-tracked package: this file ----
#
# Declarations in a standalone file have isStabilityTracked: false.
# Same "functions": [] anchor applies.

# UNTRACKED-LABEL: "structs":
# UNTRACKED: "functions": [],
# UNTRACKED-NEXT: "isStabilityTracked": false,
# UNTRACKED-NEXT: "isStable": false,
# UNTRACKED: "name": "UntrackedStruct",


struct UntrackedStruct:
    """A struct outside any stability-tracked package."""

    pass
