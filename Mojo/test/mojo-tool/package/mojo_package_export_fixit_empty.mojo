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

# Test that --experimental-export-fixit creates a YAML file even when there
# are no fix-its (unlike clang-tidy semantics: we always create the file).
# RUN: rm -f %t3.yaml
# RUN: mojo precompile --experimental-export-fixit=%t3.yaml %S/test_package_fixit_patch_empty -o %t3.mojoc 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file=%t3.yaml --check-prefix=YAML

# CHECK: Fix-its exported to:
# CHECK: Apply with: 'clang-apply-replacements

# Verify YAML file is created with empty replacements
# YAML: ---
# YAML: MainSourceFile:
# YAML: Replacements:  []
# YAML: ...
