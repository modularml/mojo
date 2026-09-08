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

# Test that --experimental-export-fixit generates a YAML file and continues
# normal execution. Since the package has deprecation warnings with fix-its,
# the package build will fail, but the YAML file should still be written.
# RUN: not mojo precompile --experimental-export-fixit=%t.yaml %S/test_package_fixit 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file=%t.yaml --check-prefix=YAML

# CHECK: error: use of unknown declaration '__origin_of'; did you mean 'origin_of'?
# CHECK: Fix-its exported to:
# CHECK: Apply with: 'clang-apply-replacements

# Verify YAML format (using LLVM's yaml::Output)
# YAML: ---
# YAML: MainSourceFile:
# YAML: Replacements:
# YAML: - FilePath:
# YAML: Offset:
# YAML: Length:
# YAML: ReplacementText: origin_of
# YAML: ...

# Test mutual exclusion of --experimental-fixit and --experimental-export-fixit.
# RUN: not mojo precompile --experimental-fixit --experimental-export-fixit=%t2.yaml %S/test_package_fixit 2>&1 | FileCheck %s --check-prefix=ERROR
# ERROR: cannot use both --experimental-fixit and --experimental-export-fixit simultaneously
