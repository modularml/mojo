#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##
set -euo pipefail

# Install `lit` for use in the tests
brew install lit

# Ensure `FileCheck` from the pre-installed LLVM 15 package is visible
echo $(brew --prefix llvm@15)/bin/ >> $GITHUB_PATH
