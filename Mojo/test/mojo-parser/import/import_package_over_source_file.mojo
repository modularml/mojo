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

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s | FileCheck %s

# This test checks that we pick the source package named test_package over the
# identically named test_package.mojo module in the same directory. If we did,
# we wouldn't see lit.package below; we'd see lit.file_module.

# CHECK: lit.package @test_package
import test_package

def main():
  test_package.method_defined_in_init()
