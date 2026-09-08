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

# Test that JSON diagnostics include fixit information.
# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s

# Verify the JSON contains fixit with expected structure.
# CHECK: "diagnostic":{
# CHECK-SAME: "fixIts":[{
# CHECK-SAME: "end":{
# CHECK-SAME: "start":{
# CHECK-SAME: "text":"origin_of"


def test_fixit[T: AnyType](a: T):
    # __origin_of is deprecated and suggests origin_of as a fixit
    _ = __origin_of(a)


def main():
    test_fixit(1)
