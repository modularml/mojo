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

# Test that --diagnostic-format json produces valid JSON with expected structure.
# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s --check-prefix=JSON

# Verify the JSON contains expected fields and structure.
# JSON: "diagnostic":{
# JSON-SAME: "file":
# JSON-SAME: "fixIts":
# JSON-SAME: "location":{
# JSON-SAME: "column":
# JSON-SAME: "line":
# JSON-SAME: "text":
# JSON-SAME: "kind":"error"
# JSON-SAME: "message":"use of unknown declaration 'unknown_identifier'"

# Test that default text format still works.
# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s --check-prefix=TEXT
# TEXT: error:

# Test that JSON format with MLIR diagnostics is an error.
# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=true %s 2>&1 | FileCheck %s --check-prefix=ERROR
# ERROR: error: --diagnostic-format=json is incompatible with --use-mlir-diagnostics=true


def main():
    _ = unknown_identifier
