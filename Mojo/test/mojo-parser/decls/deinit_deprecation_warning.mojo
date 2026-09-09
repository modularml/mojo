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

# Test that the deprecated '__deinit__' destructor spelling is diagnosed with a
# warning and a fix-it to the canonical '__deinit__' spelling, while the
# canonical spelling itself is silent.
# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Verify the fix-it text itself via the JSON diagnostic format.
# RUN: %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s --check-prefix=FIXIT

# FIXIT: "diagnostic":{
# FIXIT-SAME: "fixIts":[{
# FIXIT-SAME: "text":"__deinit__"


struct UsesOldSpelling:
    var x: Int

    def __init__(out self):
        self.x = 0

    # expected-warning @+1 {{'__del__' is deprecated; use '__deinit__'}}
    def __del__(deinit self):
        pass


struct UsesNewSpelling:
    var x: Int

    def __init__(out self):
        self.x = 0

    def __deinit__(deinit self):
        pass
