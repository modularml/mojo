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

# Test that 'try' statement at module scope emits an error.
# This is in a separate file because after 'try' is rejected, the 'except'
# keyword becomes an invalid token at module scope, causing a cascading error.

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# expected-error @below {{'try' must be contained in a function}}
try:
    pass
# expected-error @below {{unexpected token in expression}}
except:
    pass
