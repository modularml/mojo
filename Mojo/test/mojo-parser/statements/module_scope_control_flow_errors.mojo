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

# Test that control flow statements at module/file scope emit errors instead
# of being silently accepted.

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Control flow statements at module scope
##===----------------------------------------------------------------------===##

# expected-error @below {{'if' must be contained in a function}}
if True:
    pass

# expected-error @below {{'for' must be contained in a function}}
for i in range(10):
    pass

# expected-error @below {{'while' must be contained in a function}}
while True:
    pass

# expected-error @below {{'with' must be contained in a function}}
with foo:
    pass

##===----------------------------------------------------------------------===##
# Comptime control flow statements at module scope
##===----------------------------------------------------------------------===##

# expected-error @below {{'comptime if' must be contained in a function}}
comptime if True:
    pass

# expected-error @below {{'comptime for' must be contained in a function}}
comptime for i in range(10):
    pass
