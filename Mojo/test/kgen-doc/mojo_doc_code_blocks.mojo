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

# RUN: kgen-doc %s | FileCheck %s

"""Module Summary.

Handle `%#`:

```mojo
# Simple statements.
%# var value = 5
print(value)

# Multi-line/complex statements.
%# def return_value() -> Int:
%#   return 10
print(return_value())
```
"""

# CHECK: Handle `%#`:
# Check that the hidden lines are not displayed in the documentation.
# CHECK-SAME: # Simple statements.\nprint(value)
# CHECK-SAME: # Multi-line/complex statements.\nprint(return_value())
