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

# ===----------------------------------------------------------------------=== #
# Package file
# ===----------------------------------------------------------------------=== #
"""Basic mathematical utilities.

This package defines a collection of utility functions for manipulating
integer values.

You can import these APIs from the `my_math` package. For example:

```mojo
from my_math import dec, inc
```
"""

from .utils import dec, inc
