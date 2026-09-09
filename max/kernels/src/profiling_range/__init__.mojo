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

"""Provides Mojo wrappers for MAX's libkineto-backed range profiler.

Provides the `Range` context manager and the `is_enabled()` query for
instrumenting GPU and CPU code with named activity spans visible in profiling
tools such as Nsight and the Kineto-based trace viewer.

```mojo
from profiling_range import Range

with Range("my_span"):
    ...  # work to profile
```

The package is named `profiling_range` to avoid shadowing Mojo's built-in
`range()` iterator.
"""

from .range import Range, is_enabled
