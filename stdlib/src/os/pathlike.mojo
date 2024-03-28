# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #
"""Implements PathLike trait.

You can import the trait from the `os` package. For example:

```mojo
from os import PathLike
```
"""


trait PathLike:
    """A trait representing file system paths."""

    fn __fspath__(self) -> String:
        """Return the file system path representation of the object.

        Returns:
          The file system path representation as a string.
        """
        ...
