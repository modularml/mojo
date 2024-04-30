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
"""Implements functions and variables for interacting with execution and system
environment.

You can import these APIs from the `sys` package. For example:

```mojo
from sys import argv
```
"""

from sys import external_call

from memory import UnsafePointer

from utils import StringRef


# TODO: When we have global variables, this should be a global list.
fn argv() -> VariadicList[StringRef]:
    """The list of command line arguments.

    Returns:
        The list of command line arguments provided when mojo was invoked.
    """
    var result = VariadicList[StringRef]("")
    external_call["KGEN_CompilerRT_GetArgV", NoneType](
        UnsafePointer[VariadicList[StringRef]].address_of(result)
    )
    return result
