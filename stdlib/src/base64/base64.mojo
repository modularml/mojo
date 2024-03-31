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
"""Provides functions for base64 encoding strings.

You can import these APIs from the `base64` package. For example:

```mojo
from base64 import b64encode
```
"""

from collections import List
from sys.info import simdwidthof

from memory.unsafe import DTypePointer

# ===----------------------------------------------------------------------===#
# b64encode
# ===----------------------------------------------------------------------===#


fn b64encode(str: String) -> String:
    """Performs base64 encoding on the input string.

    Args:
      str: The input string.

    Returns:
      Base64 encoding of the input string.
    """
    alias lookup = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    var b64chars = lookup.data()

    var length = len(str)
    var out = List[Int8](capacity=length + 1)

    @parameter
    @always_inline
    fn s(idx: Int) -> Int:
        return int(str._as_ptr().bitcast[DType.uint8]()[idx])

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    var end = length - (length % 3)
    for i in range(0, end, 3):
        var si = s(i)
        var si_1 = s(i + 1)
        var si_2 = s(i + 2)
        out.append(b64chars.load(si // 4))
        out.append(b64chars.load(((si * 16) % 64) + si_1 // 16))
        out.append(b64chars.load(((si_1 * 4) % 64) + si_2 // 64))
        out.append(b64chars.load(si_2 % 64))

    var i = end
    if i < length:
        var si = s(i)
        out.append(b64chars.load(si // 4))
        if i == length - 1:
            out.append(b64chars.load((si * 16) % 64))
            out.append(ord("="))
        elif i == length - 2:
            var si_1 = s(i + 1)
            out.append(b64chars.load(((si * 16) % 64) + si_1 // 16))
            out.append(b64chars.load((si_1 * 4) % 64))
        out.append(ord("="))
    out.append(0)
    return String(out^)
