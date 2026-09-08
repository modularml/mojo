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

from .old_impl import *


# CHECK-LABEL: def old_origin_of_2
def old_origin_of_2[T: AnyType](c: T):
    # CHECK-NEXT: _ = origin_of(c)
    # CHECK-NOT: _ = __origin_of(c)
    _ = __origin_of(c)
