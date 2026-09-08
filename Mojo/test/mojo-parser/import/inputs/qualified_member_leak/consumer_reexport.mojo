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

# `other` is imported explicitly, but `shared_fn` is NOT a member of `other`
# (it lives in the package's __init__). A qualified `other.shared_fn` must be a
# hard error - the intra-package __init__ fallback must not fire for a member
# access whose base is some other module.
from . import other


def consume() -> Int:
    return other.shared_fn()
