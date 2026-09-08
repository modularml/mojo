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

"""A package that re-exports a symbol and contains sibling modules.

`shared_fn` is re-exported here so it lives in __init__'s scope, and `helpers`
is a sibling module. Neither is a member of the `other` module - a *qualified*
access like `other.shared_fn` / `other.helpers` must therefore be a hard error,
not a deprecated-intra-package resolution against this package.
"""

from .helpers import shared_fn
