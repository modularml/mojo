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
"""Core object-lifetime and value-semantics traits.

The `traits` package defines the stabilized traits that describe how Mojo
values are created, copied, moved, and destroyed: `AnyType`, `Deinitable`,
`Movable`, `Copyable`, and `ImplicitlyCopyable`. These traits are Mojo
built-ins and are automatically imported into every Mojo program through the
prelude.
"""

from .anytype import AnyType
from .copyable import Copyable, ImplicitlyCopyable, IsTriviallyCopyable
from .deinitable import Deinitable, IsTriviallyDeinitable
from .movable import Movable, IsTriviallyMovable
