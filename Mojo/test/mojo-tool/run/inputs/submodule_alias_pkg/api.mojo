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

# Aliased from-import of a sibling submodule (mirrors `from std.sys import
# _libc as libc`): this binds a gated ImportOp under `impl` over the `_impl`
# submodule. The gate must not be serialized alongside its now-dead
# `unresolved_import` placeholder, or reloading this package fails with
# "invalid redefinition of 'impl'".
from . import _impl as impl


def api_value() -> Int:
    return impl.impl_value()
