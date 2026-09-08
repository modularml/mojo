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

# Import `foo` along two paths that both resolve to `source.foo`: directly and
# through `relay`. Under keep-and-gate each path becomes its own resolved
# `lit.import` gate, so this module's scope ends up holding two gates for `foo`.
# Using `foo` forces both gates to resolve during precompile, so they are
# serialized as *resolved* gates. Reloading this bytecode must collapse the two
# gates rather than reject the second as an "invalid redefinition".

from .relay import foo
from .source import foo


def use_foo() -> Int:
    return foo()
