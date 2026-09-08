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

# One file reached under two names is one module, so the second name aliases the
# first rather than declaring a second copy of every type in it. `lib` and
# `lib/mylib` both reach the same `utils.mojo`.
#
# Both names are used below, since an unused import is never resolved and so
# would never bind the module a second time. `Thing` comes from one name and the
# value from the other, so this only type-checks if the two are one type.

# RUN: %parse-mojo-isolated -verify-diagnostics \
# RUN:   -I=%S/inputs/dual_mount/lib -I=%S/inputs/dual_mount/lib/mylib %s

# expected-note @below {{'mylib.utils' is the name used in error messages and debug info}}
from mylib.utils import Thing
# expected-warning @below {{'utils' and 'mylib.utils' name the same module; remove the duplicate import root or file that reaches it twice}}
from utils import make


def unwrap() -> Int:
    var value: Thing = make()
    return value.x
