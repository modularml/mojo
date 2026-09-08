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

# One directory reached under two names is one package, so the types it declares
# exist once. Overlapping import roots are the easy way to reach that: `lib` and
# `lib/mylib` both name the same `utils`.

# RUN: %parse-mojo-isolated -verify-diagnostics \
# RUN:   -I=%S/inputs/dual_mount_pkg/lib \
# RUN:   -I=%S/inputs/dual_mount_pkg/lib/mylib %s

# expected-note @below {{'mylib.utils' is the name used in error messages and debug info}}
from mylib.utils.a import Thing, make
# expected-warning @below {{'utils' and 'mylib.utils' name the same package; remove the duplicate import root or file that reaches it twice}}
from utils.a import Thing as SameThing


def unwrap(value: SameThing) -> Int:
    return value.x


# Passing one name's type where the other is expected is what used to fail with
# `cannot implicitly convert 'Thing' value to 'Thing'`.
def use() -> Int:
    var value: Thing = make()
    return unwrap(value)
