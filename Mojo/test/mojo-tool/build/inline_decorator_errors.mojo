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

# `@inline` takes an `InlineLevel`, so a parameter of another type is rejected
# where the decorator is written rather than at the instantiation.

# RUN: not %mojo-build %s -o /dev/null 2>&1 | FileCheck %s


# CHECK: error: cannot implicitly convert 'Int' value to 'InlineLevel'
@inline(policy)
def parametric[policy: Int]() -> Int:
    return 1


def main():
    print(parametric[1]())
