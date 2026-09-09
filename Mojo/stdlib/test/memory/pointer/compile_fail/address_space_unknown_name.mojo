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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# An address-space name with no built-in constant and no definition from the
# active `PluginHooks` backend is a compile-time error, rather than silently
# constructing a bogus address space. (On a build with a backend that defines
# `SCRATCHPAD`, this same access resolves to that backend's value.)


def main():
    # CHECK: unknown address space: 'SCRATCHPAD'
    _ = AddressSpace.SCRATCHPAD
