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

# A failed import must not poison its name for the rest of the compilation:
# inputs/imported_module.mojo genuinely exists, so it must still be importable
# after the lookup of a same-named submodule of test_package fails. The bad
# call at the end is the tripwire proving the real module was bound: it is
# only diagnosed when `imported_module` resolves to the real module rather
# than binding a silent error state.

# RUN: %parse-mojo-isolated -verify-diagnostics -I=%S/inputs %s

# expected-error @+1 {{unable to locate module 'imported_module'}}
import test_package.imported_module

import imported_module

def main():
    # expected-error @+1 {{invalid call to 'imported_fn': unexpected argument}}
    imported_module.imported_fn(42)
