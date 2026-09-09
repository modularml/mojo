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

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# Error paths for dotted-name imports: diagnostics must render the dotted
# component names, and structural errors must fire for dotted segments just as
# they do for plain ones.

# expected-error @+1 {{unable to locate module 'no.such.pkg'}}
import `no.such.pkg`

# // -----

# A dotted *module* is still a module: importing through it is an error.
# expected-error @+1 {{'module.with.dots' is a module, not a package; it has no nested module or package 'nested'}}
import `dotted.pkg`.`module.with.dots`.nested

# // -----

# A dotted name that only matches a *prefix* of an on-disk entry must not
# resolve ('sub.pkg' exists only inside dotted.pkg, not at the top level).
# expected-error @+1 {{unable to locate module 'sub.pkg'}}
import `sub.pkg`
