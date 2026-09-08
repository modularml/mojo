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

# A standalone module cannot import its own bare name: it would only ever
# resolve to itself, silently shadowing any same-named system package.
# Self-imports inside packages are unaffected (see
# import_relative_self_reexport.mojo and import_self_named_package.mojo).

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics %s

# expected-error @+1 {{module 'import_self' cannot import itself}}
import import_self

def main():
    import_self.foo()

def foo(): pass

# // -----

# expected-error @+1 {{module 'import_self' cannot import itself}}
from import_self import foo

def main():
    foo()

# // -----

# Aliasing doesn't make the self-import any less ambiguous.

# expected-error @+1 {{module 'import_self' cannot import itself}}
import import_self as myself

def main():
    myself.foo()

def foo(): pass

# // -----

# A dotted path is rejected at its first component.

# expected-error @+1 {{module 'import_self' cannot import itself}}
import import_self.submodule

def main():
    pass
