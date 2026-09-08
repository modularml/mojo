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
"""Language foundation: built-in types, traits, and fundamental operations.

The `builtin` package provides the core types, traits, and operations that form
the foundation of the Mojo language. It defines fundamental types like integers,
booleans, and strings, essential traits for type behavior (`Copyable`, `Movable`,
`Comparable`), and basic operations used throughout all Mojo code. Most symbols
from this package are automatically available without explicit imports through
the prelude.

This package is implicitly imported. It
defines the core vocabulary of Mojo programming that every developer uses
without thinking about imports. Library authors implement traits from this
package to integrate custom types with language features.
"""
