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
"""Minimal standalone `std` for the missing-import suggestion tests (MOCO-1051).

This is a purpose-built, fully self-contained stand-in for the standard library,
precompiled to a `std.mojoc` so the tests exercise the feature's real bytecode
path (the form shipped to users). It is intentionally tiny: the Mojo compiler
only hard-requires the `std`, `std.prelude`, and `std.builtin` packages to
*exist*, so their `__init__.mojo` files can be empty. The `import_suggestion`
sub-package holds the actual test fixtures.
"""
