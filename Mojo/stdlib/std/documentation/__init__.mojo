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
"""Documentation built-ins: decorators and utilities for doc generation.

The `documentation` package provides tools for controlling documentation
generation and visibility in Mojo. It offers decorators and utilities that
influence how APIs appear in generated documentation, allowing library authors
to hide implementation details while maintaining clean public interfaces.

Use this package when authoring libraries to control which symbols appear in
generated documentation, hide internal implementation details, or manage the
public API surface shown to users.
"""

from .documentation import doc_hidden
