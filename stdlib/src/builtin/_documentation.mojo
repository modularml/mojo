# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Provides decorators and utilities for interacting with Mojo documentation
generation and validation.

These are Mojo built-ins, so you don't need to import them.
"""

# ===-------------------------------------------------------------------===#
# doc_private
# ===-------------------------------------------------------------------===#


fn doc_private():
    """Indicate that the decorated declaration is private from the viewpoint
    of documentation generation.

    This decorator allows for hiding the documentation for a declaration during
    generation. This is often used to hide `__init__`, and other special
    methods, that are intended for internal consumption.

    For example:

    ```mojo
    struct Foo:
      @doc_private
      fn __init__(inout self):
        "This should not be called directly, prefer Foo.create instead."
        return

      @staticmethod
      fn create() -> Self:
        return Self()
    ```
    """
    return
