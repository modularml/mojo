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

# RUN: mojo-lsp-simple-client --fail-on-diagnostics %s > %t 2>&1; true
# RUN: FileCheck %s < %t

# A docstring code block is parsed in a synthetic wrapper buffer (a distinct
# buffer from this file), but an import of this module written inside it is
# still a self-import of the module the wrapper wraps, and must be rejected as
# one rather than resolving the module into its own docstring example.
#
# CHECK: module 'docstring_self_import' cannot import itself


def example():
    """Example whose code block self-imports the enclosing module.

    ```mojo
    import docstring_self_import
    ```
    """
    pass
