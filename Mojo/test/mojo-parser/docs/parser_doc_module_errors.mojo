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

# RUN: %parse-mojo-isolated -o /dev/null -mojo-diagnose-missing-doc-strings -verify-diagnostics %s
# FileModuleOp is located at line 1, col 1 of the buffer, so the diagnostic
# fires there. @-16 counts back from this line to line 1.
# expected-warning @-16 {{public module 'parser_doc_module_errors' is missing a doc string}}

# A public module without a doc string should raise a diagnostic.
# This file intentionally has no module-level doc string.


struct ArgStruct(Movable where False):
    """A stub type for arguments."""

    pass
