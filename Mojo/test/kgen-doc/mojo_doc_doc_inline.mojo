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

# An import is normally absent from the importing module's docs. @__doc_inline
# documents it there using the target's own declaration, so the docstring is
# never duplicated at the import site.

# HiddenStruct is banned everywhere; NOT_INLINED_ALIAS is checked separately
# below, since it is legitimately documented under src.
# RUN: kgen-doc %S/test_doc_inline | FileCheck %s --implicit-check-not HiddenStruct
# RUN: kgen-doc %S/test_doc_inline | FileCheck %s --check-prefix=ONCE

# The first module documented is __init__, which imports all of src's symbols
# but decorates only some of them.

# CHECK:      "name": "INLINED_ALIAS"
# CHECK:      "summary": "Docstring of the alias."
# CHECK:      "name": "inlined_func"
# CHECK:      "summary": "Docstring of the function."
# CHECK:      "name": "InlinedStruct"
# CHECK:      "summary": "Docstring of the struct."
# CHECK:      "summary": "Re-exports src with inlined documentation."

# The undecorated import reaches src's docs and nowhere else, so a leak into
# __init__ would show up as a second occurrence rather than a first.
# ONCE-COUNT-1: "name": "NOT_INLINED_ALIAS"
# ONCE-NOT:     "name": "NOT_INLINED_ALIAS"
