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

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# The shorthand is mapped by name so that `@inline` never instantiates
# `InlineLevel`, whose own conformances carry the decorator. That mapping is
# separate from the one the qualified spelling reaches through type checking,
# so each level is pinned through both spellings to keep the two in agreement.


# CHECK: lit.fn @"shorthand_always()"() -> index always_inline
@inline(.always)
def shorthand_always() -> __mlir_type.index:
    return Int(1).__mlir_index__()


# CHECK: lit.fn @"qualified_always()"() -> index always_inline
@inline(InlineLevel.always)
def qualified_always() -> __mlir_type.index:
    return Int(1).__mlir_index__()


# CHECK: lit.fn @"shorthand_nodebug()"() -> index always_inline_no_debug
@inline(.nodebug)
def shorthand_nodebug() -> __mlir_type.index:
    return Int(2).__mlir_index__()


# CHECK: lit.fn @"qualified_nodebug()"() -> index always_inline_no_debug
@inline(InlineLevel.nodebug)
def qualified_nodebug() -> __mlir_type.index:
    return Int(2).__mlir_index__()


# CHECK: lit.fn @"shorthand_never()"() -> index no_inline
@inline(.never)
def shorthand_never() -> __mlir_type.index:
    return Int(3).__mlir_index__()


# CHECK: lit.fn @"qualified_never()"() -> index no_inline
@inline(InlineLevel.never)
def qualified_never() -> __mlir_type.index:
    return Int(3).__mlir_index__()


# CHECK: lit.fn @"shorthand_automatic()"() -> index attributes
@inline(.automatic)
def shorthand_automatic() -> __mlir_type.index:
    return Int(4).__mlir_index__()


# CHECK: lit.fn @"qualified_automatic()"() -> index attributes
@inline(InlineLevel.automatic)
def qualified_automatic() -> __mlir_type.index:
    return Int(4).__mlir_index__()
