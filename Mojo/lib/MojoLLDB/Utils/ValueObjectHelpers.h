//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_UTILS_VALUEOBJECTHELPERS_H
#define KGEN_LIB_MOJOLLDB_UTILS_VALUEOBJECTHELPERS_H

#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"

namespace M::KGEN::Mojo {

/// Look up a child by name on `parent` and return its non-synthetic view,
/// or null if the child is missing, errored, or (defensively) doesn't have
/// a non-synthetic backing. Collapses the
/// `GetChildMemberWithName` + error-check + `GetNonSyntheticValue` +
/// null-check pattern used at multiple Mojo formatter entry points.
inline lldb::ValueObjectSP nonSyntheticChild(lldb_private::ValueObject &parent,
                                             const char *name) {
  lldb::ValueObjectSP child = parent.GetChildMemberWithName(name);
  if (!child || !child->GetError().Success())
    return nullptr;
  return child->GetNonSyntheticValue();
}

/// Walk through Mojo's single-field LIT/KGEN wrapper types
/// (`_mlir_value`, `value`, `address`, `_value`, `_storage`) until we land on
/// a plain pointer or scalar ValueObject. Also descends through a
/// `!kgen.simd<1, ...>` wrapper (as used by `Scalar[T]`) into its single
/// element to reach the underlying integer/float scalar that LLDB doesn't
/// recognize as scalar directly. Leaves the input unchanged if it already
/// is a scalar/pointer, or if none of the known wrappers apply.
///
/// Used by the stdlib `String` summary provider and the `Variant` formatter
/// to dig through LIT struct / REPL wrapping to the raw pointer/scalar that
/// holds a primitive value.
///
/// NOTE: Each iteration calls GetChildMemberWithName on potentially synthetic
/// ValueObjects. Call sites should only pass children known to exist for the
/// type (e.g., _len/_data after resolving to a Span). Passing an arbitrary
/// synthetic ValueObject whose children provider uses cantFail may crash if
/// none of the known field names exist and the type is not yet scalar or
/// pointer.
inline lldb::ValueObjectSP unwrapToScalarOrPointer(lldb::ValueObjectSP field) {
  static constexpr const char *kWrapperNames[] = {
      "_mlir_value", "value", "address", "_value", "_storage"};
  while (field && !field->IsPointerType() && !field->IsScalarType()) {
    bool unwrapped = false;
    for (const char *name : kWrapperNames) {
      if (auto inner = field->GetChildMemberWithName(name)) {
        field = inner;
        unwrapped = true;
        break;
      }
    }
    if (unwrapped)
      continue;
    // Handle Scalar[T] (i.e. SIMD[T, 1]): the _mlir_value is a 1-element
    // SIMD vector that LLDB does not recognize as a scalar type, so
    // GetValueAsUnsigned fails on it directly. Descend into the single
    // child element to reach the underlying integer scalar.
    //
    // We check the type name explicitly to avoid descending into arbitrary
    // 1-child wrappers that happen to lack a named member field.
    if (llvm::StringRef(field->GetTypeName()).starts_with("!kgen.scalar<")) {
      if (auto inner = field->GetChildAtIndex(0)) {
        field = inner;
        continue;
      }
    }
    break;
  }
  return field;
}

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_UTILS_VALUEOBJECTHELPERS_H
