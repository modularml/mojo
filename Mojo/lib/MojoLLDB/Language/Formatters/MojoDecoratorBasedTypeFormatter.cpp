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

#include "MojoDecoratorBasedTypeFormatter.h"
#include "../../TypeSystem/MojoTypeSystem.h"
#include "../../Utils/Errors.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;

namespace {
/// Returns true for multi-element, non-bool SIMD vector types
/// (e.g. `!kgen.simd<4, si32>`), excluding scalars (`!kgen.simd<1, …>`)
/// and bool vectors (`!kgen.simd<N, bool>`).
static bool isMultiElementNonBoolSIMD(llvm::StringRef typeName) {
  return typeName.starts_with("!kgen.simd<") &&
         !typeName.starts_with("!kgen.simd<1,") &&
         !typeName.ends_with(", bool>");
}

/// Synthetic type front end corresponding to the @lldb_formatter_wrapping_type
/// decorator. It replaces a variable with its first child.
class WrappingTypeSyntheticFrontEnd
    : public lldb_private::SyntheticValueProviderFrontEnd {
public:
  WrappingTypeSyntheticFrontEnd(lldb_private::ValueObject &backend)
      : SyntheticValueProviderFrontEnd(backend) {}

  lldb::ValueObjectSP GetSyntheticValue() override {
    if (!m_backend.MightHaveChildren() ||
        getExpectedValueOr(m_backend.GetNumChildren(), 0u) == 0)
      return {};
    return m_backend.GetChildAtIndex(0, /*can_create=*/true);
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!MightHaveChildren())
      return 0;
    return GetSyntheticValue()->GetNumChildren();
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    return GetSyntheticValue()->GetChildAtIndex(idx);
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    return GetSyntheticValue()->GetIndexOfChildWithName(name);
  }

  bool MightHaveChildren() override {
    // If the summary provider for this child asks for no children, then we
    // simply report as this type has no children, otherwise structs like `Bool`
    // are displayed with its nested `i1` field.
    //
    // Exception: for multi-element types like Tuple's !kgen.struct, the
    // HideChildren flag is intended to suppress redundant children on the
    // pack itself, not to hide Tuple's elements from the wrapping display.
    lldb::ValueObjectSP sv = GetSyntheticValue();
    if (!sv)
      return false;
    // SIMD vectors (multi-element, non-bool) display all values in their
    // summary as [v0, v1, …]. Hide the raw __mlir_type.* children that would
    // otherwise appear in the expanded view.
    if (isMultiElementNonBoolSIMD(sv->GetTypeName().GetStringRef()))
      return false;
    lldb::TypeSummaryImplSP typeSummary = sv->GetSummaryFormat();
    if (typeSummary && (typeSummary->GetOptions() & eTypeOptionHideChildren) &&
        getExpectedValueOr(sv->GetNumChildren(), 0u) <= 1)
      return false;
    return sv->MightHaveChildren();
  }
};
} // namespace

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::MojoLLDBWrappingTypeTypeSyntheticFrontEndCreator(
    CXXSyntheticChildren *x, const ValueObjectSP &valobjSP) {
  return new WrappingTypeSyntheticFrontEnd(*valobjSP);
}

bool M::KGEN::Mojo::MojoLLDBWrappingTypeSummaryProvider(
    ValueObject &valobj, Stream &stream, TypeSummaryOptions summaryOptions) {
  ValueObjectSP nonSyntheticValobj = valobj.GetNonSyntheticValue();
  ValueObjectSP impl = nonSyntheticValobj->GetChildAtIndex(0);
  if (!impl)
    return false;
  std::string dest;
  impl->GetSummaryAsCString(dest, summaryOptions);
  if (!dest.empty()) {
    stream << dest;
    return true;
  }

  // If the inner value is a multi-element non-bool SIMD vector, format it as
  // [v0, v1, ..., vN] rather than falling through to show nothing. This avoids
  // the verbose "(__mlir_type.si32) [0] = 5" expanded-children display that
  // would otherwise appear for Mojo SIMD[DType.T, N] wrapper types.
  if (isMultiElementNonBoolSIMD(impl->GetTypeName().GetStringRef())) {
    auto numChildren = getExpectedValueOr(impl->GetNumChildren(), 0u);
    stream.PutCString("[");
    for (size_t i = 0; i < numChildren; ++i) {
      ValueObjectSP child = impl->GetChildAtIndex(i);
      if (i > 0)
        stream.PutCString(", ");
      if (!child) {
        stream.PutCString("<error>");
        continue;
      }
      const char *value = child->GetValueAsCString();
      stream.PutCString(value ? value : "<error>");
    }
    stream.PutCString("]");
    return true;
  }

  // Fall back to scalar value if available (e.g. index, si8, f32).
  if (const char *val = impl->GetValueAsCString()) {
    stream << val;
    return true;
  }
  return false;
}
