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

#include "MojoKGENVariantTypeFormatter.h"
#include "../../Utils/Errors.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

MojoKGENVariantTypeSyntheticFrontEnd::MojoKGENVariantTypeSyntheticFrontEnd(
    const lldb::ValueObjectSP &backend)
    : SyntheticChildrenFrontEnd(*backend), content() {
  if (backend)
    Update();
}

llvm::Expected<uint32_t>
MojoKGENVariantTypeSyntheticFrontEnd::CalculateNumChildren() {
  return 1;
}

lldb::ValueObjectSP
MojoKGENVariantTypeSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= 1)
    return ValueObjectSP();
  return content;
}

lldb::ChildCacheState MojoKGENVariantTypeSyntheticFrontEnd::Update() {
  std::optional<ValueObjectSP> parsed =
      MojoKGENVariantTypeSyntheticFrontEnd::parseKGENVariant(m_backend.GetSP());
  if (!parsed)
    return lldb::ChildCacheState::eRefetch;

  content = *parsed;
  return lldb::ChildCacheState::eRefetch;
}

std::optional<ValueObjectSP>
MojoKGENVariantTypeSyntheticFrontEnd::parseKGENVariant(
    lldb::ValueObjectSP valobj) {
  valobj = valobj->GetNonSyntheticValue();
  if (!valobj || !valobj->GetError().Success())
    return {};

  auto numChildren = getExpectedValueOr(valobj->GetNumChildren(), 0u);
  if (numChildren < 1)
    return {};

  // The discriminator is the last field.
  ValueObjectSP discrField = valobj->GetChildAtIndex(numChildren - 1);
  if (!discrField || !discrField->GetError().Success())
    return {};

  bool success = true;
  size_t discr = discrField->GetValueAsUnsigned(0, &success);
  if (!success)
    return {};

  if (discr >= numChildren - 1)
    return {};

  ValueObjectSP dataVal = valobj->GetChildAtIndex(discr);
  if (!dataVal || !dataVal->GetError().Success())
    return {};

  return dataVal;
}

bool MojoKGENVariantTypeSyntheticFrontEnd::MightHaveChildren() { return true; }

llvm::Expected<size_t>
MojoKGENVariantTypeSyntheticFrontEnd::GetIndexOfChildWithName(
    lldb_private::ConstString targetName) {
  if (content->GetName() == targetName)
    return 0;
  return llvm::createStringError("Child not found");
}

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::mojoKGENVariantSyntheticFrontEndCreator(
    CXXSyntheticChildren *, const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  CompilerType type = valobjSP->GetCompilerType();
  if (!type.IsValid())
    return nullptr;
  return new M::KGEN::Mojo::MojoKGENVariantTypeSyntheticFrontEnd(valobjSP);
}
