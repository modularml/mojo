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

#include "MojoListTypeFormatter.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/lib/MojoLLDB/Utils/ValueObjectHelpers.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

MojoListSyntheticFrontEnd::MojoListSyntheticFrontEnd(
    const lldb::ValueObjectSP &backend)
    : SyntheticChildrenFrontEnd(*backend), start(0), size(0), elementType(),
      elementSize(0) {
  if (backend)
    Update();
}

llvm::Expected<uint32_t> MojoListSyntheticFrontEnd::CalculateNumChildren() {
  return size;
}

lldb::ValueObjectSP MojoListSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= size)
    return ValueObjectSP();
  uint64_t addr = start + (idx * elementSize);
  return CreateChildValueObjectFromAddress(
      llvm::formatv("[{0}]", idx).str(), addr,
      m_backend.GetExecutionContextRef(), elementType);
}

lldb::ChildCacheState MojoListSyntheticFrontEnd::Update() {
  std::optional<std::pair<ValueObjectSP, size_t>> parsed =
      MojoListSyntheticFrontEnd::parseList(m_backend.GetSP());
  if (!parsed)
    return lldb::ChildCacheState::eRefetch;

  ValueObjectSP data = parsed->first;
  start = data->GetPointerValue().address;
  elementType = data->GetCompilerType().GetPointeeType();
  if (elementType.IsValid()) {
    auto exeCtxScope = ExecutionContext(m_backend.GetExecutionContextRef())
                           .GetBestExecutionContextScope();
    if (llvm::Expected<uint64_t> eltSize =
            elementType.GetByteSize(exeCtxScope)) {
      elementSize = eltSize.get();
      // This means that we were able to parse everything we needed, so this is
      // where we set the size.
      size = parsed->second;
      return lldb::ChildCacheState::eRefetch;
    }
  }
  return lldb::ChildCacheState::eRefetch;
}

std::optional<std::pair<ValueObjectSP, size_t>>
MojoListSyntheticFrontEnd::parseList(lldb::ValueObjectSP valobj) {
  if (!valobj || !valobj->GetError().Success())
    return {};

  valobj = valobj->GetNonSyntheticValue();
  if (!valobj || !valobj->GetError().Success())
    return {};

  ValueObjectSP sizeField = valobj->GetChildMemberWithName("_len");
  if (!sizeField || !sizeField->GetError().Success())
    return {};

  // The REPL sees a struct around an int, but DWARF shows directly the int.
  lldb::ValueObjectSP sizeVal = unwrapToScalarOrPointer(sizeField);
  if (!sizeVal || !sizeVal->GetError().Success())
    return {};

  bool success = true;
  size_t size = sizeVal->GetValueAsUnsigned(0, &success);
  if (!success)
    return {};

  ValueObjectSP dataVal = valobj->GetChildMemberWithName("_data");
  if (!dataVal || !dataVal->GetError().Success())
    return {};

  // The REPL sees a struct around a pointer, but DWARF shows directly the
  // pointer. Unwrap through the known UnsafePointer wrapper field names
  // (`_mlir_value`, `address`, etc.) rather than hardcoding one.
  ValueObjectSP dataPointer = unwrapToScalarOrPointer(dataVal);

  if (!dataPointer || !dataPointer->GetError().Success())
    return {};

  // If the size is 0, the data address might be invalid.
  if (size == 0)
    return std::make_pair(dataPointer, size);

  lldb::addr_t data = dataPointer->GetPointerValue().address;
  if (!data || data == LLDB_INVALID_ADDRESS)
    return {};

  return std::make_pair(dataPointer, size);
}

bool MojoListSyntheticFrontEnd::MightHaveChildren() { return true; }

llvm::Expected<size_t> MojoListSyntheticFrontEnd::GetIndexOfChildWithName(
    lldb_private::ConstString name) {
  if (size == 0)
    return llvm::createStringError("List is empty");

  const char *nameStr = name.GetCString();
  if (!nameStr)
    return llvm::createStringError("Invalid name");

  std::optional<size_t> index = ExtractIndexFromString(nameStr);
  if (!index)
    return llvm::createStringError("Invalid index format");

  size_t idx = *index;
  if (idx >= size)
    return llvm::createStringError("Index out of bounds");

  return idx;
}

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::mojoListSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                                const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  CompilerType type = valobjSP->GetCompilerType();
  if (!type.IsValid())
    return nullptr;
  return new M::KGEN::Mojo::MojoListSyntheticFrontEnd(valobjSP);
}
