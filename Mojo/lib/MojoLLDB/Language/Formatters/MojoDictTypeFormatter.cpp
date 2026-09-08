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

#include "MojoDictTypeFormatter.h"
#include "../../Utils/ValueObjectHelpers.h"
#include "MojoListTypeFormatter.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

MojoDictSyntheticFrontEnd::MojoDictSyntheticFrontEnd(
    const lldb::ValueObjectSP &backend)
    : SyntheticChildrenFrontEnd(*backend) {
  if (backend)
    Update();
}

llvm::Expected<uint32_t> MojoDictSyntheticFrontEnd::CalculateNumChildren() {
  return m_liveSlots.size();
}

lldb::ValueObjectSP MojoDictSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= m_liveSlots.size() || !m_entryType.IsValid() || m_entrySize == 0)
    return ValueObjectSP();
  uint64_t addr = m_slotsAddr + (uint64_t(m_liveSlots[idx]) * m_entrySize);
  return CreateChildValueObjectFromAddress(
      llvm::formatv("[{0}]", idx).str(), addr,
      m_backend.GetExecutionContextRef(), m_entryType);
}

lldb::ChildCacheState MojoDictSyntheticFrontEnd::Update() {
  m_liveSlots.clear();
  m_slotsAddr = 0;
  m_entryType = CompilerType();
  m_entrySize = 0;

  ExecutionContext execCtx(m_backend.GetExecutionContextRef());
  auto exeCtxScope = execCtx.GetBestExecutionContextScope();

  ValueObjectSP valobj = m_backend.GetSP()->GetNonSyntheticValue();
  if (!valobj || !valobj->GetError().Success())
    return lldb::ChildCacheState::eRefetch;

  // Dict layout: { _table: SwissTable { _ctrl, _slots, _len, _capacity, ... },
  //               _order: List[Int32] }
  // Navigate into _table for the hash-table internals.
  ValueObjectSP tableField = valobj->GetChildMemberWithName("_table");
  if (!tableField || !tableField->GetError().Success())
    return lldb::ChildCacheState::eRefetch;

  // _len: number of live entries
  ValueObjectSP lenField =
      unwrapToScalarOrPointer(tableField->GetChildMemberWithName("_len"));
  if (!lenField || !lenField->GetError().Success())
    return lldb::ChildCacheState::eRefetch;
  bool ok = false;
  size_t len = lenField->GetValueAsUnsigned(0, &ok);
  if (!ok || len == 0)
    return lldb::ChildCacheState::eRefetch;

  // _capacity: total slot count, used to bound slot indices read from _order
  ValueObjectSP capField =
      unwrapToScalarOrPointer(tableField->GetChildMemberWithName("_capacity"));
  if (!capField || !capField->GetError().Success())
    return lldb::ChildCacheState::eRefetch;
  bool capOk = false;
  uint64_t capacity = capField->GetValueAsUnsigned(0, &capOk);
  if (!capOk || capacity == 0)
    return lldb::ChildCacheState::eRefetch;

  // _ctrl: pointer to control-byte array
  ValueObjectSP ctrlField = tableField->GetChildMemberWithName("_ctrl");
  if (!ctrlField || !ctrlField->GetError().Success())
    return lldb::ChildCacheState::eRefetch;
  ValueObjectSP ctrlPtr = unwrapToScalarOrPointer(ctrlField);
  if (!ctrlPtr || !ctrlPtr->IsPointerType())
    return lldb::ChildCacheState::eRefetch;
  lldb::addr_t ctrlAddr = ctrlPtr->GetPointerValue().address;
  if (!ctrlAddr || ctrlAddr == LLDB_INVALID_ADDRESS)
    return lldb::ChildCacheState::eRefetch;

  // _slots: pointer to DictEntry[K,V,H] array
  ValueObjectSP slotsField = tableField->GetChildMemberWithName("_slots");
  if (!slotsField || !slotsField->GetError().Success())
    return lldb::ChildCacheState::eRefetch;
  ValueObjectSP slotsPtr = unwrapToScalarOrPointer(slotsField);
  if (!slotsPtr || !slotsPtr->IsPointerType())
    return lldb::ChildCacheState::eRefetch;
  m_slotsAddr = slotsPtr->GetPointerValue().address;
  if (!m_slotsAddr || m_slotsAddr == LLDB_INVALID_ADDRESS)
    return lldb::ChildCacheState::eRefetch;

  m_entryType = slotsPtr->GetCompilerType().GetPointeeType();
  if (!m_entryType.IsValid())
    return lldb::ChildCacheState::eRefetch;

  auto entrySizeOrErr = m_entryType.GetByteSize(exeCtxScope);
  if (!entrySizeOrErr || *entrySizeOrErr == 0)
    return lldb::ChildCacheState::eRefetch;
  m_entrySize = *entrySizeOrErr;

  // _order: List[Int32] of slot indices in insertion order
  ValueObjectSP orderField = valobj->GetChildMemberWithName("_order");
  if (!orderField || !orderField->GetError().Success())
    return lldb::ChildCacheState::eRefetch;

  auto parsedOrder = MojoListSyntheticFrontEnd::parseList(orderField);
  if (!parsedOrder || parsedOrder->second == 0)
    return lldb::ChildCacheState::eRefetch;
  auto [orderData, orderLen] = *parsedOrder;
  lldb::addr_t orderAddr = orderData->GetPointerValue().address;
  if (!orderAddr || orderAddr == LLDB_INVALID_ADDRESS)
    return lldb::ChildCacheState::eRefetch;

  // Int32 = SIMD[DType.int32, 1] is always 4 bytes in memory.
  uint64_t orderEltStride = 4;
  CompilerType int32EltType = orderData->GetCompilerType().GetPointeeType();
  if (int32EltType.IsValid()) {
    if (auto sz = int32EltType.GetByteSize(exeCtxScope))
      orderEltStride = *sz;
  }

  // Build live-slot index array by walking _order and checking ctrl bytes.
  // A slot is occupied when its control byte is < 0x80 (h2 fingerprint).
  // 0xFF = EMPTY, 0x80 = DELETED, 0x00-0x7F = occupied.
  lldb::ProcessSP process = execCtx.GetProcessSP();
  if (!process)
    return lldb::ChildCacheState::eRefetch;

  m_liveSlots.reserve(len);
  for (size_t i = 0; i < orderLen && m_liveSlots.size() < len; ++i) {
    Status err;
    uint64_t rawSlot = process->ReadUnsignedIntegerFromMemory(
        orderAddr + i * orderEltStride, /*byte_size=*/4,
        /*fail_value=*/UINT64_MAX, err);
    if (err.Fail() || rawSlot == UINT64_MAX)
      continue;

    // _order stores Int32 (signed). Negative or out-of-range values indicate
    // corruption; skip them to avoid bad ctrl/slots accesses.
    int32_t slotIndex = static_cast<int32_t>(static_cast<uint32_t>(rawSlot));
    if (slotIndex < 0 || static_cast<uint64_t>(slotIndex) >= capacity)
      continue;

    uint64_t ctrlByte = process->ReadUnsignedIntegerFromMemory(
        ctrlAddr + slotIndex, /*byte_size=*/1, /*fail_value=*/0xFF, err);
    if (err.Fail())
      continue;

    if (ctrlByte < 0x80)
      m_liveSlots.push_back(static_cast<uint32_t>(slotIndex));
  }

  // If memory reads partially failed, m_liveSlots.size() may be less than len.
  // Partial display is intentional — the summary and child count stay
  // self-consistent because both use m_liveSlots.size().

  return lldb::ChildCacheState::eRefetch;
}

bool MojoDictSyntheticFrontEnd::MightHaveChildren() { return true; }

llvm::Expected<size_t> MojoDictSyntheticFrontEnd::GetIndexOfChildWithName(
    lldb_private::ConstString name) {
  if (m_liveSlots.empty())
    return llvm::createStringError("no synthetic children");
  const char *nameStr = name.GetCString();
  if (!nameStr)
    return llvm::createStringError("Invalid name");
  std::optional<size_t> index = ExtractIndexFromString(nameStr);
  if (!index)
    return llvm::createStringError("Invalid index format");
  size_t idx = *index;
  if (idx >= m_liveSlots.size())
    return llvm::createStringError("Index out of bounds");
  return idx;
}

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::mojoDictSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                                const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  if (!valobjSP->GetCompilerType().IsValid())
    return nullptr;
  return new MojoDictSyntheticFrontEnd(valobjSP);
}
