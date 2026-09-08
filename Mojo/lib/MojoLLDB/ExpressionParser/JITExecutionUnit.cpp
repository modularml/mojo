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

#include "JITExecutionUnit.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Expression/ObjectFileJIT.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static const unsigned eSectionIDInvalid = -1;

enum class AllocationKind { Stub, Code, Data, Global, Bytes };

static lldb::SectionType
getSectionTypeFromSectionName(const StringRef &name, AllocationKind allocKind) {
  lldb::SectionType sectType = lldb::eSectionTypeCode;
  switch (allocKind) {
  case AllocationKind::Stub:
  case AllocationKind::Code:
    sectType = lldb::eSectionTypeCode;
    break;
  case AllocationKind::Data:
  case AllocationKind::Global:
    sectType = lldb::eSectionTypeData;
    break;
  case AllocationKind::Bytes:
    sectType = lldb::eSectionTypeOther;
    break;
  }

  if (!name.empty()) {
    if (name == "__text" || name == ".text" || name == "__data" ||
        name == ".data") {
      sectType = lldb::eSectionTypeCode;
    } else if (name.starts_with("__debug_") || name.starts_with(".debug_")) {
      const uint32_t nameIdx = name[0] == '_' ? 8 : 7;
      StringRef dwarfName(name.substr(nameIdx));
      switch (dwarfName[0]) {
      case 'a':
        if (dwarfName == "abbrev")
          sectType = lldb::eSectionTypeDWARFDebugAbbrev;
        else if (dwarfName == "aranges")
          sectType = lldb::eSectionTypeDWARFDebugAranges;
        else if (dwarfName == "addr")
          sectType = lldb::eSectionTypeDWARFDebugAddr;
        break;

      case 'f':
        if (dwarfName == "frame")
          sectType = lldb::eSectionTypeDWARFDebugFrame;
        break;

      case 'i':
        if (dwarfName == "info")
          sectType = lldb::eSectionTypeDWARFDebugInfo;
        break;

      case 'l':
        if (dwarfName == "line")
          sectType = lldb::eSectionTypeDWARFDebugLine;
        else if (dwarfName == "loc")
          sectType = lldb::eSectionTypeDWARFDebugLoc;
        else if (dwarfName == "loclists")
          sectType = lldb::eSectionTypeDWARFDebugLocLists;
        break;

      case 'm':
        if (dwarfName == "macinfo")
          sectType = lldb::eSectionTypeDWARFDebugMacInfo;
        break;

      case 'p':
        if (dwarfName == "pubnames")
          sectType = lldb::eSectionTypeDWARFDebugPubNames;
        else if (dwarfName == "pubtypes")
          sectType = lldb::eSectionTypeDWARFDebugPubTypes;
        break;

      case 's':
        if (dwarfName == "str")
          sectType = lldb::eSectionTypeDWARFDebugStr;
        else if (dwarfName == "str_offsets")
          sectType = lldb::eSectionTypeDWARFDebugStrOffsets;
        break;

      case 'r':
        if (dwarfName == "ranges")
          sectType = lldb::eSectionTypeDWARFDebugRanges;
        break;

      default:
        break;
      }
    } else if (name.starts_with("__apple_") || name.starts_with(".apple_")) {
      sectType = lldb::eSectionTypeInvalid;
    } else if (name == "__objc_imageinfo") {
      sectType = lldb::eSectionTypeOther;
    }
  }
  return sectType;
}

//===----------------------------------------------------------------------===//
// JITExecutionUnit::Impl
//===----------------------------------------------------------------------===//

struct JITExecutionUnit::Impl {
  Impl(SymbolTable symbolTable, ExportMap exportedSymbols,
       OwningBinary<llvm::object::Binary> newObject, ConstString &name,
       const SymbolContext &symCtx, std::vector<std::string> &cpuFeatures)
      : context(std::make_unique<llvm::LLVMContext>()),
        symbolTable(std::move(symbolTable)),
        exportedSymbols(std::move(exportedSymbols)), cpuFeatures(cpuFeatures),
        name(name), symCtx(symCtx) {
    std::tie(object, objectBuffer) = newObject.takeBinary();
  }

  std::vector<AllocationRecord> records;

  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::ExecutionEngine> executionEngine;
  std::unique_ptr<llvm::ObjectCache> objectCache;
  SymbolTable symbolTable;
  ExportMap exportedSymbols;
  std::unique_ptr<llvm::object::Binary> object;
  BufferRef objectBuffer;

  std::vector<std::string> cpuFeatures;

  /// The jitted functions and global variables.
  std::vector<JittedFunction> jittedFunctions;
  std::vector<JittedGlobalVariable> jittedGlobalVariables;

  const lldb_private::ConstString name;
  lldb_private::SymbolContext symCtx;
  std::vector<lldb_private::ConstString> failedLookups;

  std::atomic<bool> didJit{false};

  lldb::addr_t functionLoadAddr = LLDB_INVALID_ADDRESS;
  lldb::addr_t functionEndLoadAddr = LLDB_INVALID_ADDRESS;

  /// True for platforms where global symbols have a _ prefix.
  bool usesGlobalUnderscorePrefix = true;

  /// True after allocations have been reported. It is possible that sections
  /// will be allocated when this is true, in which case they weren't depended
  /// on by any function. (Top-level code defining a variable, but defining no
  /// functions using that variable, would do this.) If this is true, any
  /// allocations need to be committed immediately -- no opportunity for
  /// relocation.
  bool reportedAllocations = false;
};

//===----------------------------------------------------------------------===//
// JITExecutionUnit
//===----------------------------------------------------------------------===//

JITExecutionUnit::JITExecutionUnit(SymbolTable symbolTable,
                                   ExportMap exportedSymbols,
                                   OwningBinary<llvm::object::Binary> object,
                                   ConstString &name,
                                   const lldb::TargetSP &target,
                                   const SymbolContext &symCtx,
                                   std::vector<std::string> &cpuFeatures)
    : IRMemoryMap(target),
      impl(std::make_unique<Impl>(std::move(symbolTable),
                                  std::move(exportedSymbols), std::move(object),
                                  name, symCtx, cpuFeatures)) {}
JITExecutionUnit::~JITExecutionUnit() = default;

lldb_private::ConstString JITExecutionUnit::getFunctionName() {
  return impl->name;
}

//===----------------------------------------------------------------------===//
// JITExecutionUnit::MemoryManager
//===----------------------------------------------------------------------===//

class JITExecutionUnit::MemoryManager : public llvm::SectionMemoryManager {
public:
  MemoryManager(JITExecutionUnit &parent);
  ~MemoryManager() override;

  //===--------------------------------------------------------------------===//
  // Allocation
  //===--------------------------------------------------------------------===//

  /// Allocate space for executable code, and add it to the space blocks map.
  uint8_t *allocateCodeSection(uintptr_t size, unsigned alignment,
                               unsigned sectionID,
                               StringRef sectionName) override;

  /// Allocate space for data, and add it to the space blocks map.
  uint8_t *allocateDataSection(uintptr_t size, unsigned alignment,
                               unsigned sectionID, StringRef sectionName,
                               bool isReadOnly) override;

  /// Called when object loading is complete and section page permissions
  /// can be applied. Currently unimplemented for LLDB.
  bool finalizeMemory(std::string *errMsg) override { return false; }

  /// Ignore any EHFrame registration.
  void registerEHFrames(uint8_t *addr, uint64_t loadAddr,
                        size_t size) override {}
  void deregisterEHFrames() override {}

  //===--------------------------------------------------------------------===//
  // Symbol Resolution
  //===--------------------------------------------------------------------===//

  /// Find the address of the symbol Name. If Name is a missing weak symbol then
  /// missingWeak will be true.
  uint64_t getSymbolAddressAndPresence(const std::string &name,
                                       bool &missingWeak);

  uint64_t getSymbolAddress(const std::string &name) override;

  llvm::JITSymbol findSymbol(const std::string &name) override;

  void *getPointerToNamedFunction(const std::string &name,
                                  bool abortOnFailure = true) override;

private:
  /// The memory allocator to use in actually creating space. All calls are
  /// passed through to it.
  std::unique_ptr<SectionMemoryManager> defaultMM;
  /// The execution unit this is a proxy for.
  JITExecutionUnit &parent;
};

JITExecutionUnit::MemoryManager::MemoryManager(JITExecutionUnit &parent)
    : defaultMM(new llvm::SectionMemoryManager()), parent(parent) {}

JITExecutionUnit::MemoryManager::~MemoryManager() = default;

//===----------------------------------------------------------------------===//
// Allocation

uint8_t *JITExecutionUnit::MemoryManager::allocateCodeSection(
    uintptr_t size, unsigned alignment, unsigned sectionID,
    StringRef sectionName) {
  Log *log = GetLog(LLDBLog::Expressions);

  uint8_t *returnValue =
      defaultMM->allocateCodeSection(size, alignment, sectionID, sectionName);
  parent.impl->records.emplace_back(
      (uintptr_t)returnValue,
      lldb::ePermissionsReadable | lldb::ePermissionsExecutable,
      getSectionTypeFromSectionName(sectionName, AllocationKind::Code), size,
      alignment, sectionID, sectionName.str().c_str());

  LLDB_LOGF(log,
            "JITExecutionUnit::allocateCodeSection(Size=0x%" PRIx64
            ", Alignment=%u, SectionID=%u) = %p",
            (uint64_t)size, alignment, sectionID, (void *)returnValue);

  // If we've already reported allocation, commit this one immediately.
  if (parent.impl->reportedAllocations) {
    Status err;
    lldb::ProcessSP process(
        parent.GetBestExecutionContextScope()->CalculateProcess());
    parent.commitOneAllocation(process, err, parent.impl->records.back());
  }
  return returnValue;
}

uint8_t *JITExecutionUnit::MemoryManager::allocateDataSection(
    uintptr_t size, unsigned alignment, unsigned sectionID,
    StringRef sectionName, bool isReadOnly) {
  Log *log = GetLog(LLDBLog::Expressions);

  uint8_t *returnValue = defaultMM->allocateDataSection(
      size, alignment, sectionID, sectionName, isReadOnly);

  uint32_t permissions = lldb::ePermissionsReadable;
  if (!isReadOnly)
    permissions |= lldb::ePermissionsWritable;

  parent.impl->records.emplace_back(
      (uintptr_t)returnValue, permissions,
      getSectionTypeFromSectionName(sectionName, AllocationKind::Data), size,
      alignment, sectionID, sectionName.str().c_str());
  LLDB_LOGF(log,
            "JITExecutionUnit::allocateDataSection(Size=0x%" PRIx64
            ", Alignment=%u, SectionID=%u) = %p",
            (uint64_t)size, alignment, sectionID, (void *)returnValue);

  // If we've already reported allocation, commit this one immediately.
  if (parent.impl->reportedAllocations) {
    Status err;
    lldb::ProcessSP process =
        parent.GetBestExecutionContextScope()->CalculateProcess();
    parent.commitOneAllocation(process, err, parent.impl->records.back());
  }

  return returnValue;
}

//===----------------------------------------------------------------------===//
// Symbol Resolution

uint64_t JITExecutionUnit::MemoryManager::getSymbolAddressAndPresence(
    const std::string &name, bool &missingWeak) {
  Log *log = GetLog(LLDBLog::Expressions);
  const char *namePtr = name.c_str() + parent.impl->usesGlobalUnderscorePrefix;
  ConstString nameCS(namePtr);

  lldb::addr_t ret = parent.findSymbol(nameCS, missingWeak);
  if (ret == LLDB_INVALID_ADDRESS) {
    LLDB_LOGF(log,
              "JITExecutionUnit::getSymbolAddress(Name=\"%s\") = <not found>",
              namePtr);
    parent.reportSymbolLookupError(nameCS);
    return 0;
  }

  LLDB_LOGF(log, "JITExecutionUnit::getSymbolAddress(Name=\"%s\") = %" PRIx64,
            namePtr, ret);
  return ret;
}

uint64_t
JITExecutionUnit::MemoryManager::getSymbolAddress(const std::string &name) {
  bool missingWeak = false;
  return getSymbolAddressAndPresence(name, missingWeak);
}

llvm::JITSymbol
JITExecutionUnit::MemoryManager::findSymbol(const std::string &name) {
  bool missingWeak = false;
  uint64_t addr = getSymbolAddressAndPresence(name, missingWeak);

  auto extraFlags =
      missingWeak ? llvm::JITSymbolFlags::Weak : llvm::JITSymbolFlags::None;
  return llvm::JITSymbol(addr, llvm::JITSymbolFlags::Exported | extraFlags);
}

void *JITExecutionUnit::MemoryManager::getPointerToNamedFunction(
    const std::string &name, bool abortOnFailure) {
  return (void *)getSymbolAddress(name);
}

//===----------------------------------------------------------------------===//
// JIT Symbols
//===----------------------------------------------------------------------===//

/// Check whether the symbol table entry at \p addr has a linkage name that
/// matches \p expectedName. This guards against false DWARF function matches
/// where the .debug_names accelerator table matches a C++ method by its
/// DW_AT_name basename (e.g., "write" matching YAMLVFSWriter::write) and
/// resolves to the wrong address. If the symtab symbol at the address has a
/// different linkage name, the DWARF match is a false positive.
///
/// Returns true (accept) if the symtab confirms the name or if there is no
/// symtab information to contradict it.
///
/// This performs a reverse address lookup, which is not free but acceptable
/// because JIT symbol resolution is not a hot path.
static bool isConfirmedBySymtab(Target &target, ConstString expectedName,
                                lldb::addr_t addr) {
  Address resolved;
  if (!target.ResolveLoadAddress(addr, resolved))
    return true; // Can't resolve — no evidence to reject.

  SymbolContext sc;
  resolved.CalculateSymbolContext(&sc, lldb::eSymbolContextSymbol);
  if (!sc.symbol)
    return true; // No symtab entry at this address — accept.

  // If the symbol's mangled (linkage) name is set and differs from the
  // expected name, this is a false DWARF match.
  ConstString linkageName = sc.symbol->GetMangled().GetMangledName();
  if (linkageName && linkageName != expectedName)
    return false;

  return true;
}

namespace {
class LoadAddressResolver {
public:
  LoadAddressResolver(Target *target, bool &symbolWasMissingWeak)
      : target(target), symbolWasMissingWeak(symbolWasMissingWeak) {}

  std::optional<lldb::addr_t> Resolve(SymbolContextList &scList) {
    if (scList.IsEmpty())
      return std::nullopt;
    lldb::addr_t loadAddr = LLDB_INVALID_ADDRESS;

    // symbolWasMissingWeak will be true only if we found only weak undefined
    // references to this symbol.
    symbolWasMissingWeak = true;

    for (auto candidate : scList.SymbolContexts()) {
      // Only symbols can be weak undefined.
      if (!candidate.symbol ||
          candidate.symbol->GetType() != lldb::eSymbolTypeUndefined ||
          !candidate.symbol->IsWeak())
        symbolWasMissingWeak = false;

      // First try the symbol.
      if (candidate.symbol) {
        loadAddr = candidate.symbol->ResolveCallableAddress(
            *target, candidate.module_sp);
        if (loadAddr == LLDB_INVALID_ADDRESS) {
          Address addr = candidate.symbol->GetAddress();
          loadAddr = target->GetProcessSP() ? addr.GetLoadAddress(target)
                                            : addr.GetFileAddress();
        }
      }

      // If that didn't work, try the function.
      if (loadAddr == LLDB_INVALID_ADDRESS && candidate.function) {
        Address addr = candidate.function->GetAddress();
        loadAddr = target->GetProcessSP() ? addr.GetLoadAddress(target)
                                          : addr.GetFileAddress();
      }

      // We found a load address.
      if (loadAddr != LLDB_INVALID_ADDRESS) {
        // If the load address is external, we're done.
        const bool isExternal =
            (candidate.function) ||
            (candidate.symbol && candidate.symbol->IsExternal());
        if (isExternal)
          return loadAddr;

        // Otherwise, remember the best internal load address.
        if (bestInternalLoadAddr == LLDB_INVALID_ADDRESS)
          bestInternalLoadAddr = loadAddr;
      }
    }

    // You test the address of a weak symbol against NULL to see if it is
    // present. So we should return 0 for a missing weak symbol.
    if (symbolWasMissingWeak)
      return 0;

    return std::nullopt;
  }

  lldb::addr_t GetBestInternalLoadAddress() const {
    return bestInternalLoadAddr;
  }

  void SetBestInternalLoadAddress(lldb::addr_t addr) {
    bestInternalLoadAddr = addr;
  }

private:
  Target *target;
  bool &symbolWasMissingWeak;
  lldb::addr_t bestInternalLoadAddr = LLDB_INVALID_ADDRESS;
};
} // namespace

auto JITExecutionUnit::getJittedFunctions() -> ArrayRef<JittedFunction> {
  return impl->jittedFunctions;
}

auto JITExecutionUnit::getJittedGlobalVariables()
    -> ArrayRef<JittedGlobalVariable> {
  return impl->jittedGlobalVariables;
}

lldb::addr_t JITExecutionUnit::findSymbol(lldb_private::ConstString name,
                                          bool &missingWeak) {
  std::vector<ConstString> names(1, name);

  lldb::addr_t ret = findInSymbols(names, impl->symCtx, missingWeak);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;

  // If we find the symbol in runtimes or user defined symbols it can't be a
  // missing weak symbol.
  missingWeak = false;
  ret = findInRuntimes(names, impl->symCtx);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;
  return findInUserDefinedSymbols(names, impl->symCtx);
}

lldb::addr_t
JITExecutionUnit::findInSymbols(const std::vector<ConstString> &names,
                                const lldb_private::SymbolContext &sc,
                                bool &symbolWasMissingWeak) {
  symbolWasMissingWeak = false;

  Target *target = sc.target_sp.get();
  if (!target) {
    // We shouldn't be doing any symbol lookup at all without a target.
    return LLDB_INVALID_ADDRESS;
  }

  LoadAddressResolver resolver(target, symbolWasMissingWeak);
  ModuleFunctionSearchOptions functionOptions;
  functionOptions.include_symbols = true;
  functionOptions.include_inlines = false;

  for (const ConstString &name : names) {
    // Search order follows upstream IRExecutionUnit: DWARF functions first
    // (module, then all images), then symbol table.
    //
    // Differs from upstream: DWARF function matches are cross-referenced
    // against the symbol table via isConfirmedBySymtab(). LLDB's .debug_names
    // accelerator table indexes C++ methods by their DW_AT_name (basename),
    // so FindFunctions("write", eFunctionNameTypeFull) can incorrectly match
    // a C++ method like llvm::vfs::YAMLVFSWriter::write and resolve to the
    // wrong address. If the symtab symbol at the resolved address has a
    // different linkage name, the match is rejected and we keep searching.
    //
    if (sc.module_sp) {
      SymbolContextList scList;
      lldb::addr_t prevBest = resolver.GetBestInternalLoadAddress();
      sc.module_sp->FindFunctions(name, CompilerDeclContext(),
                                  lldb::eFunctionNameTypeFull, functionOptions,
                                  scList);
      if (auto loadAddr = resolver.Resolve(scList)) {
        if (isConfirmedBySymtab(*target, name, *loadAddr))
          return *loadAddr;
        // False DWARF match — roll back any internal address that Resolve()
        // may have recorded from this batch.
        resolver.SetBestInternalLoadAddress(prevBest);
      }
    }

    if (sc.target_sp) {
      SymbolContextList scList;
      lldb::addr_t prevBest = resolver.GetBestInternalLoadAddress();
      sc.target_sp->GetImages().FindFunctions(name, lldb::eFunctionNameTypeFull,
                                              functionOptions, scList);
      if (auto loadAddr = resolver.Resolve(scList)) {
        if (isConfirmedBySymtab(*target, name, *loadAddr))
          return *loadAddr;
        resolver.SetBestInternalLoadAddress(prevBest);
      }
    }

    if (sc.target_sp) {
      SymbolContextList scList;
      sc.target_sp->GetImages().FindSymbolsWithNameAndType(
          name, lldb::eSymbolTypeAny, scList);
      if (auto loadAddr = resolver.Resolve(scList))
        return *loadAddr;
    }

    lldb::addr_t bestInternalLoadAddr = resolver.GetBestInternalLoadAddress();
    if (bestInternalLoadAddr != LLDB_INVALID_ADDRESS)
      return bestInternalLoadAddr;
  }

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
JITExecutionUnit::findInRuntimes(const std::vector<ConstString> &names,
                                 const lldb_private::SymbolContext &sc) {
  lldb::ProcessSP process =
      sc.target_sp ? sc.target_sp->GetProcessSP() : nullptr;
  if (!process)
    return LLDB_INVALID_ADDRESS;

  for (const ConstString &name : names) {
    for (LanguageRuntime *runtime : process->GetLanguageRuntimes()) {
      lldb::addr_t symbolLoadAddr = runtime->LookupRuntimeSymbol(name);
      if (symbolLoadAddr != LLDB_INVALID_ADDRESS)
        return symbolLoadAddr;
    }
  }
  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t JITExecutionUnit::findInUserDefinedSymbols(
    const std::vector<ConstString> &names,
    const lldb_private::SymbolContext &sc) {
  lldb::TargetSP target = sc.target_sp;
  for (const ConstString &name : names) {
    lldb::addr_t symbolLoadAddr = target->GetPersistentSymbol(name);
    if (symbolLoadAddr != LLDB_INVALID_ADDRESS)
      return symbolLoadAddr;
  }
  return LLDB_INVALID_ADDRESS;
}

void JITExecutionUnit::reportSymbolLookupError(ConstString name) {
  impl->failedLookups.push_back(name);
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

lldb::addr_t
JITExecutionUnit::getRemoteAddressForLocal(lldb::addr_t localAddress) {
  Log *log = GetLog(LLDBLog::Expressions);

  for (AllocationRecord &record : impl->records) {
    if (localAddress >= record.hostAddress &&
        localAddress < record.hostAddress + record.size) {
      if (record.processAddress == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

      lldb::addr_t ret =
          record.processAddress + (localAddress - record.hostAddress);

      LLDB_LOGF(log,
                "JITExecutionUnit::GetRemoteAddressForLocal() found 0x%" PRIx64
                " in [0x%" PRIx64 "..0x%" PRIx64 "], and returned 0x%" PRIx64
                " from [0x%" PRIx64 "..0x%" PRIx64 "].",
                localAddress, (uint64_t)record.hostAddress,
                (uint64_t)record.hostAddress + (uint64_t)record.size, ret,
                record.processAddress, record.processAddress + record.size);

      return ret;
    }
  }

  return LLDB_INVALID_ADDRESS;
}

JITExecutionUnit::AddrRange
JITExecutionUnit::getRemoteRangeForLocal(lldb::addr_t localAddress) {
  for (AllocationRecord &record : impl->records) {
    if (localAddress >= record.hostAddress &&
        localAddress < record.hostAddress + record.size) {
      if (record.processAddress == LLDB_INVALID_ADDRESS)
        return AddrRange(0, 0);
      return AddrRange(record.processAddress, record.size);
    }
  }

  return AddrRange(0, 0);
}

bool JITExecutionUnit::commitOneAllocation(lldb::ProcessSP &process,
                                           Status &error,
                                           AllocationRecord &record) {
  if (record.processAddress != LLDB_INVALID_ADDRESS)
    return true;

  switch (record.sectType) {
  case lldb::eSectionTypeInvalid:
  case lldb::eSectionTypeDWARFDebugAbbrev:
  case lldb::eSectionTypeDWARFDebugAddr:
  case lldb::eSectionTypeDWARFDebugAranges:
  case lldb::eSectionTypeDWARFDebugCuIndex:
  case lldb::eSectionTypeDWARFDebugFrame:
  case lldb::eSectionTypeDWARFDebugInfo:
  case lldb::eSectionTypeDWARFDebugLine:
  case lldb::eSectionTypeDWARFDebugLoc:
  case lldb::eSectionTypeDWARFDebugLocLists:
  case lldb::eSectionTypeDWARFDebugMacInfo:
  case lldb::eSectionTypeDWARFDebugPubNames:
  case lldb::eSectionTypeDWARFDebugPubTypes:
  case lldb::eSectionTypeDWARFDebugRanges:
  case lldb::eSectionTypeDWARFDebugStr:
  case lldb::eSectionTypeDWARFDebugStrOffsets:
  case lldb::eSectionTypeDWARFAppleNames:
  case lldb::eSectionTypeDWARFAppleTypes:
  case lldb::eSectionTypeDWARFAppleNamespaces:
  case lldb::eSectionTypeDWARFAppleObjC:
  case lldb::eSectionTypeDWARFGNUDebugAltLink:
    error.Clear();
    break;
  default:
    llvm::Expected<lldb::addr_t> processAddressOr =
        Malloc(record.size, record.alignment, record.permissions,
               eAllocationPolicyProcessOnly, /*zero_memory=*/false,
               /*used_policy=*/nullptr);
    if (llvm::Error err = processAddressOr.takeError()) {
      record.processAddress = LLDB_INVALID_ADDRESS;
      error = Status::FromErrorString("Failed to allocate");
    } else {
      record.processAddress = processAddressOr.get();
    }
    break;
  }

  return error.Success();
}

bool JITExecutionUnit::commitAllocations(lldb::ProcessSP &process) {
  lldb_private::Status err;
  if (llvm::all_of(impl->records, [&](auto &record) {
        return commitOneAllocation(process, err, record);
      }))
    return true;

  // If we failed, free any of the allocations we've made so far.
  for (AllocationRecord &record : impl->records) {
    if (record.processAddress != LLDB_INVALID_ADDRESS) {
      Free(record.processAddress, err);
      record.processAddress = LLDB_INVALID_ADDRESS;
    }
  }
  return false;
}

void JITExecutionUnit::reportAllocations(llvm::ExecutionEngine &engine) {
  impl->reportedAllocations = true;

  for (AllocationRecord &record : impl->records) {
    if (record.processAddress == LLDB_INVALID_ADDRESS ||
        record.sectionId == eSectionIDInvalid)
      continue;

    engine.mapSectionAddress((void *)record.hostAddress, record.processAddress);
  }

  // Trigger re-application of relocations.
  engine.finalizeObject();
}

bool JITExecutionUnit::writeData(lldb::ProcessSP &process) {
  bool wroteSomething = false;
  for (AllocationRecord &record : impl->records) {
    if (record.processAddress == LLDB_INVALID_ADDRESS)
      continue;
    lldb_private::Status err;
    WriteMemory(record.processAddress, (uint8_t *)record.hostAddress,
                record.size, err);
    if (err.Success())
      wroteSomething = true;
  }
  return wroteSomething;
}

void JITExecutionUnit::AllocationRecord::dump(Log *log) {
  if (!log)
    return;

  LLDB_LOGF(log,
            "[0x%llx+0x%llx]->0x%llx (alignment %d, section ID %d, name %s)",
            (unsigned long long)hostAddress, (unsigned long long)size,
            (unsigned long long)processAddress, (unsigned)alignment,
            (unsigned)sectionId, name.c_str());
}

//===----------------------------------------------------------------------===//
// Compilation
//===----------------------------------------------------------------------===//

namespace {
struct IRExecDiagnosticHandler : public llvm::DiagnosticHandler {
  Status *err;
  IRExecDiagnosticHandler(Status *err) : err(err) {}
  bool handleDiagnostics(const llvm::DiagnosticInfo &DI) override {
    if (DI.getKind() == llvm::DK_SrcMgr) {
      const auto &dism = llvm::cast<llvm::DiagnosticInfoSrcMgr>(DI);
      if (err && err->Success()) {
        *err = Status::FromErrorStringWithFormatv(
            "Inline assembly error: %s",
            dism.getSMDiag().getMessage().str().c_str());
      }
      return true;
    }

    return false;
  }
};

class ObjectDumper : public llvm::ObjectCache {
public:
  ObjectDumper(FileSpec outputDir) : outDir(outputDir) {}

  void notifyObjectCompiled(const llvm::Module *module,
                            llvm::MemoryBufferRef object) override {
    llvm::SmallVector<char, 256> resultPath;
    FileSpec modelSpec = outDir.CopyByAppendingPathComponent(
        "jit-object-" + module->getModuleIdentifier() + "-%%%.o");

    int fd = 0;
    std::error_code result =
        llvm::sys::fs::createUniqueFile(modelSpec.GetPath(), fd, resultPath);
    if (!result) {
      llvm::raw_fd_ostream fds(fd, true);
      fds.write(object.getBufferStart(), object.getBufferSize());
    }
  }

  std::unique_ptr<llvm::MemoryBuffer>
  getObject(const llvm::Module *module) override {
    // Return nothing - we're just abusing the object-cache mechanism to dump
    // objects.
    return nullptr;
  }

private:
  FileSpec outDir;
};

} // namespace

Status JITExecutionUnit::getRunnableInfo(lldb::addr_t &funcAddr,
                                         lldb::addr_t &funcEnd) {
  lldb::ProcessSP process(GetProcessWP().lock());
  funcAddr = LLDB_INVALID_ADDRESS;
  funcEnd = LLDB_INVALID_ADDRESS;

  Status error;
  if (!process) {
    error =
        Status::FromErrorString("Couldn't write the JIT compiled code into the "
                                "process because the process is invalid");
    return error;
  }

  if (impl->didJit) {
    funcAddr = impl->functionLoadAddr;
    funcEnd = impl->functionEndLoadAddr;
    return Status();
  }
  impl->didJit = true;

  static std::recursive_mutex runnableInfoMutex;
  std::lock_guard<std::recursive_mutex> guard(runnableInfoMutex);

  Log *log = GetLog(LLDBLog::Expressions);
  if (log) {
    std::string s;
    llvm::raw_string_ostream oss(s);
    impl->symbolTable.getOp()->print(oss);
    oss.flush();
    LLDB_LOGF(log, "Symbol table being sent to JIT: \n%s", s.c_str());
  }

  auto ownedModule =
      std::make_unique<llvm::Module>("Dummy JIT LLVM module", *impl->context);
  llvm::Module *module = &*ownedModule;
  llvm::EngineBuilder builder(std::move(ownedModule));

  std::string errorString;
  builder.setEngineKind(llvm::EngineKind::JIT)
      .setErrorStr(&errorString)
      .setRelocationModel(llvm::Reloc::Static)
      .setMCJITMemoryManager(std::make_unique<MemoryManager>(*this))
      .setOptLevel(llvm::CodeGenOptLevel::Less);

  llvm::Triple triple;
  StringRef mArch;
  StringRef mCPU;
  llvm::SmallVector<std::string, 0> mAttrs;

  for (std::string &feature : impl->cpuFeatures)
    mAttrs.push_back(feature);

  llvm::TargetMachine *targetMachine =
      builder.selectTarget(triple, mArch, mCPU, mAttrs);

  impl->executionEngine.reset(builder.create(targetMachine));
  impl->executionEngine->UnregisterJITEventListener(
      llvm::JITEventListener::createGDBRegistrationListener());

  // Declare __dso_local.
  llvm::Type *dsoHandleTy = llvm::Type::getInt64Ty(*impl->context);
  module->getOrInsertGlobal("__dso_handle", dsoHandleTy, [&] {
    auto *gv = new llvm::GlobalVariable(
        *module, dsoHandleTy, /*isConstant=*/true,
        llvm::GlobalVariable::ExternalLinkage,
        llvm::ConstantInt::get(dsoHandleTy, 0), "__dso_handle");
    gv->setVisibility(llvm::GlobalVariable::DefaultVisibility);
    return gv;
  });

  if (!impl->executionEngine) {
    return Status::FromErrorStringWithFormatv("Couldn't JIT the function: %s",
                                              errorString.c_str());
  }

  impl->usesGlobalUnderscorePrefix =
      (impl->executionEngine->getDataLayout().getGlobalPrefix() == '_');

  if (FileSpec saveObjectsDir = process->GetTarget().GetSaveJITObjectsDir()) {
    impl->objectCache = std::make_unique<ObjectDumper>(saveObjectsDir);
    impl->executionEngine->setObjectCache(impl->objectCache.get());
  }

  // Make sure we see all sections, including ones that don't have relocations.
  impl->executionEngine->setProcessAllSections(true);
  impl->executionEngine->DisableLazyCompilation();

  llvm::Error err = llvm::Error::success();
  SmallVector<std::unique_ptr<llvm::object::ObjectFile>> objFiles;

  auto objectFile = dyn_cast<llvm::object::ObjectFile>(impl->object);

  if (!objectFile)
    return Status("not an object file");

  objFiles.push_back(std::move(objectFile));

  // If this is arm elf, we need to add the object files in reverse order. This
  // works around relocation issues related to using MCJIT.
  if (GetArchitecture().GetMachine() >= llvm::Triple::arm &&
      GetArchitecture().GetMachine() <= llvm::Triple::aarch64_32) {
    for (auto &it : llvm::reverse(objFiles))
      impl->executionEngine->addObjectFile(std::move(it));
  } else {
    for (auto &it : objFiles)
      impl->executionEngine->addObjectFile(std::move(it));
  }

  // Handle all errors - we didn't hit anything.
  llvm::handleAllErrors(std::move(err));

  // Register each function in the module.
  for (auto &[sym, exportVal] : impl->exportedSymbols) {
    auto func = impl->symbolTable.lookup<FuncOp>(sym);
    if (!func) {
      // Not a function; skip it.
      continue;
    }
    bool external = func.isExported();

    // Lookup the function by its global name.
    std::string name = sym.getValue().str();
    if (impl->usesGlobalUnderscorePrefix)
      name = impl->executionEngine->getDataLayout().getGlobalPrefix() + name;
    void *fnPtr = impl->executionEngine->getPointerToNamedFunction(name);
    if (!error.Success())
      return error;
    if (!fnPtr) {
      error = Status::FromErrorStringWithFormatv(
          "'%s' was in the parsed module, but wasn't compiled into the "
          "standalone archive",
          name.c_str());
      return error;
    }
    // Add the function's address to the list of JIT'ted functions, using its
    // symbol name. All KGEN functions are marked internal.
    impl->jittedFunctions.emplace_back(sym.getValue().data(), external,
                                       reinterpret_cast<uintptr_t>(fnPtr));
  }

  commitAllocations(process);
  reportAllocations(*impl->executionEngine);

  writeData(process);

  if (!impl->failedLookups.empty()) {
    StreamString ss;
    ss.PutCString("Couldn't lookup symbols:\n");

    bool emitNewLine = false;
    for (ConstString failedLookup : impl->failedLookups) {
      if (emitNewLine)
        ss.PutCString("\n");
      emitNewLine = true;
      ss.PutCString("  ");
      ss.PutCString(Mangled(failedLookup).GetDemangledName().GetStringRef());
    }
    impl->failedLookups.clear();
    error = Status::FromErrorString(ss.GetString().data());
    return error;
  }

  impl->functionLoadAddr = LLDB_INVALID_ADDRESS;
  impl->functionEndLoadAddr = LLDB_INVALID_ADDRESS;

  for (JittedFunction &jittedFn : impl->jittedFunctions) {
    jittedFn.remoteAddr = getRemoteAddressForLocal(jittedFn.localAddr);

    if (!impl->name.IsEmpty() && jittedFn.name == impl->name) {
      AddrRange funcRange = getRemoteRangeForLocal(jittedFn.localAddr);
      impl->functionEndLoadAddr = funcRange.first + funcRange.second;
      impl->functionLoadAddr = jittedFn.remoteAddr;
    }
  }

  if (log) {
    LLDB_LOGF(log, "Code can be run in the target.");
    LLDB_LOGF(log, "Sections: ");
    for (AllocationRecord &record : impl->records) {
      record.dump(log);
      if (record.processAddress != LLDB_INVALID_ADDRESS) {
        Status err;
        DataBufferHeap buffer(record.size, 0);
        ReadMemory(buffer.GetBytes(), record.processAddress, record.size, err);

        if (err.Success()) {
          DataExtractor extractor(buffer.GetBytes(), buffer.GetByteSize(),
                                  lldb::eByteOrderBig, 8);
          extractor.PutToLog(log, 0, buffer.GetByteSize(),
                             record.processAddress, 16,
                             DataExtractor::TypeUInt8);
        }
        continue;
      }

      DataExtractor extractor((const void *)record.hostAddress, record.size,
                              lldb::eByteOrderBig, 8);
      extractor.PutToLog(log, 0, record.size, record.hostAddress, 16,
                         DataExtractor::TypeUInt8);
    }
  }

  funcAddr = impl->functionLoadAddr;
  funcEnd = impl->functionEndLoadAddr;
  return error;
}

//===----------------------------------------------------------------------===//
// ObjectFileJITDelegate
//===----------------------------------------------------------------------===//

lldb::ByteOrder JITExecutionUnit::GetByteOrder() const {
  ExecutionContext exeCtx(GetBestExecutionContextScope());
  return exeCtx.GetByteOrder();
}

uint32_t JITExecutionUnit::GetAddressByteSize() const {
  ExecutionContext exeCtx(GetBestExecutionContextScope());
  return exeCtx.GetAddressByteSize();
}

void JITExecutionUnit::PopulateSectionList(
    lldb_private::ObjectFile *objFile, lldb_private::SectionList &sectionList) {
  for (AllocationRecord &record : impl->records) {
    if (!record.size)
      continue;
    sectionList.AddSection(std::make_shared<lldb_private::Section>(
        objFile->GetModule(), objFile, record.sectionId,
        ConstString(record.name), record.sectType, record.processAddress,
        record.size, record.hostAddress, record.size, 0, record.permissions));
  }
}

ArchSpec JITExecutionUnit::GetArchitecture() {
  ExecutionContext exeCtx(GetBestExecutionContextScope());
  if (Target *target = exeCtx.GetTargetPtr())
    return target->GetArchitecture();
  return ArchSpec();
}
