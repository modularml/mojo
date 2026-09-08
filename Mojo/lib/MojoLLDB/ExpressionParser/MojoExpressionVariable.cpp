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

#include "MojoExpressionVariable.h"
#include "JITExecutionUnit.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "mlir/IR/Types.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

using JittedEntity = JITExecutionUnit::JittedEntity;

//===----------------------------------------------------------------------===//
// MojoExpressionVariable
//===----------------------------------------------------------------------===//

char MojoExpressionVariable::ID;

MojoExpressionVariable::MojoExpressionVariable(ExecutionContextScope *exeScope,
                                               lldb::ByteOrder byteOrder,
                                               uint32_t addrByteSize) {
  m_frozen_sp =
      ValueObjectConstResult::Create(exeScope, byteOrder, addrByteSize);
}

MojoExpressionVariable::MojoExpressionVariable(
    const lldb::ValueObjectSP &valobj) {
  m_frozen_sp = valobj;
}

MojoExpressionVariable::MojoExpressionVariable(ExecutionContextScope *exeScope,
                                               ConstString name,
                                               const TypeFromUser &type,
                                               lldb::ByteOrder byteOrder,
                                               uint32_t addrByteSize) {
  m_frozen_sp =
      ValueObjectConstResult::Create(exeScope, byteOrder, addrByteSize);
  SetName(name);
  SetCompilerType(type);
}

//===----------------------------------------------------------------------===//
// MojoPersistentExpressionState
//===----------------------------------------------------------------------===//

char MojoPersistentExpressionState::ID;

//===----------------------------------------------------------------------===//
// Expression Instance

/// Walk the external JIT symbols within the given execution unit, invoking the
/// provided callback for each.
static void walkExternalJITSymbols(
    JITExecutionUnit &executionUnit, Log *log,
    function_ref<void(const JITExecutionUnit::JittedEntity &)> callback) {
  LLDB_LOGF(log, "Processing JITted Functions:\n");
  for (const auto &jitSym : executionUnit.getJittedFunctions()) {
    if (jitSym.external && jitSym.name != executionUnit.getFunctionName() &&
        jitSym.remoteAddr != LLDB_INVALID_ADDRESS) {
      LLDB_LOGF(log, "  Function: %s at 0x%" PRIx64 ".",
                jitSym.name.GetCString(), jitSym.remoteAddr);
      callback(jitSym);
    }
  }

  LLDB_LOGF(log, "Processing JIIted Symbols:\n");
  for (const auto &jitSym : executionUnit.getJittedGlobalVariables()) {
    if (jitSym.remoteAddr != LLDB_INVALID_ADDRESS) {
      LLDB_LOGF(log, "  Symbol: %s at 0x%" PRIx64 ".", jitSym.name.GetCString(),
                jitSym.remoteAddr);
      callback(jitSym);
    }
  }
}

void MojoPersistentExpressionState::registerExpressionInstance(
    std::shared_ptr<JITExecutionUnit> executionUnit,
    std::vector<lldb::ExpressionVariableSP> &&variables,
    std::optional<std::string> pythonModuleName) {
  Log *log = GetLog(LLDBLog::Expressions);

  // Register the JIT symbols within the execution unit.
  if (executionUnit) {
    auto walkFn = [&](const JittedEntity &jitSym) {
      symbolMap[jitSym.name.GetStringRef()] = jitSym.remoteAddr;
    };
    walkExternalJITSymbols(*executionUnit, log, walkFn);
  }

  // Push a new expression state.
  expressionInstances.emplace_back(std::make_unique<ExpressionInstanceState>(
      std::move(executionUnit), std::move(variables),
      std::move(pythonModuleName)));
}

std::pair<size_t, std::string>
MojoPersistentExpressionState::getNextExpressionModuleName() {
  size_t nextID = nextExpressionModuleID++;
  return {nextID, ("Expression [" + Twine(nextID) + "]").str()};
}

bool MojoPersistentExpressionState::isExpressionModuleName(
    StringRef moduleName) {
  return moduleName.contains("Expression [") && moduleName.ends_with("]");
}

//===----------------------------------------------------------------------===//
// Python Expression State

bool MojoPersistentExpressionState::hasInitializedPython() const {
  return llvm::any_of(expressionInstances, [](const auto &exprInst) {
    return exprInst->pythonModuleName.has_value();
  });
}

std::string MojoPersistentExpressionState::getNextPythonExpressionModuleName() {
  return (getPythonExpressionModuleNamePrefix() +
          std::to_string(nextPythonModuleID++))
      .str();
}

//===----------------------------------------------------------------------===//
// PersistentExpressionState

lldb::ExpressionVariableSP
MojoPersistentExpressionState::CreatePersistentVariable(
    const lldb::ValueObjectSP &valobj) {
  return AddNewlyConstructedVariable(new MojoExpressionVariable(valobj));
}

lldb::ExpressionVariableSP
MojoPersistentExpressionState::CreatePersistentVariable(
    ExecutionContextScope *exeScope, ConstString name,
    const CompilerType &compilerType, lldb::ByteOrder byteOrder,
    uint32_t addrByteSize) {
  return AddNewlyConstructedVariable(new MojoExpressionVariable(
      exeScope, name, compilerType, byteOrder, addrByteSize));
}

lldb::addr_t MojoPersistentExpressionState::LookupSymbol(ConstString name) {
  auto si = symbolMap.find(name.GetStringRef());
  if (si != symbolMap.end())
    return si->second;
  return PersistentExpressionState::LookupSymbol(name);
}

void MojoPersistentExpressionState::collectPersistentVariables(
    SmallVectorImpl<std::pair<StringRef, Type>> &variables) {
  DenseSet<ConstString> persistentVariableNames;
  for (int i : llvm::reverse(llvm::seq<int>(0, GetSize()))) {
    lldb::ExpressionVariableSP var = GetVariableAtIndex(i);
    assert(var && "expected valid variable in persistent state");
    if (!persistentVariableNames.insert(var->GetName()).second)
      continue;

    // All persistent variable types are wrapped in a reference type, so unwrap
    // the types before adding them to the current expression.
    auto ptrType = cast<LIT::REPLResultRefType>(
        Type::getFromOpaquePointer(var->GetCompilerType().GetOpaqueQualType()));

    Type varType = ptrType.getElementType();
    variables.emplace_back(var->GetName().GetStringRef(), varType);
  }
}
