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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGE_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGE_H

#include "lldb/Target/Language.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"
#include "llvm/ADT/StringRef.h"
#include <set>
#include <vector>

namespace M::KGEN::Mojo {
class MojoLanguage : public lldb_private::Language {

public:
  MojoLanguage() = default;

  ~MojoLanguage() override = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeMojo;
  }

  std::unique_ptr<TypeScavenger> GetTypeScavenger() override { return nullptr; }
  lldb::TypeCategoryImplSP GetFormatters() override;

  lldb_private::HardcodedFormatters::HardcodedSummaryFinder
  GetHardcodedSummaries() override;

  lldb_private::HardcodedFormatters::HardcodedSyntheticFinder
  GetHardcodedSynthetics() override;

  bool IsNilReference(lldb_private::ValueObject &valobj) override {
    if (!(valobj.GetObjectRuntimeLanguage() == lldb::eLanguageTypeMojo) ||
        !valobj.IsPointerType()) {
      return false;
    }
    bool canReadValue = true;
    bool isZero = valobj.GetValueAsUnsigned(0, &canReadValue) == 0;
    return canReadValue && isZero;
  }

  bool IsSourceFile(llvm::StringRef filePath) const override {
    return filePath.ends_with(".mojo") || filePath.ends_with("🔥");
  }

  //===--------------------------------------------------------------------===//
  // Static Functions
  //===--------------------------------------------------------------------===//

  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() { return "mojo"; }

  bool SymbolNameFitsToLanguage(
      const lldb_private::Mangled &mangled) const override {
    return false;
  }

  bool DemangledNameContainsPath(
      llvm::StringRef path,
      lldb_private::ConstString demangled) const override {
    return demangled.GetStringRef().contains(path);
  }

  lldb_private::ConstString GetDemangledFunctionNameWithoutArguments(
      lldb_private::Mangled mangled) const override {
    return {};
  }

  bool GetFunctionDisplayName(const lldb_private::SymbolContext &sc,
                              const lldb_private::ExecutionContext *exe_ctx,
                              FunctionNameRepresentation representation,
                              lldb_private::Stream &s) override {
    return false;
  }

  std::vector<lldb_private::ConstString> GenerateAlternateFunctionManglings(
      const lldb_private::ConstString mangled) const override {
    return {};
  }

  lldb_private::ConstString FindBestAlternateFunctionMangledName(
      const lldb_private::Mangled mangled,
      const lldb_private::SymbolContext &sym_ctx) const override {
    return {};
  }
  llvm::StringRef GetInstanceName() override { return "self"; }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  bool SupportsExceptionBreakpointsOnThrow() const override { return true; }

  llvm::StringRef GetThrowKeyword() const override { return "raise"; }
};

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGE_H
