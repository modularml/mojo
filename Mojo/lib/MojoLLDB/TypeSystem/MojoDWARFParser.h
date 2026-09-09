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

#ifndef KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJODWARFPARSER_H
#define KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJODWARFPARSER_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "lldb/source/Plugins/SymbolFile/DWARF/DWARFASTParser.h"
#include "lldb/source/Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"

namespace lldb_private::plugin::dwarf {
class DWARFDebugInfoEntry;
class SymbolFileDWARF;
} // namespace lldb_private::plugin::dwarf

namespace M {
class MojoASTDeclRef;

namespace KGEN::LIT {
class ASTDecl;
} // namespace KGEN::LIT

} // namespace M

namespace M::KGEN::Mojo {
class MojoTypeSystem;

/// The purpose of the class is to translate DWARF entries into decls and types,
/// trying to reconstruct the original source code to some extent. The result
/// objects must have correct memory layouts for variable printing.
class MojoDWARFParser : public lldb_private::plugin::dwarf::DWARFASTParser {
public:
  MojoDWARFParser(MojoTypeSystem &typeSystem);

  ~MojoDWARFParser() override;

  /// Create a type from the given DW_AT_type die or return a cached one.
  ///
  /// If `typeIsNewPtr` is provided, it should be set to true if a new type is
  /// effectively created.
  lldb::TypeSP
  ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                     const lldb_private::plugin::dwarf::DWARFDIE &die,
                     bool *typeIsNewPtr) override;

  lldb_private::ConstString ConstructDemangledNameFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override {
    // Unimplemented.
    return {};
  }

  /// Create a function from the given DW_AT_subprogram die.
  lldb_private::Function *
  ParseFunctionFromDWARF(lldb_private::CompileUnit &comp_unit,
                         const lldb_private::plugin::dwarf::DWARFDIE &die,
                         lldb_private::AddressRanges funcRange) override;

  /// Parse recursively all the children of the given die that is a forward
  /// declaration type.
  /// Return true if the type could be completed.
  bool CompleteTypeFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die,
      lldb_private::Type *type,
      const lldb_private::CompilerType &compilerType) override;

  lldb_private::CompilerDecl GetDeclForUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override {
    // Unimplemented.
    return {};
  }

  void EnsureAllDIEsInDeclContextHaveBeenParsed(
      lldb_private::CompilerDeclContext declContext) override {
    // Unimplemented.
  }

  /// Get the decl corresponding to a scoped die.
  lldb_private::CompilerDeclContext GetDeclContextForUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override;

  lldb_private::CompilerDeclContext GetDeclContextContainingUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override {
    // Unimplemented.
    return {};
  }

  std::string GetDIEClassTemplateParams(
      lldb_private::plugin::dwarf::DWARFDIE die) override {
    // Unimplemented.
    return {};
  }

private:
  bool CompleteStructureTypeFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die,
      lldb_private::Type *type, const lldb_private::CompilerType &compilerType);

  /// Set the symbol context scope for the recently created type.
  void updateSymbolContextScopeForType(
      const lldb_private::SymbolContext &sc,
      const lldb_private::plugin::dwarf::DWARFDIE &die, lldb::TypeSP &type);

  /// Get parent decl of the given die.
  ///
  /// If `declDieCopy` is provided, it should be set to the parent decl.
  MojoASTDeclRef getDeclContextContainingDIE(
      const lldb_private::plugin::dwarf::DWARFDIE &die,
      lldb_private::plugin::dwarf::DWARFDIE *declDieCopy);

  /// Get the decl that corresponds to the given die. It has an internal cache
  /// to prevent duplicates.
  MojoASTDeclRef
  getDeclForDIE(const lldb_private::plugin::dwarf::DWARFDIE &die);

  /// Get the decl that corresponds to the given die only if it has been created
  /// already.
  MojoASTDeclRef
  getCachedDeclForDIE(const lldb_private::plugin::dwarf::DWARFDIE &die);

  /// Extract the SourceName of a given die by inspecting its annotations.
  DebugInfo::SourceNameAttr
  extractSourceName(const lldb_private::plugin::dwarf::DWARFDIE &die);

  MojoTypeSystem &typeSystem;

  using DIEToDeclMap =
      llvm::DenseMap<const lldb_private::plugin::dwarf::DWARFDebugInfoEntry *,
                     MojoASTDeclRef>;
  /// Mapping from DWARF DIE to the generated Mojo decl.
  DIEToDeclMap dieToDecl;

  /// List of decls that have been completed by
  /// `MojoDWARFParser::CompleteTypeFromDWARF`.
  llvm::DenseSet<LIT::ASTDecl *> completedDecls;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJODWARFPARSER_H
