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

#ifndef KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H
#define KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "MojoDWARFParser.h"
#include "Support/Context.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"

namespace M {
class MojoASTTypeRef;
class MojoParserContext;
class TargetInfoAttr;
namespace DebugInfo {
class SourceNameAttr;
} // namespace DebugInfo
} // namespace M

namespace M::AsyncRT {
class CPUDevice;
} // namespace M::AsyncRT

namespace M::KGEN::Mojo {
/// Forward declaration for use below.
class MojoDiagnostic;

class MojoTypeSystem : public lldb_private::TypeSystem {
  static char ID;

public:
  MojoTypeSystem(ContextRef ctx, lldb_private::Target *target,
                 const lldb_private::ArchSpec &archSpec);
  ~MojoTypeSystem() override;

  /// Return the MLIR context for this type system.
  MLIRContext *getMLIRContext();

  /// Return the parser's SharedState for this type system.
  LIT::SharedState &getSharedState();

  /// Return the AsyncRT cpuDevice for this type system.
  AsyncRT::CPUDevice &getCPUDevice();

  /// Return the Mojo parser context attached to this type system.
  MojoParserContext &getParserContext();

  /// Return the target info that corresponds to the current LLDB target, it
  /// might be invalid if it couldn't be computed.
  TargetInfoAttr GetTargetInfo() const;

  /// Return if the given language is supported by this type system.
  bool SupportsLanguage(lldb::LanguageType language) override {
    return language == lldb::eLanguageTypeMojo;
  }

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  using CreateContextFn = ContextRef (*)(void);
  static void Initialize(CreateContextFn fn);
  static void Terminate();

  llvm::StringRef GetPluginName() override { return getPluginNameStatic(); }
  static llvm::StringRef getPluginNameStatic() { return "Mojo"; }

  //===--------------------------------------------------------------------===//
  // Expression Parsing
  //===--------------------------------------------------------------------===//

  /// Push a new working directory to the parser context.
  void pushWorkingDirectory(StringRef workingDirectory);

  /// Pop the last working directory from the parser context.
  void popWorkingDirectory();

  /// Add a set of directories to the parser context's import directories.
  void addImportDirectories(ArrayRef<std::string> directories);

  //===--------------------------------------------------------------------===//
  // Dumping
  //===--------------------------------------------------------------------===//

  void Dump(llvm::raw_ostream &output, llvm::StringRef filter,
            bool show_color) override;

  bool DumpTypeValue(lldb::opaque_compiler_type_t type, lldb_private::Stream &s,
                     lldb::Format format,
                     const lldb_private::DataExtractor &data,
                     lldb::offset_t dataOffset, size_t dataByteSize,
                     uint32_t bitfieldBitSize, uint32_t bitfieldBitOffset,
                     lldb_private::ExecutionContextScope *exeScope) override;

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump(lldb::opaque_compiler_type_t type) const override {
  }
#endif

  /// Dump the type to stdout.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override;

  /// Print a description of the type to a stream. The exact implementation
  /// varies, but the expectation is that eDescriptionLevelFull returns a
  /// source-like representation of the type, whereas eDescriptionLevelVerbose
  /// does a dump of the underlying AST if applicable.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, lldb_private::Stream &s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override;

  //===--------------------------------------------------------------------===//
  // Type Queries
  //===--------------------------------------------------------------------===//

#ifndef NDEBUG
  /// Verify the integrity of the type to catch CompilerTypes that mix
  /// and match invalid TypeSystem/Opaque type pairs.
  bool Verify(lldb::opaque_compiler_type_t type) override {
    // MLIR type construction should already handle verifying the necessary
    // invariants here.
    return true;
  }
#endif

  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override {
    return lldb::eLanguageTypeMojo;
  }

  lldb::Format GetFormat(lldb::opaque_compiler_type_t type) override;

  bool GetCompleteType(lldb::opaque_compiler_type_t type) override {
    // This will be needed when we support incomplete types (like forward
    // declarations) from DWARF.
    return true;
  }

  bool CanPassInRegisters(const lldb_private::CompilerType &type) override {
    // This is used by ThreadPlan::GetReturnValueObject when stepping out of a
    // function to figure out the return value. We don't need this functionality
    // yet.
    return false;
  }

  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override {
    // This is very clang-specific.
    return 0;
  }

  llvm::Expected<lldb_private::CompilerType> GetDereferencedType(
      lldb::opaque_compiler_type_t type,
      lldb_private::ExecutionContext *exe_ctx, std::string &deref_name,
      uint32_t &deref_byte_size, int32_t &deref_byte_offset,
      lldb_private::ValueObject *valobj, uint64_t &language_flags) override {

    bool isTypeValid = IsPointerOrReferenceType(type, nullptr) ||
                       IsArrayType(type, nullptr, nullptr, nullptr);

    if (!isTypeValid)
      return llvm::createStringError("not a pointer, reference or array type");

    uint32_t childBitfieldBitSize = 0;
    uint32_t childBitfieldBitOffset = 0;
    bool childIsBaseClass = false;
    bool childIsDerefOfParent = false;

    return GetChildCompilerTypeAtIndex(
        type, exe_ctx, 0, false, true, false, deref_name, deref_byte_size,
        deref_byte_offset, childBitfieldBitSize, childBitfieldBitOffset,
        childIsBaseClass, childIsDerefOfParent, valobj, language_flags);
  }

  const llvm::fltSemantics &
  GetFloatTypeSemantics(size_t byteSize, lldb::Format format) override {
    // It seems that we should return more specific types only if the target's
    // float standard differs from the host. We don't worry about that for now.
    return llvm::APFloatBase::Bogus();
  }

  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    return 0;
  }

  lldb_private::CompilerType
  GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                             const size_t index) override {
    // Unimplemented.
    return {};
  }

  uint32_t GetPointerByteSize() override;

  lldb_private::CompilerType GetPointerDiffType(bool is_signed) override {
    return {};
  }

  lldb_private::CompilerType GetSizeType() override { return {}; }

  unsigned GetPtrAuthKey(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    return 0;
  }

  unsigned GetPtrAuthDiscriminator(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    return 0;
  }

  bool GetPtrAuthAddressDiversity(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    return false;
  }

  lldb_private::ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                                        bool baseOnly) override;

  lldb_private::ConstString
  GetDisplayTypeName(lldb::opaque_compiler_type_t type) override;

  uint32_t GetTypeInfo(
      lldb::opaque_compiler_type_t type,
      lldb_private::CompilerType *pointeeOrElementCompilerType) override;

  /// An overload of GetTypeInfo that uses a null
  /// `pointeeOrElementCompilerType`.
  uint32_t GetTypeInfo(lldb::opaque_compiler_type_t type) {
    return GetTypeInfo(type, /*pointeeOrElementCompilerType=*/nullptr);
  }

  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override;

  llvm::Expected<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             lldb_private::ExecutionContextScope *exeScope) override;

  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  lldb_private::ExecutionContextScope *exeScope) override;

  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type) override;

  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    // -1 means that this couldn't be computed.
    return -1;
  }

  lldb_private::CompilerType
  GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                 size_t idx) override {
    // Unimplemented.
    return {};
  }

  lldb_private::CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    return {};
  }

  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    return {};
  }

  lldb_private::TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override {
    // Unimplemented.
    return {};
  }

  lldb_private::CompilerType
  GetArrayElementType(lldb::opaque_compiler_type_t type,
                      lldb_private::ExecutionContextScope *exeScope) override {
    // This is C++-specific.
    return {};
  }

  //===--------------------------------------------------------------------===//
  // DeclContext
  //===--------------------------------------------------------------------===//

  lldb_private::ConstString DeclGetName(void *opaqueDecl) override {
    // This will be used when we create decls from DWARF.
    // https://github.com/modularml/modular/issues/20114
    return {};
  }

  lldb_private::CompilerType GetTypeForDecl(void *opaqueDecl) override {
    // This will be used when we create decls from DWARF.
    // https://github.com/modularml/modular/issues/20114
    return {};
  }

  /// Return the name of the given decl context (i.e. a decl that is also a
  /// scope).
  lldb_private::ConstString DeclContextGetName(void *opaqueDeclCtx) override;

  /// Return the type name of the given decl context (i.e. a decl that is also a
  /// scope).
  lldb_private::ConstString
  DeclContextGetScopeQualifiedName(void *opaqueDeclCtx) override;

  bool DeclContextIsClassMethod(void *opaqueDeclCtx) override {
    // This will be used when we create decls from DWARF.
    // https://github.com/modularml/modular/issues/20114
    return {};
  }

  bool DeclContextIsContainedInLookup(void *opaqueDeclCtx,
                                      void *otherOpaqueDeclCtx) override {
    // This will be used when we create decls from DWARF.
    // https://github.com/modularml/modular/issues/20114
    return {};
  }

  lldb::LanguageType DeclContextGetLanguage(void *) override {
    return lldb::eLanguageTypeMojo;
  }

  //===--------------------------------------------------------------------===//
  // IsType Queries
  //===--------------------------------------------------------------------===//

  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override {
    // This method is only used by swift.
    return false;
  }

  bool IsCharType(lldb::opaque_compiler_type_t type) override {
    // We currently don't know if a 1-byte int is a char or not.
    // https://github.com/modularml/modular/issues/23220.
    return false;
  }

  bool IsCompleteType(lldb::opaque_compiler_type_t type) override {
    // We should revisit this when we add support for incomplete types from
    // DWARF.
    return true;
  }

  bool IsConst(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return false;
  }

  bool IsFloatingPointType(lldb::opaque_compiler_type_t type) override;

  bool IsIntegerType(lldb::opaque_compiler_type_t type,
                     bool &isSigned) override;

  bool IsScopedEnumerationType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return false;
  }

  bool IsScalarType(lldb::opaque_compiler_type_t type) override;

  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    lldb_private::CompilerType *elementType,
                    uint64_t *size) override {
    return false;
  }

  uint32_t
  IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                         lldb_private::CompilerType *baseTypePtr) override {
    // This is used to detect "Homogeneous Floating-point Aggregates"
    // Unimplemented.
    return 0;
  }

  bool IsBlockPointerType(
      lldb::opaque_compiler_type_t type,
      lldb_private::CompilerType *functionPointerTypePtr) override {
    // This is Objective-C-specific.
    return false;
  }

  bool IsMemberDataPointerType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return false;
  }

  bool IsMemberFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return false;
  }

  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return false;
  }

  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override {
    // This might be useful when we support incomplete types from DWARF.
    return false;
  }

  bool
  IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                           lldb_private::CompilerType *pointeeType) override;

  bool IsTypedefType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return false;
  }

  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       lldb_private::CompilerType *pointeeType,
                       bool *isRValue) override;

  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   lldb_private::CompilerType *elementType, uint64_t *size,
                   bool *isIncomplete) override {
    // This is C++-specific.
    return false;
  }

  bool IsAggregateType(lldb::opaque_compiler_type_t type) override;

  bool IsDefined(lldb::opaque_compiler_type_t type) override {
    // This might be useful when we support incomplete types from DWARF.
    return false;
  }

  bool IsFunctionType(lldb::opaque_compiler_type_t type) override {
    // This is used by StackFrame to obtain the return value of a function.
    // Unimplemented.
    return false;
  }

  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return false;
  }

  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             lldb_private::CompilerType *targetType,
                             bool checkCplusplus, bool checkObjc) override {
    // This is C++-specific.
    return false;
  }

  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     lldb_private::CompilerType *pointeeType) override;

  bool IsVoidType(lldb::opaque_compiler_type_t type) override {
    // This is only used for the --element-count flag of the expr command.
    // We might be able to replace it with None.
    // Unimplemented.
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Type Creation Queries
  //===--------------------------------------------------------------------===//

  lldb_private::CompilerType
  GetEnumerationIntegerType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return {};
  }

  lldb_private::CompilerType
  GetBasicTypeFromAST(lldb::BasicType basic_type) override {
    // This is C++-specific.
    return {};
  }

  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return lldb::eBasicTypeInvalid;
  }

  lldb_private::CompilerType
  GetLValueReferenceType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return {};
  }

  lldb_private::CompilerType
  GetRValueReferenceType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return {};
  }

  lldb_private::CompilerType
  GetNonReferenceType(lldb::opaque_compiler_type_t type) override;

  lldb_private::CompilerType
  GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                      size_t bitSize) override {
    // Unimplemented.
    return {};
  }

  lldb_private::CompilerType
  GetTypedefedType(lldb::opaque_compiler_type_t type) override {
    // This is C++-specific.
    return {};
  }

  lldb_private::CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override;

  lldb_private::CompilerType GetArrayType(lldb::opaque_compiler_type_t type,
                                          uint64_t size) override {
    // This is C++-specific.
    return {};
  }

  lldb_private::CompilerType
  GetCanonicalType(lldb::opaque_compiler_type_t type) override;

  lldb_private::CompilerType
  GetPointeeType(lldb::opaque_compiler_type_t type) override;

  lldb_private::CompilerType
  GetPointerType(lldb::opaque_compiler_type_t type) override;

  //===--------------------------------------------------------------------===//
  // Type Navigation
  //===--------------------------------------------------------------------===//

  llvm::Expected<uint32_t>
  GetNumChildren(lldb::opaque_compiler_type_t type, bool omitEmptyBaseClasses,
                 const lldb_private::ExecutionContext *exeCtx) override;

  // Note on `POP::UnionType` support: `GetNumChildren` and
  // `GetChildCompilerTypeAtIndex` enumerate one child per union arm at
  // offset 0 so the Variant formatter can drill into the active payload.
  // Other TypeSystem dispatches (`IsAggregateType`, `GetFormat`,
  // `GetByteSize`, …) intentionally don't special-case unions today — if
  // a caller ever needs to treat a raw `!pop.union` as a proper aggregate
  // type, those will need matching handlers.
  uint32_t GetNumFields(lldb::opaque_compiler_type_t type) override {
    // Unimplemented.
    return 0;
  }

  lldb_private::CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type,
                                             size_t idx, std::string &name,
                                             uint64_t *bitOffsetPtr,
                                             uint32_t *bitfieldBitSizePtr,
                                             bool *isBitfieldPtr) override {
    // Unimplemented.
    return {};
  }

  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override {
    // We don't have inheritance.
    return 0;
  }

  uint32_t
  GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override {
    // We don't have inheritance.
    return 0;
  }

  lldb_private::CompilerType
  GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                            uint32_t *bitOffsetPtr) override {
    // We don't have inheritance.
    return {};
  }

  lldb_private::CompilerType
  GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                             uint32_t *bitOffsetPtr) override {
    // We don't have inheritance.
    return {};
  }

  llvm::Expected<lldb_private::CompilerType> GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, lldb_private::ExecutionContext *exeCtx,
      size_t idx, bool transparentPointers, bool omitEmptyBaseClasses,
      bool ignoreArrayBounds, std::string &childName, uint32_t &childByteSize,
      int32_t &childByteOffset, uint32_t &childBitfieldBitSize,
      uint32_t &childBitfieldBitOffset, bool &childIsBaseClass,
      bool &childIsDerefOfParent, lldb_private::ValueObject *valobj,
      uint64_t &languageFlags) override;

  llvm::Expected<uint32_t>
  GetIndexOfChildWithName(lldb::opaque_compiler_type_t type, StringRef name,
                          bool omitEmptyBaseClasses) override;

  // GetIndexOfChildMemberWithName returns a path of child indices towards
  // a member, including descending into child structs.
  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                llvm::StringRef name, bool omitEmptyBaseClasses,
                                std::vector<uint32_t> &childIndices) override;

  //===--------------------------------------------------------------------===//
  // Mojo-specific Type Queries
  //===--------------------------------------------------------------------===//

  /// Return the list of decorators attached to the struct type, or an empty
  /// list if the type is not a struct.
  llvm::ArrayRef<TypedAttr>
  getStructDecorators(lldb::opaque_compiler_type_t type);

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  /// Return a new user expression for the given expression text, or nullptr in
  /// the case of an error.
  lldb_private::UserExpression *
  GetUserExpression(StringRef expr, StringRef prefix,
                    lldb_private::SourceLanguage language,
                    lldb_private::Expression::ResultType desiredType,
                    const lldb_private::EvaluateExpressionOptions &options,
                    lldb_private::ValueObject *ctxObj) override;

  /// Return a pointer to the persistent expression state for this type system.
  lldb_private::PersistentExpressionState *
  GetPersistentExpressionState() override;

  //===--------------------------------------------------------------------===//
  // Utils
  //===--------------------------------------------------------------------===//

  /// Create a CompilerType for the given MLIR type.
  lldb_private::CompilerType createCompilerType(Type type);

  /// Create a CompilerType for the given MojoASTTypeRef.
  lldb_private::CompilerType createCompilerType(MojoASTTypeRef type);

  //===--------------------------------------------------------------------===//
  // Debug Info Parsing
  //===--------------------------------------------------------------------===//

  /// Get the DWARF Parser for Mojo.
  lldb_private::plugin::dwarf::DWARFASTParser *GetDWARFParser() override;

  /// Get the MLIR type for the given type name. If the type couldn't be
  /// recovered, an invalid CompilerType is returned and it's assumed that the
  /// type is not a regular MLIR type.
  lldb_private::CompilerType
  getBuiltinTypeFromMLIRTypeName(llvm::StringRef typeName);

  /// Get the MLIR type for the given scalar DIE entry.
  ///
  /// Return an invalid CompilerType if the couldn't be recovered.
  lldb_private::CompilerType getBuiltinScalarType(llvm::StringRef typeName,
                                                  uint32_t dwarfEncoding,
                                                  uint32_t byteSize);

  /// Get the MLIR type of a given dtype by name.
  lldb_private::CompilerType createCompilerTypeFromDType(StringRef dtype);

  /// Create a SIMD type given a dtype name and a number of elements.
  lldb_private::CompilerType createSIMDType(StringRef dtype,
                                            size_t numElements);

  /// Create a POP::ArrayType given an element type and a number of elements.
  lldb_private::CompilerType
  createPOPArrayType(lldb::opaque_compiler_type_t elementType,
                     size_t numElements);

  MojoASTDeclRef getOrCreatePackageDecl(StringRef name,
                                        MojoASTDeclRef parentDeclRef);

  /// If the given `name` is parsable as a `SourceName`, traverse each decl
  /// and create it unless it's been created already.
  /// If `name` cannot be parsed, then the die tag is used as a fallback to
  /// create a decl.
  /// If `name` is empty, then a placeholder name will be created.
  /// This doesn't work for functions, for which `getOrCreateFunctionDecl`
  /// should be used.
  ///
  /// Return the leaf decl.
  MojoASTDeclRef
  getOrCreateDeclChainForDie(const lldb_private::plugin::dwarf::DWARFDIE &die,
                             StringRef name);

  /// Add a field at the end of the given struct decl and the given type.
  ///
  /// Return the decl of the newly created field.
  MojoASTDeclRef addFieldToStruct(MojoASTDeclRef structDecl,
                                  StringRef fieldName,
                                  lldb::opaque_compiler_type_t type);

private:
  /// Create a struct decl given its name under the given parent decl. If the
  /// parent decl already contains a struct with that name, return it instead.
  /// If the parent decl is null, then a placeholder module will be created.
  ///
  /// Return the decl of the struct.
  MojoASTDeclRef getOrCreateStructDecl(StringRef structName,
                                       MojoASTDeclRef parentDecl);

  /// Create a module decl given its name under the given parentDecl. If the
  /// parent module already contains a module with that name, return it
  /// instead.
  /// If the parent decl is null, then the Mojo parser's top decl will be used.
  ///
  /// Return the decl of the module.
  MojoASTDeclRef getOrCreateModuleDecl(StringRef moduleName,
                                       MojoASTDeclRef parentDecl = {});

  /// Create a function decl given its name under the given parentDecl. If the
  /// parent decl already contains a function with that name, return it
  /// instead.
  /// If the parent decl is null, then the Mojo parser's top decl will be used.
  ///
  /// Return the decl of the funcction.
  MojoASTDeclRef getOrCreateFunctionDecl(StringRef functionName,
                                         MojoASTDeclRef parentDecl = {});

  /// Traverse each decl of the `SourceName` and create it unless it's been
  /// created already.
  ///
  /// Return the leaf decl.
  MojoASTDeclRef
  createDeclsFromSourceNameRecursive(M::DebugInfo::SourceNameAttr sourceName);

  //===--------------------------------------------------------------------===//
  // RTTI Support
  //===--------------------------------------------------------------------===//
public:
  bool isA(const void *classID) const override { return classID == &ID; }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }

protected:
  struct Impl;

  std::unique_ptr<Impl> impl;
};
} // namespace M::KGEN::Mojo

/// Allow cast<MojoTypeSystem>(lldb::TypeSystemSP) ->
/// std::shared_ptr<MojoTypeSystem>. This is necessary because the standard LLVM
/// infra does not support std::shared_ptr.
namespace llvm {
template <>
struct CastInfo<M::KGEN::Mojo::MojoTypeSystem, lldb::TypeSystemSP> {
  using To = std::shared_ptr<M::KGEN::Mojo::MojoTypeSystem>;
  using From = lldb::TypeSystemSP;
  static inline bool isPossible(From &f) {
    return llvm::isa<M::KGEN::Mojo::MojoTypeSystem>(&*f);
  }

  static To doCast(From &f) {
    return std::static_pointer_cast<M::KGEN::Mojo::MojoTypeSystem>(f);
  }

  static inline To castFailed() { return nullptr; }

  static To doCastIfPossible(From &f) {
    if (!isPossible(f))
      return castFailed();
    return doCast(f);
  }
};

template <>
struct CastInfo<M::KGEN::Mojo::MojoTypeSystem, const lldb::TypeSystemSP>
    : public ConstStrippingForwardingCast<
          M::KGEN::Mojo::MojoTypeSystem, const lldb::TypeSystemSP,
          CastInfo<M::KGEN::Mojo::MojoTypeSystem, lldb::TypeSystemSP>> {};
} // namespace llvm

#endif // KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H
