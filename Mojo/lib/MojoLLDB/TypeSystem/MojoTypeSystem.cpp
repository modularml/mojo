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

#include "MojoTypeSystem.h"
#include "../ExpressionParser/MojoDiagnostic.h"
#include "../ExpressionParser/MojoExpressionParser.h"
#include "../ExpressionParser/MojoExpressionVariable.h"
#include "../ExpressionParser/MojoUserExpression.h"
#include "../Utils/Errors.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/KGENDialect/DebugInfoEncoding.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "MojoTypeDataLayout.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/SymbolExport.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/source/Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Process.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;
using namespace mlir;

/// Convert a KGENDType, which is an extension to the regular DType, into
/// approximate MLIR types.
static std::optional<mlir::Type>
getMLIRTypeForDType(MLIRContext *ctx, KGENDType dtype, size_t indexBitwidth) {
  // `address`, `index` (signed), and `uindex` (unsigned) are extensions to
  // the regular dtype.
  if (dtype.isAddress())
    return mlir::LLVM::LLVMPointerType::get(ctx);

  if (dtype.isIndex())
    return IntegerType::get(ctx, indexBitwidth, IntegerType::Signed);
  if (dtype.isUIndex())
    return IntegerType::get(ctx, indexBitwidth, IntegerType::Unsigned);

  // This checks for `bool` and `int` types.
  if (IntegerType intType = getEquivalentIntegerType(ctx, dtype))
    return intType;

  if (FloatType fpType = getEquivalentFloatType(ctx, dtype))
    return fpType;

  if (dtype.isInvalid())
    return KGEN::NoneType::get(ctx);

  return {};
}

/// Method used to get the correct MojoASTTypeRef of a given opaque type. It
/// also dereferences REPLResultRefType if present.
static MojoASTTypeRef
dereferenceIfREPLResult(lldb::opaque_compiler_type_t type) {
  if (!type)
    return MojoASTTypeRef();
  auto mlirType = mlir::Type::getFromOpaquePointer(type);
  if (auto refType = dyn_cast<LIT::REPLResultRefType>(mlirType))
    return MojoASTTypeRef(refType.getElementType());
  return MojoASTTypeRef(mlirType);
}

//===----------------------------------------------------------------------===//
// MojoTypeSystem::Impl
//===----------------------------------------------------------------------===//

struct MojoTypeSystem::Impl {
  Impl(ContextRef ctx, Target *target, const ArchSpec &archSpec)
      : cpuDevice(*ctx->get<AsyncRT::CPUDevice>()), target(target),
        archSpec(archSpec) {
    // Register all of the various dialect state.
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    registerKGENToLLVMTranslation(registry);

    // Register the context.
    registerContext(registry, ctx);

    // Set up the dialects in the context.
    mlirContext.appendDialectRegistry(registry);

    // Compute the target information for the expression.
    compilationOptions.targetTriple = archSpec.GetTriple().str();

    // TODO: Populate cpu information properly here.
    if (archSpec.IsValid()) {
      compilationOptions.targetTriple = archSpec.GetTriple().str();
      compilationOptions.relocModel =
          (archSpec.GetTriple().isOSBinFormatMachO() ||
           archSpec.GetTriple().isAArch64())
              ? llvm::Reloc::PIC_
              : llvm::Reloc::Static;
    }
    compilationOptions.targetCpu = llvm::sys::getHostCPUName();

    // TODO(#33931) workaround to disable module splitting for REPL.
    // TODO(MOTO-247) workaround to LLVM Module splitting which works
    // for ORC JIT but not always for MCJIT. Disable splitting for REPL.
    compilationOptions.enableLLVMPerFunctionSplitting = false;
    compilationOptions.enableParallelLLC = false;

    // Configure the parser context.
    LIT::ParserConfig parserConfig(&mlirContext, compilationOptions);
    parserContext =
        std::make_unique<MojoParserContext>(sourceMgr, parserConfig);

    auto targetInfoOr = M::getTargetInfoFor(
        &mlirContext, compilationOptions.targetTriple,
        compilationOptions.targetCpu, compilationOptions.targetFeatures,
        /*tuneCpu=*/"",
        /*targetAccelerator=*/compilationOptions.targetAccelerator,
        compilationOptions.relocModel);
    if (succeeded(targetInfoOr))
      targetInfo = *targetInfoOr;

    dataLayoutContext =
        std::make_unique<MojoTypeDataLayoutContext>(*parserContext, targetInfo);
  }

  /// Utility that returns a StructDeclOp if the given astType corresponds to a
  /// struct, otherwise an invalid object is returned.
  LIT::StructDeclOp getIfStructDecl(MojoASTTypeRef astType) {
    if (auto declRef = parserContext->getDecl(astType)) {
      if (LIT::StructDeclOp structDeclOp =
              dyn_cast_if_present<LIT::StructDeclOp>(
                  declRef.getIfOperation())) {
        return structDeclOp;
      }
    }
    return {};
  }

  /// The MLIR context to use for compilation/processing associated with this
  /// typesystem.
  MLIRContext mlirContext{MLIRContext::Threading::DISABLED};

  /// The AsyncRT cpuDevice to use for compilation/processing associated with
  /// this type system. This is derived from the context available in the
  /// MLIRContext.
  AsyncRT::CPUDevice &cpuDevice;

  /// The compilation options to use when compiling.
  KGEN::CompilationOptions compilationOptions;

  /// The source manager used for expression compilation.
  llvm::SourceMgr sourceMgr;

  /// The current stack of working directories.
  SmallVector<std::string> expressionWorkingDirectories;

  /// The main parser context used for compilation.
  std::unique_ptr<MojoParserContext> parserContext;

  /// The target that this typesystem is associated with. It's available only
  /// for expression evaluation.
  lldb_private::Target *target;

  lldb_private::ArchSpec archSpec;

  /// The persistent state for this typesystem.
  MojoPersistentExpressionState persistentState;

  /// The target info of the current LLDB Target.
  TargetInfoAttr targetInfo;

  /// The cache to be used for querying data layouts.
  std::unique_ptr<MojoTypeDataLayoutContext> dataLayoutContext;

  std::unique_ptr<MojoDWARFParser> dwarfParser;

  /// A cache for the full type names. The `GetTypeName` method of the type
  /// system gets called very frequently for the same type, hence the value of
  /// having a cache.
  DenseMap<lldb::opaque_compiler_type_t, ConstString> typeNames;
};

//===----------------------------------------------------------------------===//
// MojoTypeSystem
//===----------------------------------------------------------------------===//

MojoTypeSystem::MojoTypeSystem(ContextRef ctx, Target *target,
                               const ArchSpec &archSpec)
    : impl(std::make_unique<Impl>(std::move(ctx), target, archSpec)) {}

MojoTypeSystem::~MojoTypeSystem() = default;
char MojoTypeSystem::ID = 0;

MLIRContext *MojoTypeSystem::getMLIRContext() { return &impl->mlirContext; }

LIT::SharedState &MojoTypeSystem::getSharedState() {
  return impl->parserContext->getSharedState();
}

MojoParserContext &MojoTypeSystem::getParserContext() {
  return *impl->parserContext;
}

TargetInfoAttr MojoTypeSystem::GetTargetInfo() const {
  return impl->targetInfo;
}

AsyncRT::CPUDevice &MojoTypeSystem::getCPUDevice() { return impl->cpuDevice; }

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Context used for createInstance, below.
static std::atomic<MojoTypeSystem::CreateContextFn> createContextFn;

/// Create a MojoTypeSystem instance from the given module and target.
static lldb::TypeSystemSP createInstance(lldb::LanguageType language,
                                         Module *module, Target *target) {
  if (language != lldb::eLanguageTypeMojo)
    return {};

  ArchSpec arch;
  if (module)
    arch = module->GetArchitecture();
  else if (target)
    arch = target->GetArchitecture();

  if (!arch.IsValid())
    return {};

  ContextRef ctx = createContextFn.load()();
  if (!ctx)
    return {};
  return std::make_shared<MojoTypeSystem>(std::move(ctx), target, arch);
}

void MojoTypeSystem::Initialize(CreateContextFn ctxFn) {
  LanguageSet languages;
  languages.Insert(lldb::eLanguageTypeMojo);
  createContextFn.store(ctxFn);
  lldb_private::PluginManager::RegisterPlugin(getPluginNameStatic(),
                                              "Mojo TypeSystem", createInstance,
                                              languages, languages);
}

void MojoTypeSystem::Terminate() {
  lldb_private::PluginManager::UnregisterPlugin(createInstance);
}

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

void MojoTypeSystem::pushWorkingDirectory(StringRef workingDirectory) {
  std::vector<std::string> currentDirs = impl->sourceMgr.getIncludeDirs();

  // Update the include directories to include this new directory.
  if (!impl->expressionWorkingDirectories.empty()) {
    auto it =
        llvm::find(currentDirs, impl->expressionWorkingDirectories.back());
    assert(it != currentDirs.end() &&
           "working directory not found in include directories");
    *it = currentDirs.back();
  } else {
    currentDirs.insert(currentDirs.begin(), workingDirectory.str());
  }

  impl->expressionWorkingDirectories.push_back(workingDirectory.str());
  impl->sourceMgr.setIncludeDirs(currentDirs);
}

void MojoTypeSystem::popWorkingDirectory() {
  if (impl->expressionWorkingDirectories.empty())
    return;
  std::string dir = impl->expressionWorkingDirectories.pop_back_val();

  // Update the include directories to remove this directory.
  std::vector<std::string> currentDirs = impl->sourceMgr.getIncludeDirs();
  auto it = llvm::find(currentDirs, dir);
  assert(it != currentDirs.end() &&
         "working directory not found in include directories");
  if (impl->expressionWorkingDirectories.empty())
    currentDirs.erase(it);
  else
    *it = impl->expressionWorkingDirectories.back();

  impl->sourceMgr.setIncludeDirs(currentDirs);
}

void MojoTypeSystem::addImportDirectories(ArrayRef<std::string> directories) {
  std::vector<std::string> currentDirs = impl->sourceMgr.getIncludeDirs();
  currentDirs.insert(currentDirs.begin(), directories.begin(),
                     directories.end());
  impl->sourceMgr.setIncludeDirs(currentDirs);
}

//===----------------------------------------------------------------------===//
// Type Queries
//===----------------------------------------------------------------------===//

bool MojoTypeSystem::IsPointerOrReferenceType(
    lldb::opaque_compiler_type_t type,
    lldb_private::CompilerType *pointeeType) {
  return IsReferenceType(type, pointeeType, /*isRValue=*/nullptr) ||
         IsPointerType(type, pointeeType);
}

bool MojoTypeSystem::IsAggregateType(lldb::opaque_compiler_type_t type) {
  switch (GetTypeClass(type)) {
  case lldb::eTypeClassArray:
  case lldb::eTypeClassStruct:
  case lldb::eTypeClassVector:
    return true;
  default:
    return false;
  }
}

bool MojoTypeSystem::IsPointerType(lldb::opaque_compiler_type_t type,
                                   lldb_private::CompilerType *pointeeType) {
  MojoASTTypeRef astType(type);
  if (!astType)
    return false;

  if (auto pointerType = dyn_cast<KGEN::PointerType>(astType)) {
    if (pointeeType)
      *pointeeType = createCompilerType(pointerType.getElementType());
    return true;
  }
  return false;
}

bool MojoTypeSystem::IsReferenceType(lldb::opaque_compiler_type_t type,
                                     lldb_private::CompilerType *pointeeType,
                                     bool *isRValue) {
  if (!type)
    return false;
  if (auto refType = dyn_cast<LIT::REPLResultRefType>(MojoASTTypeRef(type))) {
    if (pointeeType)
      *pointeeType = createCompilerType(refType.getElementType());
    return true;
  }
  return false;
}

lldb_private::CompilerType
MojoTypeSystem::GetPointeeType(lldb::opaque_compiler_type_t type) {
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType)
    return {};
  if (auto ptrType = dyn_cast<PointerType>(astType))
    return createCompilerType(ptrType.getElementType());
  return {};
}

lldb_private::CompilerType
MojoTypeSystem::GetPointerType(lldb::opaque_compiler_type_t type) {
  MojoASTTypeRef astType(type);
  if (!astType)
    return {};
  return createCompilerType(KGEN::PointerType::get(astType.getMLIRType()));
}

lldb::TypeClass
MojoTypeSystem::GetTypeClass(lldb::opaque_compiler_type_t type) {
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType)
    return {};

  if (auto ptrType = dyn_cast<PointerType>(astType))
    return lldb::eTypeClassPointer;

  if (isa<SIMDType>(astType))
    return lldb::eTypeClassVector;

  if (isa<POP::ArrayType>(astType))
    return lldb::eTypeClassArray;

  if (impl->getIfStructDecl(astType) || isa<StructType>(astType))
    return lldb::eTypeClassStruct;

  return lldb::eTypeClassOther;
}

uint32_t MojoTypeSystem::GetTypeInfo(
    lldb::opaque_compiler_type_t type,
    lldb_private::CompilerType *pointeeOrElementCompilerType) {
  MojoASTTypeRef astType(type);
  if (!astType)
    return 0;

  if (pointeeOrElementCompilerType)
    pointeeOrElementCompilerType->Clear();

  if (auto ptrType = dyn_cast<PointerType>(astType)) {
    if (pointeeOrElementCompilerType) {
      *pointeeOrElementCompilerType =
          createCompilerType(ptrType.getElementType());
    }
    return lldb::eTypeIsPointer | lldb::eTypeHasChildren | lldb::eTypeHasValue;
  }

  if (isa<IndexType>(astType))
    return lldb::eTypeIsInteger | lldb::eTypeHasValue | lldb::eTypeIsScalar |
           lldb::eTypeIsSigned;

  if (auto intType = dyn_cast<IntegerType>(astType)) {
    auto result =
        lldb::eTypeIsInteger | lldb::eTypeHasValue | lldb::eTypeIsScalar;
    if (intType.isSignedInteger())
      return result | lldb::eTypeIsSigned;
    return result;
  }

  if (isa<FloatType>(astType))
    return lldb::eTypeIsFloat | lldb::eTypeHasValue | lldb::eTypeIsScalar;

  if (auto simdTy = dyn_cast<SIMDType>(astType)) {
    uint32_t flags = lldb::eTypeHasChildren | lldb::eTypeIsVector;
    // Scalar SIMD (size == 1) holds a single numeric value. Without
    // eTypeHasValue, ValueObjectChild::UpdateValue() skips loading m_data but
    // returns true, causing Checksum(null, 0) → UBSAN abort.
    if (simdTy.isScalar() || simdTy.isIndex())
      flags |= lldb::eTypeHasValue;
    return flags;
  }

  if (isa<KGEN::StringType>(astType))
    return lldb::eTypeIsPointer | lldb::eTypeHasChildren | lldb::eTypeHasValue;

  if (impl->getIfStructDecl(astType))
    return lldb::eTypeHasChildren | lldb::eTypeIsClass;

  return {};
}

lldb::Format MojoTypeSystem::GetFormat(lldb::opaque_compiler_type_t type) {
  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger) {
    if (flags & lldb::eTypeIsSigned)
      return lldb::eFormatDecimal;
    return lldb::eFormatUnsigned;
  }
  if (flags & lldb::eTypeIsFloat)
    return lldb::eFormatFloat;
  if (flags & lldb::eTypeIsPointer) {
    if (isa<KGEN::StringType>(MojoASTTypeRef(type)))
      return lldb::eFormatCString;
    return lldb::eFormatHex;
  }
  if (flags & lldb::eTypeIsClass)
    return lldb::eFormatHex;
  if (flags & lldb::eTypeIsFuncPrototype || flags & lldb::eTypeIsBlock)
    return lldb::eFormatAddressInfo;
  return lldb::eFormatBytes;
}

lldb_private::CompilerType
MojoTypeSystem::GetNonReferenceType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(dereferenceIfREPLResult(type));
}

lldb_private::CompilerType
MojoTypeSystem::GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(type);
}

lldb_private::CompilerType
MojoTypeSystem::GetCanonicalType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(type);
}

uint32_t MojoTypeSystem::GetPointerByteSize() {
  return impl->targetInfo.getDataLayout().getPointerSize();
}

llvm::Expected<uint64_t>
MojoTypeSystem::GetBitSize(lldb::opaque_compiler_type_t type,
                           lldb_private::ExecutionContextScope *exeScope) {
  MojoASTTypeRef astType(type);
  if (!astType)
    return llvm::createStringError("Invalid type: Cannot determine size");

  if (auto &layout = impl->dataLayoutContext->getOrCalculate(astType))
    return layout->getByteSize() * CHAR_BIT;

  return llvm::createStringError("Unknown type: Cannot determine size");
}

std::optional<size_t>
MojoTypeSystem::GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                                lldb_private::ExecutionContextScope *exeScope) {
  MojoASTTypeRef astType(type);
  if (!astType)
    return {};

  if (auto &layout = impl->dataLayoutContext->getOrCalculate(astType))
    return layout->getAlignment() * CHAR_BIT;

  return {};
}

lldb::Encoding MojoTypeSystem::GetEncoding(lldb::opaque_compiler_type_t type) {
  MojoASTTypeRef astType(type);
  if (!astType)
    return lldb::eEncodingInvalid;

  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger) {
    if (flags & lldb::eTypeIsSigned)
      return lldb::eEncodingSint;
    return lldb::eEncodingUint;
  }

  if (flags & lldb::eTypeIsFloat)
    return lldb::eEncodingIEEE754;

  if (flags & lldb::eTypeIsPointer)
    return lldb::eEncodingUint;

  return lldb::eEncodingInvalid;
}

ConstString MojoTypeSystem::GetTypeName(lldb::opaque_compiler_type_t type,
                                        bool baseOnly) {
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType)
    return {};
  mlir::Type mlirType = astType.getMLIRType();
  lldb::opaque_compiler_type_t opaqueType =
      const_cast<lldb::opaque_compiler_type_t>(mlirType.getAsOpaquePointer());

  if (auto it = impl->typeNames.find(opaqueType); it != impl->typeNames.end())
    return it->second;

  std::string name;
  llvm::raw_string_ostream os(name);
  mlirType.print(os);

  // We include the decorators in the full type name so that parts of LLDB
  // that perform queries based on type names can operate also on decorators. An
  // example of this are the data formatters.
  for (TypedAttr decorator : getStructDecorators(opaqueType)) {
    if (auto constantSymbol = dyn_cast<KGEN::SymbolConstantAttr>(decorator)) {
      os << " {@" << M::getFlattenedSymbolName(constantSymbol.getSymbol())
         << "}";
    }
  }
  return impl->typeNames.insert({opaqueType, ConstString(name)}).first->second;
}

ConstString
MojoTypeSystem::GetDisplayTypeName(lldb::opaque_compiler_type_t type) {
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType)
    return {};

  std::string name = astType.getAsString(impl->parserContext->getSharedState());

  auto mangledOr =
      LIT::MangledSymbol::demangle(StringAttr::get(&impl->mlirContext, name));

  // We need to delete the artificial module we use for expression evaluations
  // to avoid confusing the user.
  if (succeeded(mangledOr) && !mangledOr->moduleNames.empty() &&
      MojoPersistentExpressionState::isExpressionModuleName(
          mangledOr->moduleNames.back()))
    return ConstString(mangledOr->symName);

  return ConstString(name);
}

//===----------------------------------------------------------------------===//
// IsType Queries
//===----------------------------------------------------------------------===//

bool MojoTypeSystem::IsFloatingPointType(lldb::opaque_compiler_type_t type) {
  if (GetTypeInfo(type) & lldb::eTypeIsFloat) {
    return true;
  }
  return false;
}

bool MojoTypeSystem::IsIntegerType(lldb::opaque_compiler_type_t type,
                                   bool &isSigned) {
  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger) {
    isSigned = flags & lldb::eTypeIsSigned;
    return true;
  }
  return false;
}

bool MojoTypeSystem::IsScalarType(lldb::opaque_compiler_type_t type) {
  return GetTypeInfo(type) & lldb::eTypeIsScalar;
}

//===--------------------------------------------------------------------===//
// Type Navigation
//===--------------------------------------------------------------------===//

llvm::Expected<uint32_t>
MojoTypeSystem::GetNumChildren(lldb::opaque_compiler_type_t type,
                               bool omitEmptyBaseClasses,
                               const lldb_private::ExecutionContext *exeCtx) {
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType)
    return 0;

  if (auto ptrType = dyn_cast<KGEN::PointerType>(astType)) {
    CompilerType eltType = createCompilerType(ptrType.getElementType());
    if (eltType.IsAggregateType())
      return eltType.GetNumChildren(omitEmptyBaseClasses, exeCtx);
    return 1;
  }

  if (auto simdTy = dyn_cast<SIMDType>(astType)) {
    if (simdTy.isScalar())
      return 1;
    return simdTy.getResolvedSize().value_or(0);
  }

  if (auto arrayType = dyn_cast<POP::ArrayType>(astType))
    return arrayType.getResolvedSize().value_or(0);

  if (LIT::StructDeclOp structDecl = impl->getIfStructDecl(astType))
    return llvm::range_size(structDecl.getFieldDecls());

  if (auto structType = dyn_cast<StructType>(astType)) {
    std::optional<size_t> numElements = structType.getNumElements();
    return numElements.value_or(0);
  }

  // One for the discriminator, one for each variant.
  if (auto variantType = dyn_cast<VariantType>(astType))
    return 1 + variantType.getNumTypes();

  // A union exposes one child per variant type, matching the DWARF
  // `DW_TAG_variant_part` member names `v0`, `v1`, ..., emitted by
  // `KGEN::DebugInfoTypeConverter::buildDebugType(POP::UnionType)`. Note:
  // only one of these children is "live" at any point; they all share
  // offset 0 and are intended to be selected by a sibling discriminant
  // (see `_DefaultVariantStorage._impl` lowering in
  // `Mojo/lib/Transforms/LowerCallingConventions.cpp::lowerVariantType`).
  // Callers inspecting a `!pop.union` directly must pick the correct child
  // based on the discriminant — displaying all of them is a deliberate
  // trade-off that keeps the type system simple.
  if (auto unionType = dyn_cast<POP::UnionType>(astType))
    return unionType.getNumTypes();

  return 0;
}

llvm::Expected<lldb_private::CompilerType>
MojoTypeSystem::GetChildCompilerTypeAtIndex(
    lldb::opaque_compiler_type_t type, lldb_private::ExecutionContext *exeCtx,
    size_t idx, bool transparentPointers, bool omitEmptyBaseClasses,
    bool ignoreArrayBounds, std::string &childName, uint32_t &childByteSize,
    int32_t &childByteOffset, uint32_t &childBitfieldBitSize,
    uint32_t &childBitfieldBitOffset, bool &childIsBaseClass,
    bool &childIsDerefOfParent, lldb_private::ValueObject *valobj,
    uint64_t &languageFlags) {
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType)
    return lldb_private::CompilerType();

  if (!ignoreArrayBounds &&
      idx >= getExpectedValueOr(
                 GetNumChildren(type, omitEmptyBaseClasses, exeCtx), 0u))
    return lldb_private::CompilerType();

  // Pointer only has one child, so just return the unwrapped pointer type
  if (auto ptrType = dyn_cast<PointerType>(astType.getMLIRType())) {
    MojoASTTypeRef eltType(ptrType.getElementType());
    CompilerType eltCompilerType = createCompilerType(eltType);

    // If transparentPointers is true, LLDB expects that a pointer is equivalent
    // to its child aggregate type wrt traversing children.
    if (transparentPointers && eltCompilerType.IsAggregateType()) {
      childIsDerefOfParent = false;
      bool tmpChildIsDerefOfParent = false;
      return eltCompilerType.GetChildCompilerTypeAtIndex(
          exeCtx, idx, transparentPointers, omitEmptyBaseClasses,
          ignoreArrayBounds, childName, childByteSize, childByteOffset,
          childBitfieldBitSize, childBitfieldBitOffset, childIsBaseClass,
          tmpChildIsDerefOfParent, valobj, languageFlags);
    }

    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(eltType)) {
      childByteSize = layout->getByteSize();
      childByteOffset = 0;
      childIsDerefOfParent = true;
      const char *parentName =
          valobj ? valobj->GetName().GetCString() : nullptr;
      if (parentName) {
        childName.assign(1, '*');
        childName += parentName;
      }
      return createCompilerType(eltType);
    }
    return lldb_private::CompilerType();
  }

  if (auto simdType = dyn_cast<SIMDType>(astType)) {
    if (std::optional<KGENDType> kgenDTypeOpt = simdType.getResolvedDType()) {
      if (kgenDTypeOpt.has_value()) {
        std::optional<mlir::Type> eltMlirType = getMLIRTypeForDType(
            getMLIRContext(), *kgenDTypeOpt, 8 * GetPointerByteSize());
        if (!eltMlirType)
          return lldb_private::CompilerType();
        MojoASTTypeRef eltType(*eltMlirType);

        if (const std::optional<MojoTypeDataLayout> &layout =
                impl->dataLayoutContext->getOrCalculate(eltType)) {
          childName = std::string(llvm::formatv("[{0}]", idx));
          childByteSize = layout->getByteSize();
          childByteOffset = (int32_t)idx * (int32_t)childByteSize;
          return createCompilerType(eltType);
        } else {
          return lldb_private::CompilerType();
        }
      }
    }
  }

  if (auto arrayType = dyn_cast<POP::ArrayType>(astType)) {
    MojoASTTypeRef eltType(arrayType.getElementType());
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(eltType)) {
      childName = std::string(llvm::formatv("[{0}]", idx));
      childByteSize = layout->getByteSize();
      childByteOffset = (int32_t)idx * (int32_t)childByteSize;
      return createCompilerType(eltType);
    }
    return lldb_private::CompilerType();
  }

  if (LIT::StructDeclOp structDeclOp = impl->getIfStructDecl(astType)) {
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(astType)) {
      auto fieldDecl = *std::next(structDeclOp.getFieldDecls().begin(), idx);
      childName.assign(fieldDecl.getName());
      const auto &field = layout->getFields()[idx];
      childByteOffset = field.getByteOffset();
      childByteSize = field.getByteSize();
      return createCompilerType(field.getConcreteType());
    }
    return lldb_private::CompilerType();
  }

  if (isa<StructType>(astType)) {
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(astType)) {
      childName = std::string(llvm::formatv("[{0}]", idx));
      const auto &field = layout->getFields()[idx];
      childByteOffset = field.getByteOffset();
      childByteSize = field.getByteSize();
      return createCompilerType(field.getConcreteType());
    }
    return lldb_private::CompilerType();
  }

  if (auto variantType = dyn_cast<VariantType>(astType)) {
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(astType)) {
      // Name is hardcoded for now until we get proper DI variant type emission.
      childName = idx == variantType.getNumTypes()
                      ? "discriminator"
                      : std::string(llvm::formatv("variant[{0}]", idx));
      const auto &field = layout->getFields()[idx];
      childByteOffset = field.getByteOffset();
      childByteSize = field.getByteSize();
      return createCompilerType(field.getConcreteType());
    }
    return lldb_private::CompilerType();
  }

  if (auto unionType = dyn_cast<POP::UnionType>(astType)) {
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(astType)) {
      // Child names match DWARF emission: `v0`, `v1`, ...
      childName = std::string(llvm::formatv("v{0}", idx));
      const auto &field = layout->getFields()[idx];
      childByteOffset = field.getByteOffset();
      childByteSize = field.getByteSize();
      return createCompilerType(field.getConcreteType());
    }
    return lldb_private::CompilerType();
  }
  return lldb_private::CompilerType();
}

llvm::Expected<uint32_t>
MojoTypeSystem::GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                        StringRef name,
                                        bool omitEmptyBaseClasses) {
  if (!type)
    return llvm::createStringError("Invalid type");

  if (name.empty())
    return llvm::createStringError("Empty child name");

  std::vector<uint32_t> childIndices;
  GetIndexOfChildMemberWithName(type, name, omitEmptyBaseClasses, childIndices);

  if (childIndices.empty())
    return llvm::createStringError("Child not found");

  if (childIndices.size() > 1)
    return llvm::createStringError(
        "Ambiguous child name, multiple matches found");

  return childIndices[0];
}

size_t MojoTypeSystem::GetIndexOfChildMemberWithName(
    lldb::opaque_compiler_type_t type, llvm::StringRef name,
    bool omitEmptyBaseClasses, std::vector<uint32_t> &childIndices) {
  // This method should return the total number of indices in `childIndices`
  // in the case of success. As a remark, the `childIndices` vector passed in
  // might not be empty.
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType || name.empty())
    return 0;

  // LLDB expects that a pointer is equivalent to its child aggregate type wrt
  // traversing children.
  if (auto ptrType = dyn_cast<KGEN::PointerType>(astType)) {
    CompilerType eltType = createCompilerType(ptrType.getElementType());
    if (eltType.IsAggregateType()) {
      return eltType.GetIndexOfChildMemberWithName(name, omitEmptyBaseClasses,
                                                   childIndices);
    }
  }

  // Check if the name is an index of a SIMD which is 0-indexed.
  if (isa<StructType, SIMDType, POP::ArrayType>(astType)) {
    unsigned long index;
    if (name.consume_front("[") && !name.consumeInteger(10, index) &&
        name.consume_front("]") && name.empty()) {
      childIndices.push_back(index);
      return childIndices.size();
    }
    return 0;
  }

  // Check if it's a field of a struct.
  if (LIT::StructDeclOp structDeclOp = impl->getIfStructDecl(astType)) {
    for (auto field : llvm::enumerate(structDeclOp.getFieldDecls())) {
      if (field.value().getName() == name) {
        childIndices.push_back(field.index());
        return childIndices.size();
      }
    }
    return 0;
  }

  return 0;
}

//===--------------------------------------------------------------------===//
// Mojo-specific Type Queries
//===--------------------------------------------------------------------===//

llvm::ArrayRef<TypedAttr>
MojoTypeSystem::getStructDecorators(lldb::opaque_compiler_type_t type) {
  MojoASTTypeRef astType = dereferenceIfREPLResult(type);
  if (!astType)
    return {};
  if (LIT::StructDeclOp structDeclOp = impl->getIfStructDecl(astType))
    return structDeclOp.getDecorators();
  return {};
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

UserExpression *MojoTypeSystem::GetUserExpression(
    StringRef expr, StringRef prefix, lldb_private::SourceLanguage language,
    Expression::ResultType desiredType,
    const EvaluateExpressionOptions &options, ValueObject *ctxObj) {
  if (!impl->target || ctxObj)
    return nullptr;
  return new MojoUserExpression(*impl->target, expr, prefix, language,
                                desiredType, options);
}

PersistentExpressionState *MojoTypeSystem::GetPersistentExpressionState() {
  return &impl->persistentState;
}

//===--------------------------------------------------------------------===//
// Debug info parsing
//===--------------------------------------------------------------------===//

DWARFASTParser *MojoTypeSystem::GetDWARFParser() {
  if (!impl->dwarfParser)
    impl->dwarfParser = std::make_unique<MojoDWARFParser>(*this);
  return impl->dwarfParser.get();
}

CompilerType
MojoTypeSystem::getBuiltinTypeFromMLIRTypeName(llvm::StringRef typeName) {
  if (typeName.empty())
    return {};
  ScopedDiagnosticHandler diagHandler(getMLIRContext(), [&](Diagnostic &diag) {
    // These logs can get extremely noisy when attempting to parse the DWARF
    // of builtin types, so we only enable them if `verbose` is on.
    if (Log *log = GetLog(LLDBLog::Types); log && log->GetVerbose()) {
      LLDB_LOG(log,
               "[MojoTypeSystem::getBuiltinTypeFromMLIRTypeName] MLIR "
               "diagnostic: {0}",
               diag.str());
    }
  });
  if (auto type = mlir::parseType(typeName, getMLIRContext()))
    return createCompilerType(type);
  return {};
}

CompilerType MojoTypeSystem::getBuiltinScalarType(llvm::StringRef typeName,
                                                  uint32_t dwarfEncoding,
                                                  uint32_t byteSize) {
  if (succeeded(DebugInfoEncoding::getKGENDTypeFromString(typeName)))
    return createCompilerTypeFromDType(typeName);

  if (dwarfEncoding == DW_ATE_unsigned || dwarfEncoding == DW_ATE_signed)
    return getBuiltinTypeFromMLIRTypeName(typeName);

  return {};
}

lldb_private::CompilerType
MojoTypeSystem::createCompilerTypeFromDType(StringRef dtype) {
  auto dTypeOr = DebugInfoEncoding::getKGENDTypeFromString(dtype);
  if (failed(dTypeOr))
    return {};
  return createCompilerType(*getMLIRTypeForDType(getMLIRContext(), *dTypeOr,
                                                 8 * GetPointerByteSize()));
}

lldb_private::CompilerType MojoTypeSystem::createSIMDType(StringRef dtype,
                                                          size_t numElements) {
  auto dTypeOr = DebugInfoEncoding::getKGENDTypeFromString(dtype);
  if (failed(dTypeOr))
    return {};
  return createCompilerType(
      KGEN::SIMDType::get(getMLIRContext(), numElements, *dTypeOr));
}

lldb_private::CompilerType
MojoTypeSystem::createPOPArrayType(lldb::opaque_compiler_type_t elementType,
                                   size_t numElements) {
  auto mlirEltType = mlir::Type::getFromOpaquePointer(elementType);
  return createCompilerType(POP::ArrayType::get(numElements, mlirEltType));
}

/// This looks up the specified name in the indicated decl and returns a match
/// if one exists, or null if not.
static LIT::ASTDecl *lookupSingleMember(LIT::ASTDecl &decl, StringAttr name) {
  // Mark the decl as fully resolved so we can look up into it.  This is pretty
  // unfortunate, should we resolve the decl instead?
  auto oldResolvedness = decl.resolvedness;
  decl.resolvedness = LIT::DeclResolvedness::body;

  ArrayRef<LIT::ASTDecl *> existingDecls = decl.lookupInCurrentScope(name);
  decl.resolvedness = oldResolvedness;

  if (existingDecls.empty())
    return nullptr;
  assert(existingDecls.size() == 1 &&
         "We expect one single decl with a given name");
  return existingDecls[0];
}

MojoASTDeclRef
MojoTypeSystem::getOrCreatePackageDecl(StringRef name,
                                       MojoASTDeclRef parentDeclRef) {
  LIT::SharedState &sharedState = impl->parserContext->getSharedState();

  LIT::ASTDecl &parentDecl =
      parentDeclRef ? *parentDeclRef
                    : impl->parserContext->getSharedState().getTopLevelDecl();

  // We first check if the package already exists, in which case we just return
  // its decl.
  if (auto decl = lookupSingleMember(parentDecl,
                                     StringAttr::get(getMLIRContext(), name)))
    return decl;

  auto moduleBuilder = parentDecl.getDeclEndBuilder();
  auto packageName = StringAttr::get(sharedState.getContext(), name);
  auto packageOp = LIT::PackageOp::create(
      moduleBuilder, sharedState.translateLocation(parentDecl.getLoc()),
      packageName);

  return &sharedState.declResolver->addDecl(
      packageOp, SMLoc(), packageName, &parentDecl, parentDecl.getCursor(),
      parentDecl.getCursor(), /*indentation=*/-1);
}

/// Generate a human readable version of an identifier along with its parameters
/// given a SourceName.
static std::string generateParametrizedName(DebugInfo::SourceNameAttr attr) {
  ArrayRef<StringAttr> paramValues = attr.getParamValues();
  std::string displayName;
  llvm::raw_string_ostream os(displayName);
  os << attr.getName().getValue();
  if (!attr.getParamTypes().empty()) {
    os << "[";
    ArrayRef<DebugInfo::SourceNameAttr> paramTypes = attr.getParamTypes();
    size_t paramDisplayCount = std::min(paramTypes.size(), paramValues.size());
    llvm::interleaveComma(paramTypes.take_front(paramDisplayCount), os,
                          [&](DebugInfo::SourceNameAttr paramType) {
                            StringRef paramValue =
                                paramValues.front().getValue();
                            paramValue.consume_front(":type ");
                            os << paramValue;
                            paramValues = paramValues.drop_front(1);
                          });
    // If we don't have all required parameters available, we just show `...`
    if (paramDisplayCount < paramTypes.size()) {
      if (paramDisplayCount > 0)
        os << ", ";
      os << "...";
    }
    os << "]";
  }
  return displayName;
}

MojoASTDeclRef MojoTypeSystem::createDeclsFromSourceNameRecursive(
    DebugInfo::SourceNameAttr sourceName) {
  MojoASTDeclRef parentDeclRef;
  if (DebugInfo::SourceNameAttr parent = sourceName.getParent())
    parentDeclRef = createDeclsFromSourceNameRecursive(parent);

  switch (sourceName.getKind()) {
  case DebugInfo::SourceNameKind::Package:
    return getOrCreatePackageDecl(sourceName.getName(), parentDeclRef);
  case DebugInfo::SourceNameKind::Module:
    return getOrCreateModuleDecl(sourceName.getName(), parentDeclRef);
  case DebugInfo::SourceNameKind::Struct:
    return getOrCreateStructDecl(generateParametrizedName(sourceName),
                                 parentDeclRef);
  case DebugInfo::SourceNameKind::Fn:
    return getOrCreateFunctionDecl(generateParametrizedName(sourceName),
                                   parentDeclRef);
  default:
    return {};
  }
}

MojoASTDeclRef MojoTypeSystem::getOrCreateDeclChainForDie(const DWARFDIE &die,
                                                          StringRef name) {
  std::string effectiveName = name.str();
  if (name.empty() && die.Tag() != DW_TAG_subroutine_type &&
      die.Tag() != DW_TAG_inlined_subroutine) {
    die.GetDWARF()->GetObjectFile()->GetModule()->ReportWarning(
        "[MojoTypeSystem::getDeclForDie]: {0} has an empty name. Die = "
        "{1:x16}.",
        DW_TAG_value_to_name(die.Tag()), die.GetOffset());

    effectiveName = "__lldb_anonymous__" + std::to_string(die.GetOffset());
  }

  ErrorOr<DebugInfo::SourceNameAttr> sourceNameOr =
      DebugInfo::SourceNameAttr::decode(getMLIRContext(), effectiveName);
  if (succeeded(sourceNameOr) &&
      sourceNameOr->getKind() != DebugInfo::SourceNameKind::Unknown)
    return createDeclsFromSourceNameRecursive(*sourceNameOr);

  switch (die.Tag()) {
  case DW_TAG_compile_unit:
    return getOrCreateModuleDecl(effectiveName, MojoASTDeclRef{});
  case DW_TAG_structure_type: {
    return getOrCreateStructDecl(effectiveName, MojoASTDeclRef{});
  case DW_TAG_subprogram:
  case DW_TAG_subroutine_type:
  case DW_TAG_inlined_subroutine:
    return getOrCreateFunctionDecl(effectiveName, MojoASTDeclRef{});
  }
  default:
    return {};
  }
}

MojoASTDeclRef
MojoTypeSystem::getOrCreateModuleDecl(StringRef moduleName,
                                      MojoASTDeclRef parentDeclRef) {
  LIT::SharedState &sharedState = impl->parserContext->getSharedState();
  LIT::ASTDecl &parentDecl =
      parentDeclRef ? *parentDeclRef
                    : impl->parserContext->getSharedState().getTopLevelDecl();

  // We first check if the module already exists, in which case we just return
  // its decl.
  if (auto decl = lookupSingleMember(
          parentDecl, StringAttr::get(getMLIRContext(), moduleName)))
    return decl;

  // We create a fake empty file so that parser diagnostics can be emitted if
  // we are doing something wrong when creating the decls. Otherwise, we hit
  // asserts and LLDB aborts.
  auto loc = FileLineColLoc::get(getMLIRContext(), moduleName, /*line=*/1,
                                 /*column=*/1);
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBufferCopy("", loc.getFilename().getValue());
  auto &sourceMgr = impl->parserContext->getSourceMgr();
  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(
      sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc()));
  LIT::Lexer lexer(impl->parserContext->getSharedState().diags, sourceBuf);

  auto name = StringAttr::get(sharedState.getContext(), moduleName);

  OpBuilder builder = parentDecl.getDeclEndBuilder();
  Operation *fileOp = LIT::FileModuleOp::create(
      builder, sharedState.translateLocation(parentDecl.getLoc()), name);
  return &sharedState.declResolver->addFullyResolvedDecl(
      fileOp, name, lexer.getToken().getLoc(), &parentDecl);
}

MojoASTDeclRef
MojoTypeSystem::getOrCreateFunctionDecl(StringRef functionName,
                                        MojoASTDeclRef parentDecl) {
  if (!parentDecl)
    parentDecl = getOrCreateModuleDecl("__lldb_anonymous__");

  StringAttr name = StringAttr::get(getMLIRContext(), functionName);

  // We first check if the function already exists, in which case we just return
  // its decl.
  if (auto decl = lookupSingleMember(*parentDecl, name))
    return decl;

  LIT::SharedState &sharedState = impl->parserContext->getSharedState();
  auto builder = parentDecl->getDeclEndBuilder();
  auto fnType = builder.getFunctionType({}, {NoneType::get(getMLIRContext())});
  // We might need to fill in the full signature when expression evaluation is
  // needed. We don't need it for now.
  auto metadata = LIT::FnMetaOriginDataAttr::get(getMLIRContext());
  auto signature = LIT::FnTypeGeneratorType::get(
      {}, fnType, {}, {}, metadata, PogListAttr::get(getMLIRContext()));

  // FIXME(23810): We need to support nested functions.

  StringAttr nameAttr =
      LIT::DeclResolver::getMangledName(name, *parentDecl, signature);
  auto newFunction = LIT::FnOp::create(
      builder, sharedState.translateLocation(parentDecl->getLoc()), nameAttr,
      name, signature);
  return MojoASTDeclRef(&sharedState.declResolver->addDecl(
      newFunction, parentDecl->getLoc(), name, &*parentDecl, {}, {}, -1));
}

MojoASTDeclRef
MojoTypeSystem::getOrCreateStructDecl(StringRef structName,
                                      MojoASTDeclRef parentDecl) {
  if (!parentDecl)
    parentDecl = getOrCreateModuleDecl("__lldb_anonymous__");

  StringAttr name = StringAttr::get(getMLIRContext(), structName);

  // We first check if the struct already exists, in which case we just return
  // its decl.
  if (auto decl = lookupSingleMember(*parentDecl, name))
    return decl;

  OpBuilder builder = parentDecl->getDeclEndBuilder();
  auto newStruct = LIT::StructDeclOp::create(
      builder, getSharedState().translateLocation(parentDecl->getLoc()), name);
  return MojoASTDeclRef(&getSharedState().declResolver->addDecl(
      newStruct, parentDecl->getLoc(), name, &*parentDecl, {}, {}, -1));
}

MojoASTDeclRef
MojoTypeSystem::addFieldToStruct(MojoASTDeclRef structDecl, StringRef fieldName,
                                 lldb::opaque_compiler_type_t type) {
  StringAttr name = StringAttr::get(getMLIRContext(), fieldName);
  OpBuilder builder = structDecl->getDeclEndBuilder();
  auto newField = LIT::StructFieldOp::create(
      builder, getSharedState().translateLocation(structDecl->getLoc()), name,
      mlir::Type::getFromOpaquePointer(type), LIT::DocStringAttr(), false);
  auto fieldDecl = MojoASTDeclRef(&getSharedState().declResolver->addDecl(
      newField, structDecl->getLoc(), name, &*structDecl, {}, {}, -1));
  impl->dataLayoutContext->invalidateCache(structDecl.getType());
  return fieldDecl;
}

ConstString
MojoTypeSystem::DeclContextGetScopeQualifiedName(void *opaqueDeclCtx) {
  if (!opaqueDeclCtx)
    return {};
  auto *decl = static_cast<LIT::ASTDecl *>(opaqueDeclCtx);
  return ConstString(MojoASTDeclRef(decl).getType().getAsString(
      impl->parserContext->getSharedState()));
}

ConstString MojoTypeSystem::DeclContextGetName(void *opaqueDeclCtx) {
  if (!opaqueDeclCtx) {
    if (std::optional<StringRef> name =
            MojoASTDeclRef(static_cast<LIT::ASTDecl *>(opaqueDeclCtx))
                .getName()) {
      return ConstString(*name);
    }
  }
  return {};
}

//===--------------------------------------------------------------------===//
// Dumping
//===--------------------------------------------------------------------===//

void MojoTypeSystem::Dump(llvm::raw_ostream &output, llvm::StringRef filter,
                          bool show_color) {
  // FIXME: handle filtering and color output
  impl->parserContext->getModule()->dump();
}

bool MojoTypeSystem::DumpTypeValue(
    lldb::opaque_compiler_type_t type, lldb_private::Stream &s,
    lldb::Format format, const lldb_private::DataExtractor &data,
    lldb::offset_t dataOffset, size_t dataByteSize, uint32_t bitfieldBitSize,
    uint32_t bitfieldBitOffset, lldb_private::ExecutionContextScope *exeScope) {
  if (!type)
    return false;
  return lldb_private::DumpDataExtractor(
      data, &s, dataOffset, format, dataByteSize,
      /*itemCount=*/1, UINT32_MAX, LLDB_INVALID_ADDRESS, bitfieldBitSize,
      bitfieldBitOffset, exeScope);
}

void MojoTypeSystem::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                         lldb::DescriptionLevel level) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, s, level);
}

void MojoTypeSystem::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                         Stream &s,
                                         lldb::DescriptionLevel level) {
  if (!type)
    return;
  // TODO: complete the implementation. This should dump the type in a way that
  // resembles the source code.
  s << GetDisplayTypeName(type);
}

//===--------------------------------------------------------------------===//
// Utils
//===--------------------------------------------------------------------===//

lldb_private::CompilerType MojoTypeSystem::createCompilerType(mlir::Type type) {
  return lldb_private::CompilerType(
      weak_from_this(), const_cast<void *>(type.getAsOpaquePointer()));
}

lldb_private::CompilerType
MojoTypeSystem::createCompilerType(MojoASTTypeRef astType) {
  return createCompilerType(astType.getMLIRType());
}
