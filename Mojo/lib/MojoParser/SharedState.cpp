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
//
// This file provides the implementation of the SharedState class.
//
//===----------------------------------------------------------------------===//

#include "ClosureEmitter.h"
#include "DebugInfo.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "ModuleStore.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParserEvaluationContext.h"
#include "Signatures.h"

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/IRValues.h"
#include "Mojo/MojoParser/ModuleLoader.h"
#include "Mojo/MojoParser/SharedState.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/InitAllDialects.h"

#include "Support/Buffer.h"
#include "Support/Compiler/OperationUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

#define DEBUG_TYPE "mojo-parser"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

static void adjustTokenEndPoint(SharedState &shared, SMLoc &loc);

//===----------------------------------------------------------------------===//
// BytecodeResolutionReferenceWalker
//===----------------------------------------------------------------------===//

namespace {
/// This class defines an attribute and type walker that resolves references to
/// decls defined within bytecode files.
class BytecodeResolutionReferenceWalker {
public:
  BytecodeResolutionReferenceWalker(SharedState &shared) : shared(shared) {}

  /// Set and save the context location for the current bytecode resolution.
  llvm::SaveAndRestore<SMLoc> saveResolutionContextLoc(SMLoc loc) {
    return llvm::SaveAndRestore<SMLoc>(this->resolutionContextLoc, loc);
  }

  /// Walk the given attribute or type element, resolving references found
  /// within.
  template <typename T>
  WalkResult walk(T element) {
    const void *key = element.getAsOpaquePointer();

    // Check if we've already walk this element before.
    auto it = visitedAttrTypes.find(key);
    if (it != visitedAttrTypes.end())
      return it->second;

    // Walk this element, bailing if skipped or interrupted.
    WalkResult walkResult = processBytecodeReferences(element);
    if (walkResult.wasInterrupted()) {
      // Don't cache failures: they can be transient (e.g., a symbol lookup
      // that fires before its container's body is materialized). Caching
      // would permanently suppress any retry once the container is ready.

      return WalkResult::interrupt();
    }
    if (walkResult.wasSkipped())
      return WalkResult::advance();

    // Walk the sub-elements, checking for bytecode references.
    WalkResult result = WalkResult::advance();
    auto walkFn = [&](auto element) {
      if (element && !result.wasInterrupted())
        result = walk(element);
    };
    element.walkImmediateSubElements(walkFn, walkFn);

    if (!result.wasInterrupted())
      visitedAttrTypes.try_emplace(key, WalkResult::advance());
    return result.wasInterrupted() ? result : WalkResult::advance();
  }
  void walkRange(TypeRange types) {
    for (Type type : types)
      walk(type);
  }

  LogicalResult resolveBytecodeSymbolSignature(SymbolRefAttr symbol,
                                               SMLoc loc) {
    auto savedContextLoc = saveResolutionContextLoc(loc);
    if (!resolveBytecodeReferenceSignature(shared, symbol))
      return failure();
    return success();
  }

  /// Recursively walks the same sub-elements as `walk`, but resolves referenced
  /// functions only through their signatures.
  template <typename T>
  WalkResult walkSignaturesOnly(T element) {
    WalkResult result;
    if constexpr (std::is_same_v<T, Type>)
      result = processBytecodeReferences(element);
    else
      result = processBytecodeSignatureReferences(element);
    if (result.wasInterrupted())
      return result;

    auto walkFn = [&](auto subElement) {
      if (subElement && !result.wasInterrupted())
        result = walkSignaturesOnly(subElement);
    };
    element.walkImmediateSubElements(walkFn, walkFn);
    return result.wasInterrupted() ? result : WalkResult::advance();
  }

private:
  /// Given a symbol reference, fully resolve the parents of the symbol assuming
  /// that the parent references do not contain any mangling.
  ASTDecl *resolveRefParentDecl(SharedState &shared, SymbolRefAttr symbol) {
    // This is a reference to a top-level declaration.
    if (symbol.getNestedReferences().empty())
      return &shared.getTopLevelDecl();

    StringAttr rootAttr = symbol.getRootReference();
    auto nestedRefs = symbol.getNestedReferences().drop_back();
    auto it = resolvedSymbolParents.find({rootAttr, nestedRefs});
    if (it != resolvedSymbolParents.end())
      return it->second;

    // Resolve the top-level container for the reference. This should be a
    // package or module. Symbol roots are single names, never dotted paths
    // (nesting is expressed with nested references), so any periods belong to
    // the name itself: a REPL/LSP wrapper buffer or a dotted module name.
    ASTDecl *decl =
        &shared.importModule({rootAttr.getValue()}, /*currentPackage=*/nullptr,
                             resolutionContextLoc);
    if (decl->isErroneous() ||
        failed(shared.declResolver->resolveBody(*decl, resolutionContextLoc)))
      return {};
    for (FlatSymbolRefAttr name : nestedRefs) {
      if (!(decl = shared.lookupAndResolveMangledDecl(
                name.getAttr(), resolutionContextLoc, *decl,
                DeclResolvedness::body)))
        return {};
    }
    resolvedSymbolParents.try_emplace({rootAttr, nestedRefs}, decl);
    return decl;
  }

  /// Resolve the reference to a bytecode decl represented by the given symbol.
  ASTDecl *resolveBytecodeReferenceSignature(SharedState &shared,
                                             SymbolRefAttr symbol) {
    ASTDecl *moduleDecl = resolveRefParentDecl(shared, symbol);
    if (!moduleDecl)
      return nullptr;
    return shared.lookupAndResolveMangledDecl(symbol.getLeafReference(),
                                              resolutionContextLoc, *moduleDecl,
                                              DeclResolvedness::signature);
  }

  /// Process the given attributes and types for bytecode references.
  WalkResult processBytecodeReferences(Attribute attr) {
    return TypeSwitch<Attribute, WalkResult>(attr)
        .Case<SymbolConstantAttr, FuncSymbolAttr>([&](auto ref) {
          ASTDecl *decl =
              resolveBytecodeReferenceSignature(shared, ref.getSymbol());
          if (!decl)
            return failure();

          // Don't fully resolve containers, they'll get resolved if something
          // is needed from within them.
          if (isa_and_nonnull<FileModuleOp, PackageOp, StructDeclOp,
                              TraitDeclOp>(decl->getIfOperation()))
            return mlir::success();

          // Fully resolve every other decl.
          return shared.declResolver->resolveBody(*decl, resolutionContextLoc);
        })
        .Default(WalkResult::advance());
  }
  WalkResult processBytecodeReferences(Type type) {
    return TypeSwitch<Type, WalkResult>(type)
        .Case<StructMetaType, LIT::StructType>([&](auto ref) {
          return success(
              resolveBytecodeReferenceSignature(shared, ref.getSymbol()));
        })
        .Case<TraitType>([&](TraitType ref) {
          return success(llvm::all_of(
              ref.getSymbols(), [&](TraitSymbolAttr symbol) -> bool {
                return resolveBytecodeReferenceSignature(shared,
                                                         symbol.getSymbol());
              }));
        })
        .Default(WalkResult::advance());
  }

  WalkResult processBytecodeSignatureReferences(Attribute attr) {
    return TypeSwitch<Attribute, WalkResult>(attr)
        .Case<SymbolConstantAttr, FuncSymbolAttr>([&](auto ref) {
          return success(
              resolveBytecodeReferenceSignature(shared, ref.getSymbol()));
        })
        .Default(WalkResult::advance());
  }

  /// The parent shared state.
  SharedState &shared;
  /// A mapping from the parent reference of a SymbolRefAttr to the
  /// corresponding resolved ASTDecl.
  DenseMap<std::pair<Attribute, ArrayRef<FlatSymbolRefAttr>>, ASTDecl *>
      resolvedSymbolParents;
  /// The current bytecode resolution context location.
  SMLoc resolutionContextLoc;
  /// The set of cached attributes/types from which nested references have
  /// already been either successfully or erroneously resolved.
  DenseMap<const void *, WalkResult> visitedAttrTypes;
};
} // namespace

//===----------------------------------------------------------------------===//
// SharedState
//===----------------------------------------------------------------------===//

struct SharedState::Impl {
  Impl(SharedState &shared)
      : sourceNames(shared),
        bytecodeParserContext(shared.getContext(), /*verifyAfterParse=*/false),
        bytecodeRefResolutionWalker(shared), evaluationContext(shared),
        collector(collectorCache) {}
  virtual ~Impl() = default;

  /// This MLIR block is owned by SharedState, and vended to clients that have a
  /// need to build Arguments that are potentially unused.  This happens during
  /// function signature type checking, where the arguments are needed to
  /// satisfy lookup requests later in the signature, but where the body may not
  /// actually be generated.  If generated, the arguments are removed from this
  /// block and installed in the actual function.
  Block argumentOwningBlock;

  SymbolTableCollection symbolTables;

  /// Source name collector.
  SourceNames sourceNames;

  /// A map of symbol tables to unique counters for names within those
  /// symbol tables.
  DenseMap<std::pair<SymbolTable *, StringAttr>, unsigned> symbolTableCounters;

  /// The top-level decl containing everything being parsed.
  ASTDecl *topLevelDecl = nullptr;

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType typeCheckErrorType;
  /// This is the decl for the builtin 'lit.none' type/attr.
  ASTType noneType;
  NoneAttr noneAttr;

  /// A list of included files used when importing modules. These are used to
  /// generate dependency files.
  SmallVector<std::string> includedFiles;

  /// The set of pre-existing source buffers within the source manager, used if
  /// importing a module whose file is already in the source manager.
  DenseMap<StringRef, int> existingSourceMgrBuffers;

  /// Flag indicating if the deps of a module are currently being resolved.
  bool activelyResolvingModuleDeps = false;

  /// Flag indicating if we should diagnose missing doc strings while parsing.
  bool diagnoseMissingDocStrings = false;

  /// This keeps track of body decorators for a given declaration, this is
  /// logically part of ASTDecl, but is stored out of line to reduce its size
  /// since these are uncommon.
  DenseMap<const ASTDecl *, std::vector<ExprNode *>> bodyDecorators;

  /// The implicit builtin imports added to each module.
  SmallVector<ImportPathAttr> implicitBuiltinImports;

  /// The decl corresponding to the standard library package.
  ModuleState *stdPackageState = nullptr;

  /// The parser configuration used when loading bytecode.
  mlir::ParserConfig bytecodeParserContext;

  /// Closure traits have a unique generator type and are global to the module.
  /// Cache previously built traits.
  DenseMap<GeneratorType, ASTDecl *> closureTraits;

  /// The decl corresponding to the universal parametric closure trait.
  ASTDecl *parametricClosureTrait = nullptr;

  /// Stateless closure extension structs, keyed by the (source trait,
  /// target trait) operation pair and the owning file module.
  DenseMap<std::pair<std::pair<Operation *, Operation *>, ASTDecl *>, ASTDecl *>
      closureExtensions;

  /// The capture values and decls associated with their enclosing nested
  /// function. This data structure is populated during the parsing of the FnOp
  /// the key ASTDecl wraps.
  DenseMap<ASTDecl *, llvm::MapVector<StringRef, Capture>> capturesInScope;
  DenseMap<ASTDecl *, CaptureConvention> captureConventionForScope;
  DenseMap<Operation *, ClosureParamCaptures> closureParamCaptures;

  /// Function type conversion thunks in each module.
  // The key is an ArrayAttr containing two elements:
  // - The "actual" signature; the type of the underlying function that the
  //   thunk is calling.
  // - The thunk signature, not including the `callee` input parameter (for some
  //   reason).
  //   This is NOT the expected/destination type we're converting to, it's the
  //   actual thunk's signature (this is so generateConversionThunk can know the
  //   "clarifying parameters", see TAPCPTTT).
  DenseMap<Attribute, FnOp> conversionThunks;

  /// This caches non-trivial implicit convertibility checks from one type to
  /// another.
  DenseMap<std::pair<Type, Type>, bool> cachedImplicitConvertibility;

  /// This memoizes assumption-free nominal trait-conformance checks
  /// (ASTDecl::doesNominalTypeConformTo), keyed by (type decl, required trait,
  /// concrete type). Only definitive yes/no results are stored. The type decl
  /// is part of the key (not just the concrete type) because the result also
  /// depends on the struct's extensions, which are found via that decl. The raw
  /// `const ASTDecl *` is a stable identity key: ASTDecls are bump-allocated in
  /// `persistentAllocator` and never freed or address-recycled within a
  /// SharedState, so an entry can never be misattributed to a different decl.
  DenseMap<std::tuple<const ASTDecl *, Type, Type>, bool>
      nominalConformanceCache;

  /// Caches builtin/std lookups by fully-qualified name. Traits and type decls
  /// are kept in separate maps: a name is either a trait or a type within a
  /// module, so both maps only ever hold their own kind, and a name looked up
  /// as both (e.g. a prelude type probed via lookupBuiltinTrait) can never read
  /// the wrong kind out of a shared key. Only positive results are stored.
  /// StringMap owns its keys, so the hot lookup path can key on a transient
  /// stack-built string with no allocation or attribute interning.
  llvm::StringMap<ASTDecl *> builtinTraitCache;
  llvm::StringMap<ASTDecl *> builtinTypeDeclCache;

  /// An attribute walker used to resolve bytecode references.
  BytecodeResolutionReferenceWalker bytecodeRefResolutionWalker;

  /// An evaluation context used to simplify attributes during parsing.
  ParserEvaluationContext evaluationContext;

  // Cache for looking up information about parameter references in types.
  ParameterCollector::Analysis collectorCache;
  ParameterCollector collector;

  /// Synthetic REPL/LSP wrapper buffers, mapped to the source path of the
  /// module each one wraps. Code in a wrapper buffer is semantically part of
  /// the wrapped module (e.g. for self-import detection).
  DenseMap<unsigned, std::string> wrapperBuffers;
};

/// Ensure `stripFilePrefix` is an absolute path ending in a separator.
static std::string canonicalizeFileCompilationDir(StringRef stripFilePrefix) {
  if (stripFilePrefix.empty())
    return {};

  SmallString<256> workingFileCompilationDir = stripFilePrefix;
  llvm::sys::path::remove_dots(workingFileCompilationDir,
                               /*remove_dot_dot=*/true);
  llvm::sys::fs::make_absolute(workingFileCompilationDir);
  if (!llvm::sys::path::is_separator(workingFileCompilationDir.back()))
    workingFileCompilationDir.append(llvm::sys::path::get_separator());
  return workingFileCompilationDir.str().str();
}

SharedState::SharedState(llvm::SourceMgr &sourceMgr, ParserConfig &config)
    : diags(sourceMgr, config.context, config.useMLIRDiagnostics,
            config.maxNotesPerDiagnostic,
            canonicalizeFileCompilationDir(config.stripFilePrefix),
            /*disableWarnings=*/config.options.disableWarnings,
            /*warningsAsErrors=*/config.options.warningsAsErrors,
            /*extraContext=*/this,
            /*autoFixItHandler=*/config.autoFixItHandler),
      options(config.options),
      declResolver(std::make_unique<DeclResolver>(*this)),
      moduleLoader(std::make_unique<ModuleLoader>(*this)),
      parserListener(config.parserListener),
      extensionsScopeMarker(StringAttr::get(config.context, "extension:")),
      disablePrebuiltPackages(config.disablePrebuiltPackages),
      useBuiltinModule(config.useBuiltinModule),
      impl(std::make_unique<Impl>(*this)) {
  impl->diagnoseMissingDocStrings = config.diagnoseMissingDocStrings;
  docsBasePath = config.docsBasePath;

  preloadAllKGENDialects(config.context);

  // Record any existing buffers in the source manager.
  for (int i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    int bufferId = i + 1;
    impl->existingSourceMgrBuffers.try_emplace(
        sourceMgr.getMemoryBuffer(bufferId)->getBufferIdentifier(), bufferId);
  }

  // Tell the diagnostics machinery how to find the end of a token lazily when
  // it needs it.
  diags.setTokenEndPointAdjustmentFn(
      [this](SMLoc &loc) { adjustTokenEndPoint(*this, loc); });

  if (options.getDebugInfoLevelForInput() > CompilationOptions::kSynthetic) {
    diBuilder = std::make_unique<DebugInfo::DIBuilder>(config.context);

    diBuilder->initializeCompileUnit(
        options.debugInfoLanguage,
        diBuilder->createFile(diags.getBufferNameIdentifier()), "Mojo",
        /*isOptimized=*/options.optimizationLevel > 0,
        options.getDIEmissionKind());
  }
  closureEmitter = std::make_unique<ClosureEmitter>(*this);
}

SharedState::~SharedState() { declResolver.reset(); }

bool SharedState::shouldDiagnoseMissingDocStrings() const {
  return impl->diagnoseMissingDocStrings;
}

void SharedState::initialize(ASTDecl &topLevelDecl) {
  assert(!impl->topLevelDecl && "already initialized");
  impl->topLevelDecl = &topLevelDecl;
  getModuleLoader().initializeTopLevel(topLevelDecl);

  // Build the builtins decl.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  ASTDecl &builtinsDecl = declResolver->addDecl(
      topLevelDecl.getIfOperation(), topLevelDecl.getLoc(), StringAttr(),
      nullptr, topLevelDecl.getCursor(), topLevelDecl.getCursor(), -1);
  addBuiltinTypes(builtinsDecl);
  builtinsDecl.resolvedness = DeclResolvedness::body;

  // The outermost scope contains all of the __builtins__ function definitions.
  for (auto &[name, decls] : builtinsDecl.getDeclsInScope())
    declResolver->aliasDecls(decls, name, topLevelDecl.getLoc(), topLevelDecl);

  // Top level is fully resolved now.
  topLevelDecl.resolvedness = DeclResolvedness::body;
}

/// Shared state maintains an MLIR Block and deallocates it when the parser is
/// torn down.  This can be used to allocate BlockArgument's that may or may
/// not get used in the future.
Block &SharedState::getArgumentOwningBlock() {
  return impl->argumentOwningBlock;
}

void SharedState::collectParamRefsInType(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &uses) {
  bool hasConstExpr = false;         // ignored
  size_t requiredSignatureDepth = 0; // ignored
  impl->collector.collectUsesFromType(type, uses, hasConstExpr,
                                      requiredSignatureDepth);
}

void SharedState::deleteDecl(ASTDecl &decl) {
  if (!decl.getUserNameIfOperation())
    return;
  Operation *op = decl.getIfOperation();

  // Remove from global maps.
  // Func needs a special case since it may or may not be a symbol.
  if (auto func = dyn_cast<FnOp>(op)) {
    if (SymbolRefAttr sym = decl.getSymbolRef())
      declResolver->declForFuncSymbol.erase(sym);
    impl->sourceNames.forgetSourceName(func);
  } else if (auto symbolDecl = dyn_cast<mlir::SymbolOpInterface>(op)) {
    if (SymbolRefAttr sym = decl.getSymbolRef())
      declResolver->declForTypeSymbol.erase(sym);
    impl->sourceNames.forgetSourceName(symbolDecl);
    impl->symbolTables.getSymbolTable(op->getParentOp()).remove(op);
  }
  op->erase();

  // Set the IRValue to nullptr, so that any reference pointing to the decl can
  // check if it's valid.
  decl.setIRValue(nullptr);
}

ASTDecl &SharedState::getTopLevelDecl() { return *impl->topLevelDecl; }

MojoInflightDiag SharedState::emitError(Location loc, const Twine &message) {
  return diags.emitError(loc, message);
}

/// Emit an error through the parser's logic.
MojoInflightDiag SharedState::emitError(llvm::SMLoc loc, const Twine &message) {
  return diags.emitError(loc, message);
}

/// Emit a warning.
MojoInflightDiag SharedState::emitWarning(Location loc, const Twine &message) {
  return diags.emitWarning(loc, message);
}
MojoInflightDiag SharedState::emitWarning(llvm::SMLoc loc,
                                          const Twine &message) {
  return diags.emitWarning(loc, message);
}

/// Inflate a lightweight SMLoc into an MLIR Location object for addition
/// into the IR.
Location SharedState::translateLocation(llvm::SMLoc loc) const {
  auto fileLoc = diags.translateLocation(loc);
  return diBuilder ? diBuilder->createScopedLoc(fileLoc) : fileLoc;
}

FileLineColLoc SharedState::createLocation(StringRef filename, unsigned line,
                                           unsigned column) {
  return FileLineColLoc::get(getContext(), diags.getCanonicalFilename(filename),
                             line, column);
}

ASTType SharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorType;
}
ASTType SharedState::getNoneType() const { return impl->noneType; }
NoneAttr SharedState::getNoneAttr() const { return impl->noneAttr; }

/// Add declarations for magic things to the builtins decl.
void SharedState::addBuiltinTypes(ASTDecl &builtinsDecl) {
  DeclResolver &resolver = *declResolver;
  MLIRContext *context = getContext();

  // Add a declarations for builtin types.
  impl->noneType = KGEN::NoneType::get(context);
  impl->noneAttr = NoneAttr::get(context);

  // Make the type check error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  // MLIR types are parsed with a non_struct_type metadata.
  auto anyNonStructTypeType = NonStructTypeType::get(getContext());
  auto addMagicMLIRDecl = [&](StringRef name, Type magicType) {
    TypedAttr value = TypeParamAttr::get(magicType, anyNonStructTypeType);
    resolver.addFullyResolvedDecl(PValue(value), name, builtinsDecl.getLoc(),
                                  &builtinsDecl);
  };

  addMagicMLIRDecl("__mlir_attr", MagicMLIRAttrType::get(context));
  addMagicMLIRDecl("__mlir_deferred_attr",
                   MagicMLIRDeferredAttrType::get(context));
  addMagicMLIRDecl("__mlir_deferred_type",
                   MagicMLIRDeferredTypeType::get(context));
  addMagicMLIRDecl("__mlir_op", MagicMLIROpType::get(context));
  addMagicMLIRDecl("__mlir_type", MagicMLIRTypeType::get(context));
}

Operation *
SharedState::uniquifyNameAndAddToParentSymbolTable(Operation *declOp) {
  assert(declOp && "Cannot set a symbol for non-operation decl");

  // We look up the symbol in the enclosing symbol table.  For example, for a
  // method in a struct, we use the struct as the symbol table.  For atop-level
  // function we use the global module.
  Operation *parentSymbolTableOp =
      SymbolTable::getNearestSymbolTable(declOp->getParentOp());
  SymbolTable &symTab = impl->symbolTables.getSymbolTable(parentSymbolTableOp);

  // Insert the operation into the symbol table and see if it got renamed.
  // Restore the original position of the operation after.
  Block *prevBlock = declOp->getBlock();
  Block::iterator prevPos = std::next(declOp->getIterator());
  declOp->remove();
  auto resetPos =
      llvm::scope_exit([&] { declOp->moveBefore(prevBlock, prevPos); });

  StringAttr origName = SymbolTable::getSymbolName(declOp);
  Operation *existingOp = symTab.lookup(origName);
  if (existingOp && existingOp != declOp) {
    unsigned &counter = impl->symbolTableCounters[{&symTab, origName}];
    SymbolTable::setSymbolName(
        declOp, getUniqueSymbolName(origName.str(), symTab, counter));
  } else {
    existingOp = nullptr;
  }

  [[maybe_unused]] auto newName = symTab.insert(declOp);
  assert(newName == SymbolTable::getSymbolName(declOp) &&
         "symbol table insertion changed the name");
  return existingOp;
}

Operation *SharedState::lookupSymbolIn(ASTDecl *container, StringAttr name) {
  Operation *tableOp = container->getIfOperation();
  assert(tableOp && "decl is not an operation");
  return impl->symbolTables.getSymbolTable(tableOp).lookup(name);
}

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

/// Return any decorators that need to be processed as part of body resolution
/// phase for a decl.
ArrayRef<ExprNode *> ASTDecl::getBodyDecorators() const {
  if (!hasBodyDecorators)
    return {};
  return shared.getImpl().bodyDecorators[this];
}

/// During signature resolution, this is called with any decorators that need
/// to persist until body resolution.
void ASTDecl::setBodyDecorators(ArrayRef<ExprNode *> decorators) {
  if (decorators.empty())
    return;

  shared.getImpl().bodyDecorators.insert({this, decorators.vec()});
  hasBodyDecorators = true;
}

//===----------------------------------------------------------------------===//
// Name Lookup
//===----------------------------------------------------------------------===//

/// Return true if the specified type has a declared member with the specified
/// name.
bool SharedState::typeHasMember(ASTType type, StringRef name, llvm::SMLoc loc) {
  ASTDecl *typeDecl = type.getDecl(*this);
  if (!typeDecl) // MLIR types have no methods.
    return false;
  return typeHasMember(*typeDecl, name, loc);
}

bool SharedState::typeHasMember(ASTDecl &typeDecl, StringRef name,
                                llvm::SMLoc loc) {
  return lookupAndResolveDecl(name, loc, typeDecl,
                              /*searchParentScopes=*/false)
      .isSuccess();
}

/// Perform a name lookup in the specified scope and return the named
/// declaration as a LookupResult.
auto SharedState::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                       ASTDecl &scope, bool searchParentScopes,
                                       bool resolveTarget) -> LookupResult {

  // Ensure the context is fully resolved, so all its members are known.  It
  // would be bad to look something up in a scope without all members known.
  if (failed(declResolver->resolveBody(scope, loc)))
    return LookupResult::getErroneous();

  auto nameAttr = StringAttr::get(getContext(), name);

  // Look up the name.
  auto lookupInScope = [&](ASTDecl &scope) -> ArrayRef<ASTDecl *> {
    // Check if we already have a declaration for this name in the current
    // scope.
    auto result = scope.lookupInCurrentScope(nameAttr);
    if (!result.empty())
      return result;

    // If the lookup failed, try to resolve any wildcard imports in the scope.
    // We don't know if these imports will actually provide the decl we are
    // looking for, so we have to try until we find one that does.
    declResolver->expandWildcardsForName(scope, nameAttr,
                                         /*stopOnFirstHit=*/true);
    return scope.lookupInCurrentScope(nameAttr);
  };

  auto getEntry = [&]() -> LookupResult {
    // A package's own scope is empty: its public surface is its __init__'s
    // scope. Redirect any lookup *rooted at* a package to __init__, so
    // component access and re-export resolution see the package's symbols.
    if (isa_and_nonnull<PackageOp>(scope.getIfOperation())) {
      FailureOr<ASTDecl *> initOrFailure =
          declResolver->bodyResolvePackageInit(scope, loc);
      if (failed(initOrFailure))
        return LookupResult::getErroneous();
      if (ASTDecl *initDecl = *initOrFailure)
        return lookupAndResolveDecl(name, loc, *initDecl, searchParentScopes,
                                    resolveTarget);
      return LookupResult::getFailure({});
    }
    if (!searchParentScopes) {
      ArrayRef<ASTDecl *> result = lookupInScope(scope);
      if (!result.empty())
        return LookupResult::getSuccess(result);
      return LookupResult::getFailure({});
    }
    ArrayRef<ASTDecl *> skipped = {};
    ASTDecl *curSearchScope = &scope;
    do {
      ArrayRef<ASTDecl *> e = lookupInScope(*curSearchScope);
      if (!e.empty()) {
        if (curSearchScope &&
            isa_and_nonnull<StructDeclOp, TraitDeclOp>(
                curSearchScope->getIfOperation()) &&
            !(*e.front()).getIfIRValue().getIfPValue()) {
          // Skip struct/trait bodies when searching up parent scopes, unless
          // the value is a parameter.
          if (skipped.empty())
            skipped = e;

          continue;
        }
        return LookupResult::getSuccess(e);
      }
    } while ((curSearchScope = curSearchScope->parentDecl));
    // If we found a name in a context that we skip, return it in the failure
    // for diagnostic reporting.
    return LookupResult::getFailure(skipped);
  };

  LookupResult entry = getEntry();

  // If nothing was found, return a failure.
  if (entry.isFailure())
    return entry;
  SmallVector<ASTDecl *> resultDecls(entry.getIfSuccess());

  // If the lookup succeeded, make sure the signature for the referenced decls
  // are understood. Make a copy of the entries to avoid dangling references if
  // we end up invalidating the decl map.
  bool wasUnresolvedImport = !resultDecls.empty() && resultDecls.front() &&
                             isa_and_nonnull<UnresolvedImportOp>(
                                 resultDecls.front()->getIfOperation());
  for (ASTDecl *decl : resultDecls) {
    // Always resolve UnresolvedImportOps since that's the point of this whole
    // function. But only resolve the ultimate declaration it's pointing to if
    // resolveTarget is true.
    // TODO(MOCO-522): Arcana docs on imports!
    bool shouldResolve = resolveTarget || isa_and_nonnull<UnresolvedImportOp>(
                                              decl->getIfOperation());
    if (shouldResolve) {
      if (failed(
              declResolver->resolve(*decl, DeclResolvedness::signature, loc))) {
        // If the decl was erroneous somehow, then don't form a reference to it,
        // the error has already been diagnosed.
        return LookupResult::getErroneous();
      }
    }
  }
  // Get again the entry pointer since it might have been invalidated by
  // declResolver->resolve above.
  entry = getEntry();
  // If we are resolving an unresolved import, do another lookup now that import
  // has been resolved. The scope map should be updated with the proper decls.
  if (entry.isSuccess() && wasUnresolvedImport)
    return lookupAndResolveDecl(name, loc, scope, searchParentScopes,
                                resolveTarget);

  // We return a pointer into the TinyPtrVector entry in the scope.  This should
  // be stable because you can't perform a lookup into a decl that has unknown
  // entries, and we just resolved all the signatures for all the decls.
  return entry;
}

/// Perform a name lookup for a member in the specified type.
auto SharedState::lookupAndResolveDecl(StringRef name, SMLoc loc, ASTType scope,
                                       bool searchParentScopes,
                                       bool resolveTarget) -> LookupResult {
  if (auto *decl = scope.getDecl(*this))
    return lookupAndResolveDecl(name, loc, *decl, searchParentScopes,
                                resolveTarget);
  return LookupResult::getFailure({});
}

/// Perform a name lookup that collects ALL matching declarations instead of
/// stopping at the first non-import match.
auto SharedState::lookupAllDeclsWithName(StringRef name, SMLoc loc,
                                         ASTDecl &scope, bool resolve)
    -> LookupAllResult {

  auto nameAttr = StringAttr::get(getContext(), name);

  // Collect all matching declarations from all scopes.
  std::vector<ASTDecl *> allDecls;
  std::vector<ASTDecl *> skippedDecls;

  // Lambda to expand any wildcard imports that might match the desired name.
  auto expandWildcardImports = [&](ASTDecl &searchScope) {
    // If the lookup failed, try to resolve any wildcard imports in the scope.
    // We don't know if these imports will actually provide the decl we are
    // looking for, so we have to try until we find one that does.
    declResolver->expandWildcardsForName(searchScope, nameAttr);
  };

  auto collectFromAllScopes = [&]() -> LookupAllResult {
    ASTDecl *curSearchScope = &scope;
    do {
      expandWildcardImports(*curSearchScope);
      ArrayRef<ASTDecl *> e = curSearchScope->lookupInCurrentScope(nameAttr);
      if (!e.empty()) {
        if (curSearchScope &&
            isa_and_nonnull<StructDeclOp>(curSearchScope->getIfOperation()) &&
            !(*e.front()).getIfIRValue().getIfPValue()) {
          // Skip struct bodies when searching up parent scopes, unless the
          // value is a parameter. But still collect them for potential use.
          for (ASTDecl *decl : e) {
            skippedDecls.push_back(decl);
          }
          continue;
        }
        // Add all declarations from this scope to our collection.
        for (ASTDecl *decl : e)
          allDecls.push_back(decl);
      }
    } while ((curSearchScope = curSearchScope->parentDecl));

    // If we found declarations, return success.
    if (!allDecls.empty())
      return LookupAllResult::getSuccess(std::move(allDecls));

    // If we found skipped declarations but no regular ones, return them as
    // failure.
    if (!skippedDecls.empty())
      return LookupAllResult::getFailure(std::move(skippedDecls));

    return LookupAllResult::getFailure({});
  };

  LookupAllResult entry = collectFromAllScopes();

  // If nothing was found, return a failure.
  if (entry.isFailure() || entry.isErroneous())
    return entry;

  // If the lookup succeeded, make sure the signature for the referenced decls
  // are understood. Make a copy of the entries to avoid dangling references if
  // we end up invalidating the decl map.
  SmallVector<ASTDecl *> resultDecls(entry.getIfSuccess());

  bool resolvedImport = false;
  for (ASTDecl *decl : resultDecls) {
    if (auto importOp =
            dyn_cast_or_null<UnresolvedImportOp>(decl->getIfOperation())) {
      DeclResolvedness resolvedness = decl->resolvedness;
      if (resolvedness < DeclResolvedness::signature) {
        resolvedImport = true;
        if (failed(declResolver->resolveSignature(importOp, *decl,
                                                  /*resolveTarget=*/resolve))) {
          return LookupAllResult::getErroneous();
        }
      }
    }
  }

  // If we resolved an import, we need to do the lookup again because the
  // import resolution may have changed the scope contents.
  if (resolvedImport) {
    entry = collectFromAllScopes();
    if (entry.isSuccess())
      return lookupAllDeclsWithName(name, loc, scope, resolve);
  }

  // Return the collected declarations.
  return entry;
}

PackageOp SharedState::getPrecompiledStdlibPackage() {
  ModuleState *stdState = impl->stdPackageState;
  if (!stdState || !stdState->origin || !stdState->origin->bytecodeReader)
    return {};
  return dyn_cast_or_null<PackageOp>(stdState->decl->getIfOperation());
}

void SharedState::materializePrecompiledStdlibOp(Operation *op) {
  ModuleState *stdState = impl->stdPackageState;
  if (!stdState || !stdState->origin || !stdState->origin->bytecodeReader)
    return;
  mlir::BytecodeReader &reader = *stdState->origin->bytecodeReader;
  // Best-effort: a failed materialization just means this subtree contributes
  // no import suggestion. The sole caller runs on the error path of an
  // already-failing compile, so we never disturb the in-progress diagnostic.
  if (reader.isMaterializable(op))
    (void)reader.materialize(op);
}

ASTDecl &SharedState::importModule(const ImportPath &path,
                                   PackageOp currentPackage, llvm::SMLoc loc) {
  return getModuleLoader().importModule(path, currentPackage, loc);
}

SmallVector<ASTDecl *>
SharedState::getNestedModuleDecls(PackageOp packageOp) const {
  return getModuleLoader().getNestedModuleDecls(packageOp);
}

bool SharedState::hasNestedModule(PackageOp packageOp, StringRef name) const {
  return getModuleLoader().hasNestedModule(packageOp, name);
}

const llvm::MemoryBuffer *SharedState::openModuleFile(StringRef path,
                                                      llvm::SMLoc loc) {
  // Reuse an already-open buffer if we have one.
  unsigned fileID = impl->existingSourceMgrBuffers.lookup(path);
  if (!fileID) {
    std::string fullPath;
    fileID = getSourceMgr().AddIncludeFile(path.str(), loc, fullPath);
    if (!fileID)
      return nullptr;
    impl->includedFiles.push_back(fullPath);
  }
  return getSourceMgr().getMemoryBuffer(fileID);
}

ASTDecl *SharedState::tryImportSubModule(ASTDecl &parent, StringRef name,
                                         llvm::SMLoc loc) {
  return getModuleLoader().tryImportSubModule(parent, name, loc);
}

void SharedState::registerWrapperBuffer(unsigned bufferId,
                                        StringRef wrappedSourcePath) {
  impl->wrapperBuffers[bufferId] = wrappedSourcePath;
}

std::optional<StringRef>
SharedState::getWrappedSourcePath(unsigned bufferId) const {
  auto it = impl->wrapperBuffers.find(bufferId);
  return it == impl->wrapperBuffers.end()
             ? std::optional<StringRef>(std::nullopt)
             : it->second;
}

bool SharedState::hasBuiltinModule() const { return useBuiltinModule; }

/// Builds the fully-qualified cache key for a builtin/std lookup into a stack
/// buffer, so the hot lookup path allocates nothing and interns nothing. The
/// trait and type-decl caches share this format so their keys stay in sync.
static SmallString<64> makeBuiltinCacheKey(const SharedState::ImportPath &path,
                                           StringRef name) {
  SmallString<64> key;
  key.append(path.relativeLevel, '.');
  for (StringRef component : path.components) {
    key += component;
    key += '.';
  }
  key += name;
  return key;
}

/// Lookup a builtin trait like `AnyType`, `Deinitable`, `Copyable`,
/// `Movable` etc.  On error this returns null but does not print an error.
ASTDecl *SharedState::lookupBuiltinTrait(StringRef traitName, SMLoc loc) {
  if (LLVM_UNLIKELY(!hasBuiltinModule())) {
    // I don't even know why we are allowing this
    return {};
  }

  const auto path = ImportPath({"std", "prelude"});
  const SmallString<64> cacheKey = makeBuiltinCacheKey(path, traitName);
  if (auto it = impl->builtinTraitCache.find(cacheKey);
      it != impl->builtinTraitCache.end())
    return it->second;

  LookupResult lookup = lookupAndResolveDecl(
      traitName, loc, importModule(path, /*currentPackage=*/nullptr, loc), true,
      false);
  if (!lookup.isFailure() && !lookup.getIfSuccess().empty()) {
    for (ASTDecl *result : lookup.getIfSuccess()) {
      if (isa_and_nonnull<TraitDeclOp>(result->getIfOperation())) {
        impl->builtinTraitCache.try_emplace(cacheKey, result);
        return result;
      }
    }
  }
  return nullptr;
}

TraitType SharedState::lookupBuiltinTraitType(StringRef traitName, SMLoc loc) {
  ASTDecl *traitDecl = lookupBuiltinTrait(traitName, loc);
  if (!traitDecl || !isa_and_nonnull<TraitDeclOp>(traitDecl->getIfOperation()))
    return {};
  if (failed(declResolver->resolveSignature(*traitDecl, loc)))
    return {};
  return cast<TraitDeclOp>(traitDecl->getIfOperation()).getCanonicalTrait();
}

ASTDecl *SharedState::lookupNamedTypeDecl(StringRef name, ASTDecl &context,
                                          llvm::SMLoc loc) {
  LookupResult result =
      lookupAndResolveDecl(name, loc, context, /*searchParentScopes=*/true);
  if (result.isErroneous())
    return {};
  if (result.isFailure()) {
    emitError(loc, "could not find an '") << name << "' type";
    return {};
  }
  // The overload set may contain multiple entries, but if it is a struct, it
  // must be a single entry and therefore we can just check that one.
  ASTDecl &firstDecl = *result.getIfSuccess()[0];
  if (isa_and_nonnull<StructDeclOp>(firstDecl.getIfOperation()))
    return &firstDecl;

  if (auto aliasDecl =
          dyn_cast_or_null<AliasDeclOp>(firstDecl.getIfOperation())) {
    if (LIT::isMetaType(aliasDecl.getType()))
      return &firstDecl;
  }

  auto diag = emitError(loc, "'") << name << "' doesn't resolve to a type";
  diag.attachNote(firstDecl.getLoc()) << "'" << name << "' declared here";
  return {};
}

static ASTType typeForResolvedTypeDecl(SharedState &state, ASTDecl *decl,
                                       SMLoc loc) {
  if (auto structDecl = dyn_cast_or_null<StructDeclOp>(decl->getIfOperation()))
    return structDecl.bindReference();
  if (auto aliasDecl = dyn_cast_or_null<AliasDeclOp>(decl->getIfOperation())) {
    if (failed(state.declResolver->resolveBody(*decl, loc)))
      return state.getTypeCheckErrorType();
    return aliasDecl.getValueAttr();
  }
  return state.getTypeCheckErrorType();
}

ASTType SharedState::lookupBuiltinType(StringRef name, ASTDecl &context,
                                       llvm::SMLoc loc) {
  if (LLVM_LIKELY(hasBuiltinModule()))
    return getCachedBuiltinType({"std", "prelude"}, name, loc);

  if (ASTDecl *decl = lookupNamedTypeDecl(name, context, loc))
    return typeForResolvedTypeDecl(*this, decl, loc);
  return getTypeCheckErrorType();
}

ASTDecl *SharedState::getCachedBuiltinTypeDecl(const ImportPath &path,
                                               StringRef name,
                                               llvm::SMLoc loc) {
  const SmallString<64> cacheKey = makeBuiltinCacheKey(path, name);
  if (auto it = impl->builtinTypeDeclCache.find(cacheKey);
      it != impl->builtinTypeDeclCache.end())
    return it->second;

  ASTDecl &moduleDecl = importModule(path, /*currentPackage=*/nullptr, loc);
  if (ASTDecl *decl = lookupNamedTypeDecl(name, moduleDecl, loc)) {
    impl->builtinTypeDeclCache.try_emplace(cacheKey, decl);
    return decl;
  }

  return nullptr;
}

ASTType SharedState::getCachedBuiltinType(const ImportPath &path,
                                          StringRef name, llvm::SMLoc loc) {
  if (auto *decl = getCachedBuiltinTypeDecl(path, name, loc))
    return typeForResolvedTypeDecl(*this, decl, loc);

  return getTypeCheckErrorType();
}

ASTDecl *SharedState::getBuiltinCoroutineType(llvm::SMLoc loc) {
  return getCachedBuiltinTypeDecl({"std", "builtin", "_coroutine"}, "Coroutine",
                                  loc);
}

ASTDecl *SharedState::getBuiltinDevicePassableTrait(llvm::SMLoc loc) {
  ASTDecl &devicePassableModule =
      importModule({"std", "builtin", "device_passable"},
                   /*currentPackage=*/nullptr, loc);
  LookupResult result = lookupAndResolveDecl(
      "DevicePassable", loc, devicePassableModule, /*searchParentScopes=*/true);
  if (result.isErroneous())
    return {};
  if (result.isFailure()) {
    emitError(loc, "could not find a 'DevicePassable' type");
    return {};
  }
  ASTDecl &firstDecl = *result.getIfSuccess()[0];
  return &firstDecl;
}

ASTDecl *SharedState::getBuiltinRaisingCoroutineType(llvm::SMLoc loc) {
  return getCachedBuiltinTypeDecl({"std", "builtin", "_coroutine"},
                                  "RaisingCoroutine", loc);
}

ASTType SharedState::getStandardCollectionType(llvm::SMLoc loc,
                                               StringRef name) {
  return getCachedBuiltinType({"std", "collections"}, name, loc);
}

ASTType SharedState::getBuiltinSliceType(llvm::SMLoc loc, StringRef name) {
  return getCachedBuiltinType({"std", "builtin", "builtin_slice"}, name, loc);
}

ASTType SharedState::getBuiltinStubsMLIRType(llvm::SMLoc loc) {
  return getCachedBuiltinType({"std", "builtin", "_stubs"}, "__MLIRType", loc);
}

ArrayRef<ASTDecl *>
SharedState::getBuiltinFunction(ASTDecl &context, const ImportPath &modulePath,
                                StringRef fnName, llvm::SMLoc loc) {
  ASTDecl &module = importModule(modulePath, /*currentPackage=*/nullptr, loc);
  return getBuiltinFunction(module, fnName, loc);
}

ArrayRef<ASTDecl *> SharedState::getBuiltinFunction(ASTDecl &module,
                                                    StringRef fnName,
                                                    llvm::SMLoc loc) {
  LookupResult result =
      lookupAndResolveDecl(fnName, loc, module, /*searchParentScopes=*/false);
  if (!result.isSuccess() || result.getIfSuccess().empty()) {
    emitError(loc, "internal error: could not find builtin function '")
        << fnName << "'";
    return {};
  }
  ArrayRef<ASTDecl *> decls = result.getIfSuccess();
  if (!isa_and_nonnull<FnOp>(decls.front()->getIfOperation())) {
    emitError(loc, "internal error: builtin '")
        << fnName << "' does not refer to a function";
    return {};
  }
  return decls;
}

void SharedState::importBuiltinModules(ASTDecl &moduleDecl) {
  // Check if this is the first attempt at resolving the builtin modules.
  if (impl->implicitBuiltinImports.empty()) {
    // Import the main standard library package.
    impl->stdPackageState = &getModuleLoader().importModuleState(
        {"std"}, impl->topLevelDecl, moduleDecl.getLoc(),
        /*isImplicit=*/true);
    ASTDecl *last = declResolver->getParsedDeclList().back();
    if (last && last->isErroneous()) {
      std::string stdmsg =
          "'std' is required for all normal mojo compiles.\n"
          "If you see this either:\n"
          "- Your mojo installation is broken and needs to be reinstalled or\n"
          "- You are a 'std' developer and are intentionally avoiding a "
          "pre-built 'std'";
      emitError(last->loc, stdmsg);
    }

    if (failed(declResolver->resolveBody(*impl->stdPackageState->decl,
                                         moduleDecl.getLoc())))
      return;

    // Import the prelude package.
    ASTDecl &preludePackageDecl =
        *getModuleLoader()
             .importModuleState({"std", "prelude"}, impl->topLevelDecl,
                                moduleDecl.getLoc(), /*isImplicit=*/true)
             .decl;
    if (failed(
            declResolver->resolveBody(preludePackageDecl, moduleDecl.getLoc())))
      return;

    // Implicitly wildcard-import the prelude package into every module.
    impl->implicitBuiltinImports.emplace_back(ImportPathAttr::get(
        getContext(), /*relativeLevel=*/0, {"std", "prelude"}));
  }

  // Add an ImportOp for "std" to the module's scope. This makes the bare
  // name `std` resolvable and access to child modules is gated like any import.
  // Child modules must be explicitly imported by the user or re-exported by
  // std's __init__.mojo.
  StringAttr stdAttr = StringAttr::get(getContext(), "std");
  auto &block = moduleDecl.getIfOperation()->getRegion(0).front();
  OpBuilder builder = OpBuilder::atBlockEnd(&block);
  declResolver->createImportOp(
      moduleDecl, builder, stdAttr,
      ImportPathAttr::get(getContext(), /*relativeLevel=*/0, {"std"}),
      translateLocation(moduleDecl.getLoc()));

  for (ImportPathAttr import : impl->implicitBuiltinImports) {
    moduleDecl.addUnresolvedWildcardImport(
        UnresolvedWildcardImport{import, moduleDecl.getLoc()});
  }
}

ASTDecl &SharedState::createModule(StringRef moduleName,
                                   const llvm::MemoryBuffer *moduleBuffer,
                                   FileLineColLoc loc) {
  // Create a new module state.
  ModuleSpec spec{moduleName.str(),
                  /*path=*/std::string(moduleBuffer->getBufferIdentifier()),
                  ModuleSpec::Kind::SourceModule};
  ModuleLoader &loader = getModuleLoader();
  ModuleState &state = loader.createModuleState(
      StringAttr::get(getContext(), moduleName), moduleBuffer,
      loader.getTopLevelState(), loc, spec, /*importLoc=*/{});
  return *state.decl;
}

ASTDecl &SharedState::createPackage(StringRef path, StringRef name) {
  // Note the importLoc here is empty as this is a top-level package and so
  // isn't imported from anywhere.
  ModuleSpec spec{name.str(), path.str(), ModuleSpec::Kind::SourcePackage};
  ModuleLoader &loader = getModuleLoader();
  ModuleState &state =
      loader.createPackageState(spec, loader.getTopLevelState(),
                                /*importLoc=*/{});
  return *state.decl;
}

ASTDecl &SharedState::createBinaryPackage(StringRef path, StringRef name) {
  ModuleSpec spec{name.str(), path.str(), ModuleSpec::Kind::Precompiled};
  ModuleLoader &loader = getModuleLoader();
  ModuleState &state =
      loader.createBinaryPackageState(SMLoc(), spec, loader.getTopLevelState());
  return *state.decl;
}

std::optional<std::string> SharedState::getModuleSourcePath(ASTDecl &module) {
  ModuleState *state = getModuleLoader().lookupState(&module);
  if (!state)
    return std::nullopt;
  return state->sourcePath();
}

LogicalResult SharedState::materializeDeferredModule(ASTDecl &decl, SMLoc loc) {
  // Only a deferred source module (FileModuleOp with an invalid cursor and a
  // recorded source path) needs materializing; everything else is a no-op.
  ModuleState *state = getModuleLoader().lookupState(&decl);
  if (!state || !decl.getCursor().isInvalid())
    return success();
  std::optional<std::string> sourcePath = state->sourcePath();
  if (!sourcePath)
    return success();

  const llvm::MemoryBuffer *moduleBuffer = openModuleFile(*sourcePath, loc);
  if (!moduleBuffer) {
    emitError(decl.getLoc(),
              "unable to open module file '" + *sourcePath + "'");
    return failure();
  }

  // Wire up the parse cursor so the body can now be resolved. Now that the file
  // is open, give the decl its real source location.
  Lexer lexer(diags, moduleBuffer);
  decl.setLoc(lexer.getToken().getLoc());
  decl.setParseCursor(lexer.getCursor(), LexerCursor::getEOF(moduleBuffer));

  // Auto-import the core language modules and notify the listener - the
  // module's content now exists.
  if (LLVM_LIKELY(hasBuiltinModule()))
    importBuiltinModules(decl);
  notifyListenerOnModuleDecl(decl, decl.getLoc());

  return success();
}

mlir::ParserConfig &SharedState::getBytecodeParserConfig() {
  return impl->bytecodeParserContext;
}

void SharedState::addIncludedFile(std::string path) {
  impl->includedFiles.emplace_back(std::move(path));
}

bool SharedState::tryRegisterConversionThunk(Attribute key, FnOp thunk) {
  FnOp &registeredThunk = impl->conversionThunks[key];
  if (registeredThunk)
    return false;
  registeredThunk = thunk;
  return true;
}

ASTDecl *
SharedState::lookupAndResolveMangledDecl(StringAttr leafRef, SMLoc loc,
                                         ASTDecl &container,
                                         DeclResolvedness howResolved) {
  // When a bytecode module depends on a decl parsed from source, we have to
  // resolve the signatures of all the children of the source decl, because
  // otherwise they won't be registered in the symbol table.
  if (!container.loadedFromBytecode && !container.referencedFromBytecode) {
    container.referencedFromBytecode = true;
    SmallVector<ASTDecl *> toResolve;
    for (auto &[_, children] : container.getDeclsInScope())
      llvm::append_range(toResolve, children);
    for (ASTDecl *child : toResolve)
      if (failed(declResolver->resolveSignature(*child, loc)))
        return nullptr;
  }

  // Find the operation in the symbol table of its container.
  auto declOp = lookupSymbolIn<ASTDeclInterface>(&container, leafRef);
  if (!declOp)
    return nullptr;

  // Extensions are registered in the ASTDecl name table under names like
  // "extension:MyStruct, so look up using that kind of name.
  StringAttr name;
  if (auto *extOp = dyn_cast_or_null<ExtensionDeclOp>(&declOp)) {
    StringAttr baseName = extOp->getTargetStruct().value().getLeafReference();
    std::string extensionName =
        extensionsScopeMarker.getValue().str() + baseName.getValue().str();
    name = StringAttr::get(extOp->getContext(), extensionName);
  } else {
    name = declOp.getDeclName();
  }

  // If the container is loaded from bytecode, the decl should already be
  // defined within the container decl, look it up directly. This avoids going
  // through the more complex resolution paths (which also resolve other things
  // in the container that aren't needed for this lookup).
  ArrayRef<ASTDecl *> result;
  if (container.loadedFromBytecode) {
    result = container.lookupInCurrentScope(name);
  } else {
    LookupResult lookup = lookupAndResolveDecl(name, loc, container,
                                               /*searchParentScopes=*/false);
    result = lookup.getIfSuccess();
  }

  // Find the entry that matches the full symbol name.
  for (ASTDecl *decl : result) {
    if (decl->getIfOperation() != declOp)
      continue;
    if (failed(declResolver->resolve(*decl, howResolved, loc)))
      return nullptr;
    return decl;
  }

  // The scope lookup above only surfaces a package's *public* interface (its
  // __init__, via the redirect in lookupAndResolveDecl). A mangled symbol,
  // however, can reference an *internal* submodule - one that is a child of the
  // package but unlisted (navigable only through the module-state cache, not
  // the importable scope). Resolving an IR symbol is internal access, so fall
  // back to the nested-module cache for it.
  if (ModuleState *state = getModuleLoader().lookupState(&container)) {
    auto it = state->nestedModules.find(name);
    if (it != state->nestedModules.end() && it->second->decl &&
        it->second->decl->getIfOperation() == declOp) {
      if (failed(declResolver->resolve(*it->second->decl, howResolved, loc)))
        return nullptr;
      return it->second->decl;
    }
  }

  llvm::report_fatal_error(
      "expected decl in symbol table to appear in lookup: " + name.getValue());
  return nullptr;
}

LogicalResult SharedState::resolveDeclReferencesIn(SMLoc loc, Type type) {
  auto &refWalker = getImpl().bytecodeRefResolutionWalker;
  auto savedContextLoc = refWalker.saveResolutionContextLoc(loc);
  return success(!refWalker.walk(type).wasInterrupted());
}

ASTDecl *SharedState::resolveAndGetFuncDecl(SymbolRefAttr symbol, SMLoc loc) {
  if (!symbol)
    return nullptr;
  if (ASTDecl *decl = declResolver->getDeclForFuncSymbol(symbol))
    return decl;
  // Not yet registered: trigger lazy bytecode resolution, then retry. This is
  // hit for default trait methods loaded from a bytecode package.
  if (failed(
          getImpl().bytecodeRefResolutionWalker.resolveBytecodeSymbolSignature(
              symbol, loc)))
    return nullptr;
  return declResolver->getDeclForFuncSymbol(symbol);
}

LogicalResult
SharedState::resolveDeclFromBytecode(ASTDecl &decl,
                                     DeclResolvedness resolvedness) {
  Operation *declOp = decl.getIfOperation();
  auto &refWalker = getImpl().bytecodeRefResolutionWalker;
  auto savedContextLoc = refWalker.saveResolutionContextLoc(decl.getLoc());

  // Handle resolving the signature of the decl.
  if (decl.resolvedness < DeclResolvedness::signature) {
    decl.resolvedness = DeclResolvedness::signature;

    LogicalResult result =
        llvm::TypeSwitch<Operation *, LogicalResult>(declOp)
            .Case([&](FnOp funcOp) {
              declResolver->declForFuncSymbol[decl.getSymbolRef()] = &decl;

              // Resolve the references from the signature.
              refWalker.walk(declOp->getAttrDictionary());
              return success();
            })
            .Case([&](StructDeclOp structOp) {
              // Resolve the types of any parameters.
              refWalker.walk(structOp.getParamsAttr());
              refWalker.walk(structOp.getCanonicalTrait());
              if (TypeAttr nmTarget = structOp.getNonmaterializableTargetAttr())
                refWalker.walk(nmTarget);
              return success();
            })
            .Case([&](TraitDeclOp traitOp) {
              // TODO(traits): Resolve parameter types, when they exist.
              refWalker.walk(traitOp.getCanonicalTrait());
              return success();
            })
            .Case([&](ExtensionDeclOp extensionOp) -> LogicalResult {
              SymbolRefAttr targetStruct =
                  extensionOp.getTargetStruct().value();
              assert(targetStruct && "extension doesn't have target");
              // TODO(MOCO-522): Look into storing the type in the extension
              // rather than a SymbolRefAttr.
              auto structType = LIT::StructType::get(
                  targetStruct, {},
                  TypeSignatureType::get(targetStruct.getContext()));
              refWalker.walk(StructMetaType::get(structType));
              ASTDecl &structAstDecl =
                  declResolver->getDeclForTypeSymbol(targetStruct);
              auto structOp = dyn_cast_or_null<StructDeclOp>(
                  structAstDecl.getIfOperation());
              assert(structOp && "extension target is not a struct");
              decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(structOp));
              return success();
            })
            .Case([&](ImportOp) {
              // ImportOp is already resolved — nothing to do.
              return mlir::success();
            })
            .Case([&](UnresolvedImportOp unresolvedImport) {
              // Let the normal decl resolver handling insert aliases and other
              // import behavior.
              if (failed(
                      declResolver->resolveSignature(unresolvedImport, decl)))
                return failure();
              return mlir::success();
            })
            .Case([&](AliasDeclOp aliasDecl) {
              refWalker.walk(aliasDecl.getType());
              if (TypedAttr value = aliasDecl.getValueAttr())
                refWalker.walk(value);
              return mlir::success();
            })
            .Case([&](StructFieldOp field) {
              refWalker.walk(field.getType());
              return mlir::success();
            })
            .Default([](auto) { return mlir::success(); });
    if (failed(result))
      return failure();
  }
  if (resolvedness < DeclResolvedness::body)
    return success();

  decl.resolvedness = DeclResolvedness::body;

  // Start body resolution by materializing the regions of this operation from
  // the bytecode reader. To materialize, we need to resolve the bytecode reader
  // from the parent module.
  mlir::BytecodeReader *bytecodeReader = nullptr;
  SMLoc packageImportLoc;
  ASTDecl *parentDecl = &decl;
  do {
    if (!isa_and_nonnull<FileModuleOp, PackageOp>(parentDecl->getIfOperation()))
      continue;

    // Any module inside the artifact answers, since they all share its origin.
    ModuleState *moduleState = getModuleLoader().lookupState(parentDecl);
    ModuleOrigin *origin = moduleState->origin;
    if (origin && origin->bytecodeReader) {
      bytecodeReader = &*origin->bytecodeReader;
      packageImportLoc = origin->bytecodeImportLoc;
      break;
    }
  } while ((parentDecl = parentDecl->parentDecl));
  assert(bytecodeReader && "bytecode decl doesn't have a bytecode reader");

  // Functor used to resolve references within a single operation.
  auto resolveSingleOp = [&](Operation *op) -> WalkResult {
    if (bytecodeReader->isMaterializable(op) &&
        failed(bytecodeReader->materialize(op)))
      return failure();

    for (Region &region : op->getRegions())
      for (Block &block : region)
        refWalker.walkRange(block.getArgumentTypes());
    refWalker.walkRange(op->getOperandTypes());
    refWalker.walkRange(op->getResultTypes());
    refWalker.walk(op->getAttrDictionary());
    return mlir::success();
  };

  // Conformance bodies need their witness tables materialized, but resolving a
  // witness only requires the referenced function signature. Fully resolving
  // the function body here would defeat lazy bytecode loading and recursively
  // materialize its callees.
  if (auto conformance = dyn_cast<ConformanceOp>(declOp)) {
    if (bytecodeReader->isMaterializable(conformance) &&
        failed(bytecodeReader->materialize(conformance)))
      return failure();

    refWalker.walk(conformance->getAttrDictionary());
    for (WitnessOp witness : conformance.getOps<WitnessOp>())
      if (refWalker.walkSignaturesOnly(witness.getValue()).wasInterrupted())
        return failure();
    return success();
  }

  // If this isn't a container op, we don't need to resolve any nested decls,
  // simply materialize everything nested within.
  if (!isa<FileModuleOp, PackageOp, StructDeclOp, TraitDeclOp, ExtensionDeclOp>(
          declOp)) {
    return failure(declOp->walk<mlir::WalkOrder::PreOrder>(resolveSingleOp)
                       .wasInterrupted());
  }

  // Functor to build a decl for a nested operation.
  auto addDeclForOp = [&](Operation *op, StringAttr name) -> ASTDecl & {
    return declResolver->addBytecodeDecl(op, name, &decl,
                                         DeclResolvedness::unparsed);
  };

  // If this decl is a package, this is its corresponding module state.
  ModuleState *packageState = nullptr;
  if (auto declPackage = dyn_cast<PackageOp>(declOp)) {
    packageState = getModuleLoader().lookupState(&decl);

    // Fully resolve any dependencies of the package.
    if (LinkDependencyArrayAttr deps = declPackage.getDependenciesAttr()) {
      // Each dependency is a top-level package's symbol name (see
      // mojo-precompile's buildPackage), so any periods belong to the package
      // name itself; this is never a dotted path.
      for (FlatSymbolRefAttr dep : deps) {
        ASTDecl *depDecl =
            &importModule({dep.getValue()},
                          /*currentPackage=*/nullptr, decl.getLoc());
        if (failed(declResolver->resolveBody(*depDecl, decl.getLoc())))
          return failure();
      }
    }
  }

  // Materialize the body of the decl.
  if (bytecodeReader->isMaterializable(declOp)) {
    if (failed(bytecodeReader->materialize(declOp)))
      return failure();
    // Invalidate the MLIR symbol-table cache for this op. The cache is keyed
    // by op pointer and built lazily on first lookup. If lookupSymbolIn was
    // called on this op *before* materialization (possible because
    // decl.resolvedness = body is set early, above, to break cycles), the
    // cached table was built from an empty body region and won't see the
    // newly-inflated child ops. Invalidating here forces a fresh rebuild on
    // the next lookup.
    impl->symbolTables.invalidateSymbolTable(declOp);
  }

  // Process the parsed region bodies, generating any necessary nested decls.
  SmallVector<Operation *> deferredOps;
  for (Region &region : declOp->getRegions()) {
    for (Operation &op : region.getOps()) {
      TypeSwitch<Operation *>(&op)
          .Case([&](FnOp op) { addDeclForOp(op, op.getDeclName()); })
          .Case([&](ImportOp op) { addDeclForOp(op, op.getSymNameAttr()); })
          .Case([&](UnresolvedImportOp op) {
            addDeclForOp(op, op.getImportNameAttr());
          })
          .Case([&](UnresolvedWildcardImportOp op) {
            decl.addUnresolvedWildcardImport(UnresolvedWildcardImport{
                op.getModulePathAttr(), decl.getLoc()});
          })
          .Case([&](StructDeclOp op) {
            ASTDecl &structDecl = addDeclForOp(op, op.getSymNameAttr());
            structDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(op));
            for (ParamDeclAttr param : op.getParams()) {
              // Add the parameters as accessible member decls. Make sure
              // to only demangle user-visible names. Synthetic closure structs
              // use internal parameter names that should remain hidden.
              StringRef paramName =
                  op.isSynthetic() ? param.getName()
                                   : demangleParameterName(param.getName());
              declResolver->addFullyResolvedDecl(
                  PValue(ParamDeclRefAttr::get(param)), paramName,
                  structDecl.getLoc(), &structDecl);
            }
          })
          .Case([&](TraitDeclOp op) {
            ASTDecl &traitDecl = addDeclForOp(op, op.getSymNameAttr());
            traitDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(op));
            // TODO(traits): Add decls for parameters, when they exist.
          })
          .Case([&](ExtensionDeclOp op) {
            SymbolRefAttr targetStruct = op.getTargetStruct().value();
            StringAttr baseName = targetStruct.getLeafReference();
            // Extensions are registered under two names:
            // - "extension:MyStruct", for looking for all extensions for a
            //   given MyStruct
            // - "extension:", for looking for all extensions for any struct
            //   in a given scope (useful for importing).
            // Register this extension under both names now. The "extension:"
            // prefix and marker are single-sourced from extensionsScopeMarker.
            // TODO(MOCO-522): Arcana docs on this!
            std::string extensionName = extensionsScopeMarker.getValue().str() +
                                        baseName.getValue().str();
            StringAttr extensionNameAttr =
                StringAttr::get(op.getContext(), extensionName);
            ASTDecl &extensionDecl = addDeclForOp(op, extensionNameAttr);
            declResolver->aliasDeclInParent(&extensionDecl,
                                            extensionsScopeMarker);
          })
          .Case([&](AliasDeclOp op) {
            addDeclForOp(op, StringAttr::get(op.getContext(),
                                             demangleParameterName(
                                                 op.getParamDecl().getName())));
          })
          .Case([&](StructFieldOp op) { addDeclForOp(op, op.getNameAttr()); })
          .Case<FileModuleOp, PackageOp>([&](auto op) {
            assert(packageState &&
                   "FileModule or Package nested in non-package");
            StringAttr name = op.getSymNameAttr();
            // Record the import location *before* materializing the bytecode
            // decl, as in the process of translating the loc to an SMLoc we
            // resolve/register a buffer ID. After that happens we're unable to
            // stamp on an import loc.
            diags.recordImportedFileIncludeLoc(op->getLoc(), packageImportLoc);
            ASTDecl &decl = addDeclForOp(op, name);

            // Record a nested module state for this decl. The child is a
            // module inside the artifact's one file, so it shares the
            // enclosing package's origin and holds no spec of its own: it was
            // never resolved, so there is no candidate to describe.
            auto childState = std::make_unique<ModuleState>(&decl);
            childState->origin = packageState->origin;
            ModuleState &moduleState =
                packageState->insertNestedModule(name, std::move(childState));

            getModuleLoader().setState(decl, moduleState);
            if constexpr (std::is_same_v<decltype(op), PackageOp>)
              getModuleLoader().setPackageState(op, moduleState);
          })
          .Case([&](ConformanceOp op) {
            // Witness tables are considered signature-resolved from the start
            // since there's nothing else to resolve for its "signature". (see
            // CALROC for more).
            ASTDecl &decl =
                addDeclForOp(op, op.getTraitSymbol().getFlattenedName());
            decl.resolvedness = DeclResolvedness::signature;
          })
          .Default([&](Operation *op) { deferredOps.push_back(op); });
    }
  }

  // Resolve references within the deferred operations. These don't have
  // corresponding decls, so we manually resolve them now. Walk in pre-order so
  // that nested ops get visited too.
  for (Operation *op : deferredOps)
    if (op->walk(resolveSingleOp).wasInterrupted())
      return failure();

  // After processing the region, make sure any non-signature attributes get
  // resolved.
  refWalker.walk(declOp->getAttrDictionary());
  return success();
}

LogicalResult SharedState::finalizeImportedBytecodeModules() {
  // Collect all bytecode readers so we can identify which ops are still lazy
  // stubs (isMaterializable == true).
  SmallVector<mlir::BytecodeReader *> readers;
  for (auto &origin : getModuleLoader().getOrigins()) {
    if (origin->bytecodeReader)
      readers.push_back(&*origin->bytecodeReader);
  }

  // Collect unparsed bytecode decls whose ops are fully materialized (not lazy
  // stubs). These were placed in the IR as a side effect of container
  // body-resolution but were never themselves resolved. Their attributes may
  // reference structs that finalize() will delete as stubs, causing
  // verifySymbolUses to fail. We must erase them after finalize() runs.
  SmallVector<Operation *> materializedUnparsedOps;
  for (ASTDecl *decl : declResolver->parsedDeclList) {
    if (!decl->loadedFromBytecode ||
        decl->resolvedness != DeclResolvedness::unparsed)
      continue;
    if (Operation *op = decl->getIfOperation()) {
      bool isLazy = llvm::any_of(readers, [op](mlir::BytecodeReader *r) {
        return r->isMaterializable(op);
      });
      if (!isLazy)
        materializedUnparsedOps.push_back(op);
    }
    // Clear the ASTDecl pointer to avoid a dangling reference after the op
    // is erased below or deleted as a stub by finalize().
    decl->setIRValue(PValue(BoolAttr::get(getContext(), false)));
  }

  for (auto &origin : getModuleLoader().getOrigins()) {
    if (!origin->bytecodeReader)
      continue;

    // Finalize the bytecode. Any op that was never materialized is dropped,
    // *unless* its results are still referenced by materialized IR.
    if (failed(origin->bytecodeReader->finalize(
            [&](Operation *op) { return !op->use_empty(); })))
      return failure();
    // Erase the temporary ModuleOp that was used to read bytecode.
    origin->tmpModule.erase();
  }

  for (Operation *op : materializedUnparsedOps)
    op->erase();

  return success();
}

ArrayRef<std::string> SharedState::getIncludedFiles() const {
  return impl->includedFiles;
}

DebugInfo::SourceNameAttr
SharedState::getSourceName(mlir::SymbolOpInterface op) {
  return impl->sourceNames.getSourceName(op);
}

/// Given a valid pointer into a source buffer for some token, return the
/// length of the token by re-lex'ing it.  This is efficient.
static size_t getTokenLength(SharedState &shared, SMLoc loc) {
  // Because we know the pointer is to a valid place in a source buffer, and
  // because we know that all source buffers are NUL terminated, we know that
  // the end of buffer check isn't needed.  This allows us to form a lexer
  // without having to find the MemoryBuffer it came from, saving some expense
  // in diagnostic emission.
  const char *curPtr = loc.getPointer();

  // If the byte is NUL, it is an invalid token and might be end of buffer.
  if (*curPtr == '\0')
    return 0;

  // Use ~0U to indicate the end of the buffer, that should be fine as we don't
  // expect tokens to be >= 2^32 charachetrs long.
  // NOTE: We cannot use ~0ULL as it leads to integer overflow when computing
  // end of the StringRef.
  Lexer lexer(shared.diags,
              StringRef(curPtr, std::numeric_limits<uint32_t>::max()), curPtr);
  return lexer.getToken().getSpelling().size();
}

/// Given a pointer to the start of a token, find the end of it.
static void adjustTokenEndPoint(SharedState &shared, SMLoc &loc) {
  size_t tokenSize = getTokenLength(shared, loc);
  loc = SMLoc::getFromPointer(loc.getPointer() + tokenSize);
}

ASTDecl *SharedState::getOrCreateClosureTrait(SMLoc loc, ASTDecl &moduleDecl,
                                              FnTypeGeneratorType sig) {
  auto [key, numPrependedCaptures] = closureEmitter->getClosureTraitKey(sig);
  auto ptr = impl->closureTraits.find(key);
  if (ptr == impl->closureTraits.end()) {
    auto result = closureEmitter->createClosureTrait(moduleDecl, sig, key,
                                                     numPrependedCaptures, loc);
    impl->closureTraits.insert({key, result});
    return result;
  }
  return ptr->second;
}

ASTDecl *SharedState::getUniversalParametricClosureTrait() {
  if (!impl->parametricClosureTrait) {
    auto *closureTrait = IREmitter::createParametricClosureTrait(*this);
    assert(closureTrait && "internal error: failed to create closure trait");
    impl->parametricClosureTrait = closureTrait;
  }

  return impl->parametricClosureTrait;
}

bool SharedState::isUniversalParametricClosureTrait(TraitSymbolAttr symbol) {
  return getUniversalParametricClosureTrait()->getSymbolRef() ==
         symbol.getSymbol();
}

ASTDecl *SharedState::getOrCreateExtension(SMLoc loc, TraitDeclOp sourceTrait,
                                           TraitDeclOp targetTrait,
                                           ASTType sourceMetaType,
                                           ASTDecl *moduleDecl) {
  auto key = std::make_pair(
      std::make_pair(sourceTrait.getOperation(), targetTrait.getOperation()),
      moduleDecl);
  auto &extension = impl->closureExtensions[key];
  if (!extension)
    extension = closureEmitter->createExtensionStruct(
        *moduleDecl, sourceTrait, targetTrait, sourceMetaType, loc);
  return extension;
}

FnOp SharedState::getOrCreateFunctionThunk(Attribute key, CreateThunkFn create,
                                           SMLoc useLoc) {
  FnOp &thunk = impl->conversionThunks[key];
  if (!thunk)
    thunk = create(key, getTopLevelDecl(), useLoc);
  return thunk;
}

const llvm::MapVector<StringRef, Capture> &
SharedState::getCaptureRangeInScope(ASTDecl &scope) {
  return getImpl().capturesInScope[&scope];
}

void SharedState::addCaptureToScope(ASTDecl &scope, ASTDecl *captureDecl,
                                    Capture capture) {
  getImpl().capturesInScope[&scope].insert({capture.getSpelling(), capture});
  if (captureDecl->getParentDecl() != scope.parentDecl) {
    if (scope.getNearestDeclOfType<FnOp>())
      addCaptureToScope(*scope.parentDecl, captureDecl, capture);
  }
}

void SharedState::setDefaultCaptureForScope(ASTDecl &scope,
                                            CaptureConvention convention) {
  getImpl().captureConventionForScope[&scope] = convention;
}

CaptureConvention SharedState::defaultCaptureConventionInScope(ASTDecl &decl) {
  auto ptr = getImpl().captureConventionForScope.find(&decl);
  if (ptr != getImpl().captureConventionForScope.end())
    return ptr->second;
  return CaptureConvention::kConventionUnspecified;
}

bool SharedState::captureInstanceExistsInScope(ASTDecl &scope,
                                               StringRef spelling) {
  auto ptr = getImpl().capturesInScope.find(&scope);
  if (ptr == getImpl().capturesInScope.end())
    return false;
  auto capturePtr = ptr->second.find(spelling);
  return capturePtr != ptr->second.end();
}

ClosureParamCaptures *SharedState::getClosureParamCapturesForOp(Operation *op) {
  auto ptr = getImpl().closureParamCaptures.find(op);
  if (ptr == getImpl().closureParamCaptures.end())
    return nullptr;
  return &ptr->second;
}

ArrayRef<ClosureParamCapture>
SharedState::lookupClosureCaptureFromOp(Operation *startOp,
                                        StringAttr closureName) {
  auto lookup = [&](Operation *op) -> ArrayRef<ClosureParamCapture> {
    if (ClosureParamCaptures *captures = getClosureParamCapturesForOp(op)) {
      auto ptr = captures->find(closureName);
      if (ptr != captures->end())
        return ptr->second;
    }
    return {};
  };

  StructDeclOp structScope;
  for (Operation *op = startOp; op; op = op->getParentOp()) {
    if (isa<FnOp>(op)) {
      if (ArrayRef<ClosureParamCapture> captures = lookup(op);
          !captures.empty())
        return captures;
    } else if (auto structOp = dyn_cast<StructDeclOp>(op)) {
      structScope = structScope ? structScope : structOp;
    }
  }
  if (structScope)
    return lookup(structScope);
  return {};
}

void SharedState::setClosureParamCaptures(
    ASTDecl &functionDecl, ClosureParamCaptures closureParamCaptures) {
  getImpl().closureParamCaptures[functionDecl.getIfOperation()] =
      std::move(closureParamCaptures);
}

void SharedState::addClosureParamCaptures(
    ASTDecl &functionDecl, StringAttr closureName,
    SmallVector<ClosureParamCapture> captures) {
  getImpl().closureParamCaptures[functionDecl.getIfOperation()][closureName] =
      std::move(captures);
}

//===----------------------------------------------------------------------===//
// Listener Interface

/// Resolve the given decl in preparation for passing it to the listener for
/// member lookup.
static void resolveDeclForListenerLookup(DeclResolver &declResolver,
                                         ASTDecl &decl, SMLoc loc) {
  // Before passing off to the listener, resolve nested decls. This lets the
  // listener see the full set of declarations, as unresolved imports are
  // generally lazily resolved, and also ensures the availability of things like
  // documentation.
  if (failed(declResolver.resolveBody(decl, loc)))
    return;
  ArrayRef<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>> decls =
      decl.getDeclsInScope();
  for (int i = 0, e = decls.size(); i < e; ++i) {
    // Resolution may invalidate the decls vector, so we can't rely on
    // iterators here. We also don't fail, because the listener should be
    // tolerant to errors.
    auto &[name, children] = *std::next(decls.begin(), i);

    // This case sometimes occurs in invalid code in the LSP.
    if (children.empty())
      continue;

    (void)declResolver.resolveBody(*children.front(), loc);
  }
  // Resolve any pending wildcards in the decl. We don't care about failure
  // here, as we still want to enable lookup for the decls that could be
  // resolved.
  (void)declResolver.resolveAllWildcardImports(decl);
}

/// Return if the given parser listener is interested in the given location.
static bool isListenerInterestedInLoc(ParserListener *listener, SMLoc loc) {
  return listener && listener->isInterestedInLoc(loc);
}

void SharedState::notifyListenerOnAliasDecl(ASTDecl &decl,
                                            SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onAliasDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnArgumentDecl(ASTDecl &decl, StringRef argName,
                                               SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onArgumentDecl(&decl, argName, identifierLoc);
}

void SharedState::notifyListenerOnFunctionDecl(ASTDecl &decl,
                                               SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onFunctionDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnImport(SMLoc importLoc) {
  if (isListenerInterestedInLoc(parserListener, importLoc))
    parserListener->onImport(importLoc);
}

void SharedState::notifyListenerOnImport(
    SMLoc importLoc, function_ref<ASTDecl &()> getPackageDecl) {
  if (!isListenerInterestedInLoc(parserListener, importLoc))
    return;
  parserListener->onImport(
      [&]() -> ASTDecl * {
        ASTDecl &packageDecl = getPackageDecl();
        resolveDeclForListenerLookup(*declResolver, packageDecl, importLoc);
        return &packageDecl;
      },
      importLoc);
}

void SharedState::notifyListenerOnMemberLookup(ASTDecl &decl, SMLoc lookupLoc,
                                               bool searchParentScopes) {
  if (!isListenerInterestedInLoc(parserListener, lookupLoc))
    return;
  parserListener->onMemberLookup(
      [&]() -> ASTDecl * {
        resolveDeclForListenerLookup(*declResolver, decl, lookupLoc);

        // Resolve parent scopes if necessary.
        if (searchParentScopes) {
          ASTDecl *parentDecl = &decl;
          while ((parentDecl = parentDecl->getParentDecl()))
            resolveDeclForListenerLookup(*declResolver, *parentDecl, lookupLoc);
        }
        return &decl;
      },
      lookupLoc, searchParentScopes);
}

void SharedState::notifyListenerOnMemberLookup(
    SMLoc lookupLoc, function_ref<ASTDecl &()> getDeclFn,
    bool searchParentScopes) {
  if (isListenerInterestedInLoc(parserListener, lookupLoc))
    notifyListenerOnMemberLookup(getDeclFn(), lookupLoc, searchParentScopes);
}

void SharedState::notifyListenerOnModuleDecl(ASTDecl &decl,
                                             SMLoc identifierLoc) {
  // TODO: This hook should likely be removed in favor of just `onRef`. It's
  // used to index other modules for the sake of references, but we should just
  // handle this when we see the reference.
  if (parserListener)
    parserListener->onModuleDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnModuleImport(ASTDecl &decl,
                                               const ImportPath &modulePath,
                                               SMLoc loc) {
  if (!isListenerInterestedInLoc(parserListener, loc))
    return;
  if (!decl.getIfOperation())
    return;

  // Skip over relative module markers in the location. Note we're assuming each
  // "relativeLevel" is a one-char period character.
  loc = SMLoc::getFromPointer(loc.getPointer() + modulePath.relativeLevel);

  // Grab the decls for each of the referenced modules.
  SmallVector<ASTDecl *> decls;
  ASTDecl *declIt = &decl;
  for (int i = 0, e = modulePath.components.size(); i < e; ++i) {
    decls.push_back(declIt);
    declIt = declIt->getParentDecl();
  }

  // Notify the listener of each module import starting from the parent, so we
  // can skip past the position within the location.
  for (auto [name, segmentDecl] :
       llvm::zip(modulePath.components, llvm::reverse(decls))) {
    parserListener->onModuleImport(segmentDecl, name, loc);
    // TODO: adjust the location like this isn't accurate in the presence of
    // backtick identifiers. We would ideally have a loc per path segment.
    loc = SMLoc::getFromPointer(loc.getPointer() + name.size() + 1);
  }
}

void SharedState::notifyListenerOnParameterDecl(ASTDecl &decl,
                                                SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onParameterDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnStructDecl(ASTDecl &decl,
                                             SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onStructDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnStructFieldDecl(ASTDecl &decl,
                                                  SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onStructFieldDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnTraitDecl(ASTDecl &decl,
                                            SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onTraitDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnVariableDecl(ASTDecl &decl,
                                               SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onVariableDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling, SMLoc loc) {
  if (!loc.isValid())
    return;
  SMLoc endLoc = SMLoc::getFromPointer(loc.getPointer() + spelling.size());
  notifyListenerOnRef(decls, spelling, SourceRange::getByteLevel(loc, endLoc));
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling, SourceRange range) {
  if (isListenerInterestedInLoc(parserListener, range.getStart()))
    parserListener->onRef(decls, spelling, diags.convertToSMRange(range));
}

/// Return the location of the identifier in the given expression.
static SourceRange getIdentifierLocFromExpr(const ExprNode *expr) {
  if (auto attribute = dyn_cast<AttributeRefNode>(expr))
    return attribute->getAttributeNameRange();

  // For post-fix expression, ensure we get the location from the base, not the
  // operator.
  if (auto subscript = dyn_cast<SubscriptNode>(expr))
    return getIdentifierLocFromExpr(subscript->base);
  if (auto call = dyn_cast<CallNode>(expr))
    return getIdentifierLocFromExpr(call->callee);
  return expr->getRange();
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling,
                                      const ExprNode *expr) {
  notifyListenerOnRef(decls, spelling, getIdentifierLocFromExpr(expr));
}

/// Returns if the parser listener should be notified on references for the
/// given call syntax.
static bool shouldNotifyListenerForCall(CallSyntax syntax) {
  switch (syntax) {
  case CallSyntax::kDirectCall:
  case CallSyntax::kMethodCall:
  case CallSyntax::kAttribute:
    return true;
  case CallSyntax::kParamBindings:
  case CallSyntax::kMethodCallSynthetic:
  case CallSyntax::kIndirectCall:
  case CallSyntax::kTypeCall:
  case CallSyntax::kOperator:
  case CallSyntax::kReversedOperator:
  case CallSyntax::kSubscript:
  case CallSyntax::kImplicitConvert:
  case CallSyntax::kImplicitCopyCtor:
  case CallSyntax::kImplicitMoveCtor:
  case CallSyntax::kDestructor:
  case CallSyntax::kTupleGetItem:
    return false;
  }
  llvm_unreachable("unknown call syntax");
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling, const ExprNode *expr,
                                      CallSyntax syntax) {
  if (shouldNotifyListenerForCall(syntax))
    notifyListenerOnRef(decls, spelling, expr);
}

void SharedState::notifyListenerOnCall(ArrayRef<ASTDecl *> decls,
                                       SMLoc rParenLoc, CallSyntax syntax,
                                       const CallOperands &callOperands) {
  // Ignore synthetic calls to functions.
  if (syntax == CallSyntax::kMethodCallSynthetic ||
      syntax == CallSyntax::kImplicitConvert ||
      syntax == CallSyntax::kImplicitCopyCtor ||
      syntax == CallSyntax::kImplicitMoveCtor)
    return;

  if (isListenerInterestedInLoc(parserListener, rParenLoc))
    parserListener->onCall(decls, rParenLoc, callOperands);
}

void SharedState::notifyListenerOnParameterBinding(ArrayRef<ASTDecl *> decls,
                                                   llvm::SMLoc rsquareLoc,
                                                   ArrayRef<Operand> operands) {
  if (isListenerInterestedInLoc(parserListener, rsquareLoc)) {
    SmallVector<ExprNode *> parameters = llvm::map_to_vector(
        operands, [](const Operand &operand) { return operand.expr; });
    parserListener->onParameterBinding(decls, rsquareLoc, parameters);
  }
}

/// These two methods are used to memoize whether a type is implicitly
/// convertible to another type, which includes overload resolution etc.
std::optional<bool> SharedState::getCachedImplicitConvertibility(ASTType from,
                                                                 ASTType to) {
  DenseMap<std::pair<Type, Type>, bool> &cache =
      getImpl().cachedImplicitConvertibility;
  auto it = cache.find({from, to});
  if (it == cache.end())
    return {};

#ifndef NDEBUG
  // If this is the 64th convertibility hit, allow it to fail so we can detect
  // if the cache ever starts to depend on new state like declContext.  This is
  // a small bit a paranoia to make it possible to track down subtle bugs that
  // may happen in the future.
  if ((cache.size() & 63) == 0)
    return {};
#endif
  return it->second;
}
void SharedState::cacheImplicitConvertibility(ASTType from, ASTType to,
                                              bool isConvertible) {
  DenseMap<std::pair<Type, Type>, bool> &cache =
      getImpl().cachedImplicitConvertibility;
  auto [it, newlyInserted] = cache.insert({{from, to}, isConvertible});

  // If the entry is already present, make sure all checks agree.
  if (!newlyInserted)
    assert(it->second == isConvertible &&
           "convertibility cache disagrees from actual computation! Must need "
           "to include more information in the hash key");
}

/// These two methods memoize assumption-free nominal trait-conformance results.
std::optional<bool>
SharedState::getCachedNominalConformance(const ASTDecl *decl, TraitType trait,
                                         ASTType concreteType) {
  DenseMap<std::tuple<const ASTDecl *, Type, Type>, bool> &cache =
      getImpl().nominalConformanceCache;
  auto it = cache.find({decl, Type(trait), Type(concreteType)});
  if (it == cache.end())
    return {};

#ifndef NDEBUG
  // Paranoia (mirrors the convertibility cache above): whenever the cache size
  // is a multiple of 64, force a miss so the caller recomputes and the
  // store-side assert can catch the cache drifting from ground truth if the
  // result ever starts depending on state not in the key.
  if ((cache.size() & 63) == 0)
    return {};
#endif
  return it->second;
}
void SharedState::cacheNominalConformance(const ASTDecl *decl, TraitType trait,
                                          ASTType concreteType, bool conforms) {
  DenseMap<std::tuple<const ASTDecl *, Type, Type>, bool> &cache =
      getImpl().nominalConformanceCache;
  auto [it, newlyInserted] =
      cache.insert({{decl, Type(trait), Type(concreteType)}, conforms});

  // If the entry is already present, make sure the answers agree.
  if (!newlyInserted)
    assert(it->second == conforms &&
           "nominal conformance cache disagrees from actual computation! Must "
           "need to include more information in the hash key");
}

ParserEvaluationContext &SharedState::getEvaluationContext() {
  return impl->evaluationContext;
}

ParameterEvaluator SharedState::getParameterEvaluator() {
  ParameterEvaluator evaluator;
  evaluator.setEvaluationContext(&getEvaluationContext());
  return evaluator;
}

ParameterEvaluator
SharedState::getParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                                   ArrayRef<TypedAttr> paramValues) {
  ParameterEvaluator evaluator(paramDecls, paramValues);
  evaluator.setEvaluationContext(&getEvaluationContext());
  return evaluator;
}

namespace {
/// This struct is used to fold @always_inline("builtin") functions.
struct BuiltinFunctionFolder {
  SharedState &shared;
  ParameterEvaluator evaluator;
  bool doEmitError;

  // Keep track of the parameter values for each of the live SSA values in the
  // body, and start by binding the argument values.
  DenseMap<Value, TypedAttr> boundValues;

  // For our virtual memory model, we track entire indirect values like
  // var-decls in this map.  lit.struct.ref indexes to subfields are not
  // immediately processed - they are handled by load/store operations, mostly
  // to handle constructors.
  SmallDenseMap<Value, TypedAttr> varDeclSoFar;

  BuiltinFunctionFolder(SharedState &shared, bool doEmitError)
      : shared(shared), evaluator(shared.getParameterEvaluator()),
        doEmitError(doEmitError) {}

  // This helper handles emitting an error (or not) as needed.
  MojoInflightDiag emitError(Location loc) {
    auto result = shared.emitError(loc) << "'@always_inline(\"builtin\")' ";
    if (!doEmitError) // Only emit an error if requested.
      result.abandon();
    return result;
  }

  // Lookup a pre-bound value and check for validity.  This emits an error and
  // returns null if something goes wrong.
  TypedAttr findValue(Value v) {
    // RebindOp isn't tracked, just map it here.
    if (auto rebind = v.getDefiningOp<RebindOp>()) {
      auto result = findValue(rebind.getInput());
      if (!result)
        return {};
      auto destTy = evaluator.getReboundType(rebind.getType());
      // FIXME(OriginDepType) Origins shouldn't be dynamic values.
      if (isa<OriginType>(destTy))
        return OriginMutCastAttr::get(result, destTy);
      return ParamOperatorAttr::getRebind(result, destTy);
    }

    auto result = boundValues[v];
    if (!result)
      emitError(v.getLoc()) << "could not resolve operand value";
    return result;
  }

  void recordValue(Value v, TypedAttr attr) {
    assert(evaluator.getReboundType(v.getType()) == attr.getType() &&
           "incorrect fold");
    assert(!boundValues[v] && "value already has a bound value");
    boundValues[v] = attr;
  }

  // Load and store operations for VarDecls are limited, but have limited
  // support for field sensitivity and rebinds (due to sugar).
  TypedAttr getLoadedValue(Value srcRef) {
    // If this is a direct reference to a vardecl, return it.
    auto it = varDeclSoFar.find(srcRef);
    if (it != varDeclSoFar.end())
      return it->second;

    // If this is a rebind, it is just adjusting sugar.  Load the base and
    // rebind the result.
    if (auto rebind = srcRef.getDefiningOp<RebindOp>()) {
      auto base = getLoadedValue(rebind.getInput());
      if (!base)
        return {};
      ASTType actualType = evaluator.getReboundType(rebind.getType());
      actualType = actualType.getReferenceElementType();
      return ParamOperatorAttr::getRebind(base, actualType);
    }

    if (auto ger = srcRef.getDefiningOp<RefStructGEROp>()) {
      auto base = getLoadedValue(ger.getContainer());
      if (!base)
        return {};
      ASTType actualType = evaluator.getReboundType(ger.getType());
      actualType = actualType.getReferenceElementType();
      return LIT::StructExtractAttr::get(base, ger.getFieldAttr(), actualType);
    }

    // Otherwise fail.
    return {};
  }

  LogicalResult performStore(TypedAttr value, Value destRef) {
    // If this is a direct store to a variable, update it.
    auto it = varDeclSoFar.find(destRef);
    if (it != varDeclSoFar.end()) {
      it->second = value;
      return success();
    }

    // If this is a rebind, it is just adjusting sugar.
    if (auto rebind = destRef.getDefiningOp<RebindOp>()) {
      ASTType srcTy = evaluator.getReboundType(rebind.getInput().getType());
      srcTy = srcTy.getReferenceElementType();
      return performStore(ParamOperatorAttr::getRebind(value, srcTy),
                          rebind.getInput());
    }

    // To store to a subfield, we do a load of the entire value, update the
    // field, then store the whole value.
    if (auto ger = destRef.getDefiningOp<RefStructGEROp>()) {
      auto whole = getLoadedValue(ger.getContainer());
      if (!whole)
        return failure();

      // We need the struct decl to get the fields.
      auto structType = cast<LIT::StructType>(whole.getType());
      auto structDecl = ASTType(structType).getDecl(shared);
      auto structOp = cast<StructDeclOp>(structDecl->getIfOperation());

      // Rebuild the struct with the new value in the field.
      SmallVector<std::tuple<StringAttr, TypedAttr>> fields;
      for (auto field : structOp.getFieldDecls()) {
        // Use StructExtractAttr::get to compute the adjusted field type
        // (substituting in parameters etc) even for when we overwrite it.
        TypedAttr fieldValue;
        if (field.getNameAttr() == ger.getFieldAttr())
          fieldValue = value;
        else
          fieldValue = LIT::StructExtractAttr::get(whole, field);
        fields.push_back({field.getNameAttr(), fieldValue});
      }
      whole = LITStructAttr::get(fields, structType);
      return performStore(whole, ger.getContainer());
    }
    return failure();
  }

  /// Process the following operation, doing one of three things:
  /// 1) Fold it to a single TypedAttr, returning it.
  /// 2) Return a failure to indicate that the operation is not foldable.
  /// 3) Return a a null TypedAttr to indicate that the operation was processed
  /// but didn't produce a value (e.g. StoreOps).
  FailureOr<TypedAttr> fold(Operation &op);
};

// Handle a simple binary operation that folds to a SIMD binary attribute.
template <typename T>
FailureOr<TypedAttr> foldSIMDBinOp(Operation &op,
                                   BuiltinFunctionFolder &folder) {
  if (auto lhs = folder.findValue(op.getOperand(0)))
    if (auto rhs = folder.findValue(op.getOperand(1)))
      return T::get(lhs, rhs);
  return failure();
}

} // end anonymous namespace

/// Process the following operation, doing one of three things:
/// 1) Fold it to a single TypedAttr, returning it.
/// 2) Return a failure to indicate that the operation is not foldable.
/// 3) Return a a null TypedAttr to indicate that the operation was processed
/// but didn't produce a value (e.g. StoreOps).
FailureOr<TypedAttr> BuiltinFunctionFolder::fold(Operation &op) {
  if (auto paramCst = dyn_cast<ParamConstantOp>(op))
    return evaluator.getReboundAttribute(paramCst.getValue());
  if (auto cst = dyn_cast<mlir::index::ConstantOp>(op))
    return TypedAttr(cst.getValueAttr());

  if (auto extract = dyn_cast<LIT::StructExtractOp>(op)) {
    if (auto base = findValue(extract.getOperand()))
      return LIT::StructExtractAttr::get(
          base, extract.getFieldAttr(),
          evaluator.getReboundType(extract.getType()));
  }

  if (auto splatOp = dyn_cast<POP::SIMDSplatOp>(op)) {
    auto type = evaluator.getReboundType(splatOp.getType());
    if (auto op = findValue(splatOp.getScalar()))
      return SIMDSplatAttr::get(op, cast<SIMDType>(type));
  }

  // Handle a simple binary operation that folds to a POC binary op.
  auto foldBinOp = [&](POC opc) -> FailureOr<TypedAttr> {
    if (auto lhs = findValue(op.getOperand(0)))
      if (auto rhs = findValue(op.getOperand(1)))
        return ParamOperatorAttr::get(opc, lhs, rhs);
    return failure();
  };

  // Handle the select operation.
  auto foldSelectOp = [&](POP::SelectOp selectOp) -> FailureOr<TypedAttr> {
    auto cond = findValue(selectOp.getCondition());
    auto trueVal = findValue(selectOp.getTrueValue());
    auto falseVal = findValue(selectOp.getFalseValue());

    if (cond && trueVal && falseVal)
      return ParamOperatorAttr::get(POC::Cond, {cond, trueVal, falseVal},
                                    trueVal.getType());
    return failure();
  };

  // Many index binops fold directly to POC binops.
  if (auto add = dyn_cast<mlir::index::AddOp>(op))
    return foldBinOp(POC::Add);
  if (auto mul = dyn_cast<mlir::index::MulOp>(op))
    return foldBinOp(POC::Mul);
  if (auto andOp = dyn_cast<mlir::index::AndOp>(op))
    return foldBinOp(POC::And);
  if (auto orOp = dyn_cast<mlir::index::OrOp>(op))
    return foldBinOp(POC::Or);
  if (auto xorOp = dyn_cast<mlir::index::XOrOp>(op))
    return foldBinOp(POC::Xor);
  if (auto shlOp = dyn_cast<mlir::index::ShlOp>(op))
    return foldBinOp(POC::Shl);
  if (auto shrOp = dyn_cast<mlir::index::ShrSOp>(op))
    return foldBinOp(POC::Shr);
  if (auto andOp = dyn_cast<POP::AndOp>(op)) // i1 operations.
    return foldBinOp(POC::And);
  if (auto orOp = dyn_cast<POP::OrOp>(op))
    return foldBinOp(POC::Or);
  if (auto xorOp = dyn_cast<POP::XOrOp>(op))
    return foldBinOp(POC::Xor);
  if (auto div = dyn_cast<mlir::index::DivSOp>(op))
    return foldBinOp(POC::DivS);
  if (auto div = dyn_cast<mlir::index::DivUOp>(op))
    return foldBinOp(POC::DivU);
  if (auto div = dyn_cast<mlir::index::CeilDivSOp>(op))
    return foldBinOp(POC::CeilDivS);
  if (auto div = dyn_cast<mlir::index::CeilDivUOp>(op))
    return foldBinOp(POC::CeilDivU);
  if (auto div = dyn_cast<mlir::index::FloorDivSOp>(op))
    return foldBinOp(POC::FloorDivS);
  if (auto remSOp = dyn_cast<mlir::index::RemSOp>(op))
    return foldBinOp(POC::RemS);
  if (auto remUOp = dyn_cast<mlir::index::RemUOp>(op))
    return foldBinOp(POC::RemU);

  if (auto castOp = dyn_cast<POP::CastOp>(op))
    if (auto op = findValue(castOp.getOperand()))
      return POP::CastAttr::get(op, evaluator.getReboundType(castOp.getType()));

  if (auto toUI8Op = dyn_cast<POP::DTypeToUI8>(op))
    if (auto dtype = findValue(toUI8Op.getDType()))
      return POP::DTypeToUI8Attr::get(dtype);

  if (auto fromUI8Op = dyn_cast<POP::DTypeFromUI8>(op))
    if (auto value = findValue(fromUI8Op.getValue()))
      return POP::DTypeFromUI8Attr::get(value);

  if (auto castOp = dyn_cast<POP::CastFromBuiltinOp>(op))
    if (auto input = findValue(castOp.getOperand()))
      return CastFromBuiltinAttr::get(input, castOp.getType());

  if (auto castOp = dyn_cast<POP::CastToBuiltinOp>(op))
    if (auto input = findValue(castOp.getOperand()))
      return CastToBuiltinAttr::get(input, castOp.getType());

  if (auto selectOp = dyn_cast<POP::SelectOp>(op))
    return foldSelectOp(selectOp);

  // Sub doesn't have a POC opcode: "x-y" is "x+(y*-1)".
  if (auto sub = dyn_cast<mlir::index::SubOp>(op)) {
    if (auto lhs = findValue(sub.getOperand(0)))
      if (auto rhs = findValue(sub.getOperand(1)))
        return ParamOperatorAttr::getSub(lhs, rhs);
  }

  if (auto cmp = dyn_cast<mlir::index::CmpOp>(op)) {
    if (auto lhs = findValue(cmp.getOperand(0)))
      if (auto rhs = findValue(cmp.getOperand(1))) {
        switch (cmp.getPred()) {
        default:
          // TODO: we don't handle unsigned comparisons in ParamOperatorAttr
          // yet. It can do it, but we don't have a way to pass the unsigned
          // flag through easily.
          break;
        // TODO: we won't be needing the extra cast_to_builtin once we done
        // simd/int unification.
        case mlir::index::IndexCmpPredicate::EQ:
          return CastToBuiltinAttr::get(
              ParamOperatorAttr::get(POC::EQ, lhs, rhs));
        case mlir::index::IndexCmpPredicate::NE:
          return CastToBuiltinAttr::get(ParamOperatorAttr::getNE(lhs, rhs));
        case mlir::index::IndexCmpPredicate::SLT:
          return CastToBuiltinAttr::get(
              ParamOperatorAttr::get(POC::LT, lhs, rhs));
        case mlir::index::IndexCmpPredicate::SLE:
          return CastToBuiltinAttr::get(
              ParamOperatorAttr::get(POC::LE, lhs, rhs));
        case mlir::index::IndexCmpPredicate::SGT:
          return CastToBuiltinAttr::get(
              ParamOperatorAttr::get(POC::LT, rhs, lhs));
        case mlir::index::IndexCmpPredicate::SGE:
          return CastToBuiltinAttr::get(
              ParamOperatorAttr::get(POC::LE, rhs, lhs));
        }
      }
  }

  if (auto cmpOp = dyn_cast<POP::CmpOp>(op)) {
    if (auto lhs = findValue(cmpOp.getLhs())) {
      if (auto rhs = findValue(cmpOp.getRhs())) {
        POC cc;
        switch (cmpOp.getPred()) {
        case KGEN::CmpPredicate::EQ:
          cc = POC::EQ;
          break;
        case KGEN::CmpPredicate::NE:
          cc = POC::EQ;
          break;
        case KGEN::CmpPredicate::LT:
          cc = POC::LT;
          break;
        case KGEN::CmpPredicate::LE:
          cc = POC::LE;
          break;
        case KGEN::CmpPredicate::GT:
          cc = POC::LT;
          std::swap(lhs, rhs);
          break;
        case KGEN::CmpPredicate::GE:
          cc = POC::LE;
          std::swap(lhs, rhs);
          break;
        }
        auto resultType =
            cast<SIMDType>(evaluator.getReboundType(cmpOp.getType()));
        TypedAttr cmp = ParamOperatorAttr::get(cc, lhs, rhs);

        // For NE comparisons, negate the EQ result with an XOR
        if (cmpOp.getPred() == KGEN::CmpPredicate::NE) {
          // Splat a boolean 'true' value to the same width as the comparison.
          // This avoids us having to introspect the return type.
          KGEN::SIMDAttr oneVal = KGEN::SIMDAttr::get(
              KGEN::DTypeValue(true, DType::kBool),
              SIMDType::get(
                  /*size=*/1,
                  DTypeConstantAttr::get(cmpOp.getContext(), DType::kBool)));
          auto oneVecVal =
              KGEN::SIMDSplatAttr::get(oneVal, cast<SIMDType>(resultType));
          cmp = ParamOperatorAttr::get(POC::Xor, cmp, oneVecVal);
        }

        return cmp;
      }
    }
  }

  if (auto negOp = dyn_cast<POP::NegOp>(op))
    if (auto operand = findValue(negOp.getOperand()))
      return POP::SIMDNegAttr::get(operand);
  if (auto floorOp = dyn_cast<POP::FloorOp>(op))
    if (auto operand = findValue(floorOp.getOperand()))
      return POP::SIMDFloorAttr::get(operand);
  if (auto ceilOp = dyn_cast<POP::CeilOp>(op))
    if (auto operand = findValue(ceilOp.getOperand()))
      return POP::SIMDCeilAttr::get(operand);
  if (auto truncOp = dyn_cast<POP::TruncOp>(op))
    if (auto operand = findValue(truncOp.getOperand()))
      return POP::SIMDTruncAttr::get(operand);
  if (auto absOp = dyn_cast<POP::AbsOp>(op))
    if (auto operand = findValue(absOp.getOperand()))
      return POP::SIMDAbsAttr::get(operand);
  if (auto roundOp = dyn_cast<POP::RoundOp>(op))
    if (auto operand = findValue(roundOp.getOperand()))
      return POP::SIMDRoundAttr::get(operand);

  if (isa<POP::AddOp>(op))
    return foldBinOp(POC::Add);
  if (isa<POP::SubOp>(op))
    return foldSIMDBinOp<POP::SIMDSubAttr>(op, *this);
  if (isa<POP::MulOp>(op))
    return foldBinOp(POC::Mul);
  if (isa<POP::DivOp>(op))
    return foldBinOp(POC::Div);
  if (isa<POP::FloorDivOp>(op))
    return foldBinOp(POC::FloorDivS);
  if (isa<POP::SIMDAndOp>(op))
    return foldBinOp(POC::And);
  if (isa<POP::SIMDXOrOp>(op))
    return foldBinOp(POC::Xor);
  if (isa<POP::SIMDOrOp>(op))
    return foldBinOp(POC::Or);
  if (isa<POP::ShlOp>(op))
    return foldSIMDBinOp<POP::SIMDShlAttr>(op, *this);
  if (isa<POP::ShrOp>(op))
    return foldSIMDBinOp<POP::SIMDShrAttr>(op, *this);

  if (auto reduceOrOp = dyn_cast<POP::SIMDReduceOrOp>(op)) {
    if (auto lhs = findValue(op.getOperand(0))) {
      return POP::SIMDReduceOrAttr::get(
          lhs, cast<SIMDType>(evaluator.getReboundType(reduceOrOp.getType())));
    }
  }

  if (auto reduceAndOp = dyn_cast<POP::SIMDReduceAndOp>(op)) {
    if (auto lhs = findValue(op.getOperand(0))) {
      return POP::SIMDReduceAndAttr::get(
          lhs, cast<SIMDType>(evaluator.getReboundType(reduceAndOp.getType())));
    }
  }

  if (auto bitcast = dyn_cast<POP::PointerBitcastOp>(op)) {
    if (auto src = findValue(bitcast.getInput()))
      return ParamOperatorAttr::get(
          POC::PtrBitcast, src, evaluator.getReboundType(bitcast.getType()));
  }

  // FIXME(StringLiteral): Remove this operation.
  if (auto strSize = dyn_cast<POP::StringSizeOp>(op)) {
    if (auto str = findValue(strSize.getStr()))
      return POP::StringSizeAttr::get(str.getContext(), str);
  }

  if (auto call = dyn_cast<LIT::CallOp>(op)) {
    TypedAttr callee = evaluator.getReboundAttribute(call.getCallee());

    // Check for recursion. At this point we have only marked the callee as
    // being processed if we have already visited it in an always-inline
    // context.
    if (auto symCst = dyn_cast<SymbolConstantAttr>(callee)) {
      auto &resolver = shared.getDeclResolver();
      if (ASTDecl *calleeDecl =
              resolver.getDeclForFuncSymbol(symCst.getSymbol())) {
        if (resolver.isAlreadyProcessing(*calleeDecl)) {
          emitError(op.getLoc()) << "does not support recursion";
          return {};
        }
      }
    }

    SmallVector<TypedAttr> calleeOperands;
    calleeOperands.push_back(callee);
    for (auto operandVal : call.getOperands()) {
      calleeOperands.push_back(findValue(operandVal));
      if (!calleeOperands.back())
        return failure();
    }

    // Note that the recursive call here always generates an error.  We know
    // that this was inside of a "builtin" function so we're not being
    // called speculatively on an arbitrary function.
    if (auto result =
            shared.foldInlineBuiltinFunction(calleeOperands, op.getLoc(),
                                             /*emitError=*/true))
      return result;
    return failure();
  }

  if (auto varDecl = dyn_cast<VarDeclOp>(op)) {
    auto eltType = evaluator.getReboundType(varDecl.getType().getElementType());
    // Permit vardecls of certain types we know about.
    if (eltType.isIntOrIndexOrFloat() || isa<SIMDType, DTypeType>(eltType)) {
      varDeclSoFar[varDecl] = UninitMemAttr::get(eltType);
      return TypedAttr();
    }
    // The primary pattern we're trying to handle here is:
    //   %tmp = lit.var.decl Int
    //   %tmp2 = lit.ref.struct.ger %tmp, value
    //   lit.ref.store %v, %tmp2
    //   lit.load.consume %tmp
    // Which happens in ctors for builtin operations.  Ignore anything more
    // complex.
    ASTDecl *decl = ASTType(eltType).getDecl(shared);
    if (decl &&
        ASTType(eltType).isTrivialRegisterType(decl->getLoc(), shared)) {
      // A struct with no fields never get initialized, so it has to start as a
      // pre-built value.
      auto structOp = dyn_cast_or_null<StructDeclOp>(decl->getIfOperation());
      if (structOp && structOp.getFieldDecls().empty())
        varDeclSoFar[varDecl] = LIT::getEmptyStructValue(eltType);
      else
        varDeclSoFar[varDecl] = UninitMemAttr::get(eltType);
      return TypedAttr();
    }
  }

  if (isa<AliasDeclOp>(op))
    return TypedAttr(); // handled by user.

  if (isa<LoadConsumeOp, RefLoadOp>(op))
    return getLoadedValue(op.getOperand(0));

  if (isa<RefStructGEROp, RebindOp>(op))
    return TypedAttr(); // handled by user.

  if (auto store = dyn_cast<RefStoreOp>(op)) {
    TypedAttr value = findValue(store.getValue());
    if (value && succeeded(performStore(value, store.getDest())))
      return TypedAttr();
  }

  if (auto variant = dyn_cast<VariantCreateOp>(op)) {
    if (TypedAttr value = findValue(variant.getOperand())) {
      auto resType =
          cast<VariantType>(evaluator.getReboundType(variant.getType()));
      return TypedAttr(VariantAttr::get(value, variant.getIndex(), resType));
    }
  }

  // Folds a block that terminates with a single-operand yield op identified by
  // `isYieldOp`. Returns the folded yield operand on success, or failure() if
  // any op in the block cannot be folded or the yield is malformed.
  auto foldBlock =
      [&](Block &block,
          function_ref<bool(Operation &)> isYieldOp) -> FailureOr<TypedAttr> {
    for (Operation &op : block) {
      if (isYieldOp(op)) {
        if (op.getNumOperands() == 1)
          return findValue(op.getOperand(0));
        emitError(op.getLoc()) << "can only handle single-result if";
        return failure();
      }
      FailureOr<TypedAttr> result = fold(op);
      if (failed(result))
        return failure();
      if (TypedAttr val = *result)
        recordValue(op.getResult(0), val);
    }
    // If there is no block terminator then we have malformed IR, presumably
    // due to an already-diagnosed issue.
    return failure();
  };

  // We can fold hlcf.if / single-arm hlcf.elif operations in limited form that
  // end with a yield of a single value for which both sides are foldable.
  auto foldIfLike = [&](Value cond, Block &thenBlock,
                        Block &elseBlock) -> FailureOr<TypedAttr> {
    auto condVal = findValue(cond);
    if (!condVal)
      return TypedAttr();
    auto isYield = [](Operation &op) { return isa<HLCF::YieldOp>(op); };
    auto trueVal = foldBlock(thenBlock, isYield);
    if (failed(trueVal))
      return trueVal;
    auto falseVal = foldBlock(elseBlock, isYield);
    if (failed(falseVal))
      return falseVal;

    return ParamOperatorAttr::get(POC::Cond, {condVal, *trueVal, *falseVal},
                                  trueVal->getType());
  };
  if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
    auto folded =
        foldIfLike(ifOp.getCond(), ifOp.getThenBlock(), ifOp.getElseBlock());
    if (failed(folded) || *folded)
      return folded;
  }
  if (auto elifOp = dyn_cast<HLCF::ElifOp>(op)) {
    // Only fold the simple if/else shape (no additional elif arms).
    if (elifOp.getElifRegions().empty()) {
      auto folded = foldIfLike(elifOp.getCond(), elifOp.getThenBlock(),
                               elifOp.getElseBlock());
      if (failed(folded) || *folded)
        return folded;
    }
  }

  // Handle kgen.param.if: the condition is already a TypedAttr, so we just
  // need to fold both branches and produce a POC::Cond param expression.
  if (auto paramIfOp = dyn_cast<KGEN::ParamIfOp>(op)) {
    // Substitute concrete parameter bindings, as ParamConstantOp does.
    TypedAttr condVal = evaluator.getReboundAttribute(paramIfOp.getCond());
    auto isParamYield = [](Operation &op) {
      return isa<KGEN::ParamYieldOp>(op);
    };
    auto trueVal = foldBlock(paramIfOp.getThenRegion().front(), isParamYield);
    if (failed(trueVal))
      return trueVal;
    auto falseVal = foldBlock(paramIfOp.getElseRegion().front(), isParamYield);
    if (failed(falseVal))
      return falseVal;

    return ParamOperatorAttr::get(POC::Cond, {condVal, *trueVal, *falseVal},
                                  trueVal->getType());
  }

  // FIXME(MOCO-2839): We silently ignore 'kgen.param.assert' ops when folding
  // @always_inline("builtin") functions. We should either support these or
  // remove them from the key locations in the std which we wish to support
  // (SIMD).
  if (isa<KGEN::ParamAssertOp>(op))
    return TypedAttr();

  if (isa<LIT::MojoVersionMajorOp, LIT::MojoVersionMinorOp,
          LIT::MojoVersionPatchOp>(op)) {
    const ProjectVersion version = M::getMojoVersion();
    auto foldVersionOp = [&](int64_t number) -> TypedAttr {
      return KGEN::SIMDAttr::get(
          KGEN::DTypeValue(number, KGENDType::index),
          SIMDType::get(
              /*size=*/1,
              DTypeConstantAttr::get(op.getContext(), KGENDType::index)));
    };

    if (isa<LIT::MojoVersionMajorOp>(op))
      return foldVersionOp(version.major);
    if (isa<LIT::MojoVersionMinorOp>(op))
      return foldVersionOp(version.minor);
    if (isa<LIT::MojoVersionPatchOp>(op))
      return foldVersionOp(version.patch);
  }

  // Otherwise we don't know what this is, bail out.
  emitError(op.getLoc()) << "does not support MLIR operation "
                         << op.getName().getStringRef();
  return failure();
}

/// Given a parameter expression call to a function marked
/// @always_inline("builtin"), scan the function to form an inlined parameter
/// expression representation of the function given the specified argument
/// values, then return the resultant expression.  If the function cannot be
/// handled as a builtin, emit an error (when emitError is true) and return
/// null.
TypedAttr SharedState::foldInlineBuiltinFunction(ArrayRef<TypedAttr> operands,
                                                 Location callLoc,
                                                 bool emitError) {

  BuiltinFunctionFolder folder(*this, emitError);

  // Resolve the callee and check to verify it is a "builtin" call that is
  // eligible for parameter inlining.
  auto symCst = dyn_cast<SymbolConstantAttr>(operands.front());
  if (!symCst) {
    folder.emitError(callLoc) << "only supports direct calls";
    return {};
  }
  operands = operands.drop_front();

  auto &resolver = getDeclResolver();
  ASTDecl *calleeDecl = resolver.getDeclForFuncSymbol(symCst.getSymbol());
  assert(llvm::isa_and_present<FnOp>(calleeDecl->getIfOperation()) &&
         "callee isn't known?");
  auto fnOp = cast_or_null<FnOp>(calleeDecl->getIfOperation());
  if (fnOp.getInlineLevel() != InlineLevel::AlwaysBuiltin) {
    folder.emitError(callLoc) << "only supports calls to other "
                                 "'@always_inline(\"builtin\")' functions";
    return {};
  }
  if (failed(resolver.resolveBody(*calleeDecl, calleeDecl->getLoc())) ||
      // Double check to ensure body resolution's check succeeded.
      fnOp.getInlineLevel() != InlineLevel::AlwaysBuiltin) {
    return {}; // Error already diagnosed.
  }

  // The function being called may be a generic function - if so, we need to
  // remap any values and types in the body with parameter values substituted.
  for (auto [decl, value] :
       llvm::zip(fnOp.collectAllParams(/*implOrigins*/ false),
                 symCst.getParamValues())) {
    assert(value.getType() == folder.evaluator.getReboundType(decl.getType()));
    folder.evaluator.setDeclBinding(decl, value);
  }

  // Bind the argument values we are provided.
  for (auto [convention, arg, argValue] :
       llvm::zip(fnOp.getFuncTypeGenerator().getArgConventions(),
                 fnOp.getBody()->getArguments(), operands)) {
    if (convention != ArgConvention::ImmReg) {
      folder.emitError(arg.getLoc())
          << "does not support this argument convention";
      return {};
    }
    if (folder.evaluator.getReboundType(arg.getType()) != argValue.getType()) {
      folder.emitError(arg.getLoc()) << "argument type mismatch";
      return {};
    }
    folder.boundValues[arg] = argValue;
  }

  // This function handles a very limited set of operations and no control
  // flow. As such, we can proceed top-down and bail out if we see anything
  // too complex for our little brain.
  for (Operation &op : *fnOp.getBody()) {
    // Handle the final return.
    if (auto ret = dyn_cast<LIT::ReturnOp>(op)) {
      if (ret.getNumOperands() == 1)
        return folder.findValue(ret.getOperand(0));
    }

    // Otherwise it must be an operation that we can fold.
    FailureOr<TypedAttr> result = folder.fold(op);
    if (failed(result))
      return {}; // Error already diagnosed.

    // Otherwise we know this operation. If it returned a value remember it.
    if (TypedAttr val = *result)
      folder.recordValue(op.getResult(0), val);
  }

  // If there is no block terminator then we have malformed IR, presumably
  // due to an already-diagnosed issue.
  return {};
}

bool Capture::isCopy() const { return !isRef(); }
bool Capture::isRef() const {
  return kind == CaptureConvention::kConventionRef ||
         kind == CaptureConvention::kConventionRead ||
         kind == CaptureConvention::kConventionMut;
}
