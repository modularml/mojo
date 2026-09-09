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

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITInterfaces.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DocString.h"
#include "Mojo/lib/MojoParser/Traits.h"
#include "MojoUtils.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

ASTDecl::ASTDecl(SharedState &shared, DeclIRValue irValue, llvm::SMLoc loc,
                 ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                 ssize_t indentation)
    : shared(shared), irValue(irValue), loc(loc), parentDecl(parentDecl),
      cursor(cursor), endCursorState(endCursor.getState()),
      indentation(indentation) {
  resolvedness = DeclResolvedness::unparsed;
  referencedFromBytecode = false;
  hasDisabledDecls = false;
  hasReferenceError = false;
  hasBodyDecorators = false;
  loadedFromBytecode = false;
  isExplicitParamScope = false;
}

bool ASTDecl::isCallableDecl() const {
  if (auto witness = getIfWitness())
    return isa<FnTypeGeneratorType>(witness->getWitnessEntry().witnessType);
  return isa_and_nonnull<FnOp>(getIfOperation());
}

FnTypeGeneratorType ASTDecl::getDeclFullSignature() const {
  if (auto witness = getIfWitness()) // already expanded.
    return cast<FnTypeGeneratorType>(witness->getWitnessEntry().witnessType);
  return cast<FnOp>(getIfOperation()).getFullSignature();
}

bool ASTDecl::isStaticMethodDecl() const {
  if (auto witness = getIfWitness())
    return witness->getWitnessEntry().isStaticMethod;
  return cast<FnOp>(getIfOperation()).getIsStatic();
}

ImplicitConversionKind ASTDecl::getDeclImplicitConversionKind() const {
  if (auto witness = getIfWitness())
    return witness->getWitnessEntry().implicitConversion;
  return cast<FnOp>(getIfOperation()).getImplicitConversion();
}

DocStringAttr ASTDecl::getDocString() const {
  if (auto astDeclOp = dyn_cast_or_null<ASTDeclInterface>(getIfOperation()))
    return astDeclOp.getDocStringAttr();
  return {};
}

void ASTDecl::setErroneous() { hasReferenceError = true; }

std::optional<DocString> ASTDecl::getParsedDocString() const {
  if (auto rawDocStr = getDocString())
    return DocString(rawDocStr);
  return {};
}

/// Given a reference to a parameter, look at this declaration and enclosing
/// scopes to find the ASTDecl that defines it (e.g. the enclosing function,
/// struct or trait).  This can return null if not found.
std::tuple<const ASTDecl *, ArrayRef<ParamDeclAttr>, size_t>
ASTDecl::lookupParamReference(ParamDeclRefAttr paramRef) const {
  auto paramDecl = ParamDeclAttr::get(paramRef);

  const ASTDecl *current = this;
  for (; current; current = current->getParentDecl()) {
    if (auto op = current->getIfOperation()) {
      if (auto declIntf = dyn_cast<DeclInterface>(op)) {
        for (auto [idx, param] : llvm::enumerate(declIntf.getInputParams())) {
          if (param.getName() == paramDecl.getName() &&
              param.getType() == paramDecl.getType())
            return {current, declIntf.getInputParams(), idx};
        }
      }
    }
  }
  return {nullptr, {}, ~size_t(0)}; // Not found.
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringRef name) const {
  return lookupInCurrentScope(StringAttr::get(getContext(), name));
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringAttr name) const {
  assert(resolvedness == DeclResolvedness::body &&
         "cannot perform lookup in a decl that isn't fully resolved");
  if (!declsInScope)
    return {};

  auto it = declsInScope->find(name);
  if (it != declsInScope->end() && !it->second.empty()) {
    // A name whose decls are all disabled is a miss
    if (LLVM_UNLIKELY(hasDisabledDecls) &&
        llvm::all_of(it->second,
                     [](ASTDecl *decl) { return decl->isDisabled(); })) {
      return {};
    }
    return it->second;
  }
  return {};
}

/// If this is a method of a struct or trait, return the decl for the struct
/// or trait.
ASTDecl *ASTDecl::tryGetMethodParentDecl() const {
  // Methods are always FuncOps.
  if (!isa_and_nonnull<FnOp>(getIfOperation()))
    return nullptr;

  // Don't return non-null for nested functions or module-level functions.
  ASTDecl *parent = getParentDecl();
  return isa_and_nonnull<StructDeclOp, TraitDeclOp, ExtensionDeclOp>(
             parent->getIfOperation())
             ? parent
             : nullptr;
}

/// Collect the struct/trait declaration and all visible extension declarations
/// for the given type from this use-site context.
llvm::SmallVector<ASTDecl *, 4>
ASTDecl::collectTypeAndExtensions(ASTType type, llvm::SMLoc callLoc) {
  SharedState &shared = getShared();
  auto astDecl = type.getDecl(shared);

  SmallVector<ASTDecl *, 4> result;
  if (astDecl)
    result.push_back(astDecl);

  // Handle both struct and trait types for extension lookup.
  if (!astDecl || !astDecl->getUserNameIfOperation() ||
      !isa<StructDeclOp, TraitDeclOp>(astDecl->getIfOperation()))
    return result;

  // Now find all extensions that target this struct/trait.
  // Extensions are registered with the name of their target type, prefixed
  // with "extension:" (e.g., "extension:Spaceship") so that we can do this
  // lookup here.
  StringRef typeName = astDecl->getUserNameIfOperation().value();
  std::string extensionName =
      shared.extensionsScopeMarker.getValue().str() + typeName.str();
  LookupAllResult lookupResult =
      shared.lookupAllDeclsWithName(extensionName, callLoc, *this, true);

  // Only consider results from successful lookups. Lookups with isErroneous()
  // means the error was already diagnosed. Lookups with isFailure() should
  // be rare since we expect to find at least the original type declaration,
  // but we gracefully handle it by skipping extension lookup.
  if (!lookupResult.isSuccess())
    return result;

  ArrayRef<ASTDecl *> foundAstDecls = lookupResult.getIfSuccess();
  SymbolRefAttr typeSymbol = astDecl->getSymbolRef();
  for (ASTDecl *foundAstDecl : foundAstDecls) {
    if (failed(shared.declResolver->resolveBody(*foundAstDecl, callLoc))) {
      // Do nothing, skip it. Errors were already printed out, and we don't
      // mind missing call candidates from erroneous extensions.
      continue;
    }

    // The bucket is keyed by the target's leaf name only, so extensions of
    // distinct types that share a name land together; keep only those whose
    // resolved target is this type's own decl, as
    // findExtensionsInScopeForStruct does.
    auto extOp =
        dyn_cast_or_null<ExtensionDeclOp>(foundAstDecl->getIfOperation());
    if (extOp && extOp.getTargetStructAttr() == typeSymbol)
      result.push_back(foundAstDecl);
  }

  return result;
}

void ASTDecl::takeDecls(ASTDecl &src) {
  if (src.isErroneous())
    setErroneous();
  for (auto &[name, children] : src.getDeclsInScope())
    for (ASTDecl *child : children)
      child->parentDecl = this;
  declsInScope = std::move(src.declsInScope);
  counter = src.counter;
  assert(!knownAssumptions && "assumptions should be empty before overwriting");
  knownAssumptions = std::move(src.knownAssumptions);
}

/// Append the children decls of `src` into this scope (updating their
/// `parentDecl`), without replacing existing names already in this scope.
/// Used when promoting temporary pattern-binding scopes into a case scope.
void ASTDecl::mergeDeclsFrom(ASTDecl &src) {
  if (src.isErroneous())
    setErroneous();
  if (!src.declsInScope)
    return;
  if (!declsInScope)
    declsInScope.reset(new DeclInScopeType());
  for (auto &[name, children] : *src.declsInScope) {
    TinyPtrVector<ASTDecl *> &entries = (*declsInScope)[name];
    for (ASTDecl *child : children) {
      child->parentDecl = this;
      entries.push_back(child);
    }
  }
  src.declsInScope.reset();
}

DenseMap<TraitSymbolAttr, std::pair<TraitSymbolAttr, SMLoc>> *
ASTDecl::getTraitConformanceLineage(bool createIfMissing) {
  if (!traitConformanceLineage && createIfMissing)
    traitConformanceLineage.reset(new TraitConformanceLineageType());
  return traitConformanceLineage.get();
}

void ASTDecl::getKnownAssumptionsIncludingParents(
    SmallVectorImpl<ConstraintAttr> &assumptions) const {
  const ASTDecl *decl = this;
  while (decl) {
    if (decl->knownAssumptions)
      assumptions.append(decl->knownAssumptions->begin(),
                         decl->knownAssumptions->end());
    decl = decl->getParentDecl();
  }
}

llvm::SmallVector<ConstraintAttr>
ASTDecl::getAssumptionsFromScope(const ASTDecl *scope) {
  llvm::SmallVector<ConstraintAttr> assumptions;
  if (scope)
    scope->getKnownAssumptionsIncludingParents(assumptions);
  return assumptions;
}

void ASTDecl::insertKnownAssumptions(ArrayRef<ConstraintAttr> assumptions) {
  if (!knownAssumptions)
    knownAssumptions.reset(new llvm::SetVector<ConstraintAttr>());
  knownAssumptions->insert(assumptions.begin(), assumptions.end());
}

/// Return the nearest parameter scope (i.e. DeclInterface) for the given decl,
/// as well as the total depth from the nearest file module.
static std::pair<ASTDecl *, size_t> getNearestParamScopeAndDepth(
    ASTDecl *decl, function_ref<void(const ASTDecl *)> checkForCollision) {
  ASTDecl *paramScope = nullptr;
  size_t depth = 0;
  while (decl) {
    checkForCollision(decl);

    bool isParamScope =
        isa_and_nonnull<DeclInterface>(decl->getIfOperation()) ||
        decl->getIsExplicitParamScope();
    if (!paramScope && isParamScope)
      paramScope = decl;

    if (isParamScope) {
      ++depth;
      if (isa_and_nonnull<FileModuleOp>(decl->getIfOperation()))
        break;
    }

    decl = decl->getParentDecl();
  }

  return {paramScope, --depth}; // Adjust so depth starts at 0.
}

void ASTDecl::addRecursivelyStableName(StringAttr name) {
  if (!recursivelyStableNames)
    recursivelyStableNames = std::make_unique<llvm::DenseSet<StringAttr>>();
  recursivelyStableNames->insert(name);
}

bool ASTDecl::hasRecursivelyStableName(StringAttr name) const {
  for (const ASTDecl *scope = this; scope; scope = scope->getParentDecl()) {
    if (scope->recursivelyStableNames &&
        scope->recursivelyStableNames->contains(name))
      return true;
  }
  return false;
}

void ASTDecl::addDocInlineName(StringAttr name) {
  if (!docInlineNames)
    docInlineNames = std::make_unique<llvm::DenseSet<StringAttr>>();
  docInlineNames->insert(name);
}

bool ASTDecl::isDocInlineName(StringAttr name) const {
  return docInlineNames && docInlineNames->contains(name);
}

bool ASTDecl::hasRecursivelyStableType(const ASTDecl *typeDecl) const {
  if (!typeDecl)
    return false;
  if (auto name = typeDecl->getUserNameIfOperation())
    return hasRecursivelyStableName(StringAttr::get(getContext(), *name));
  return false;
}

bool UnresolvedWildcardImport::markSearched(StringRef name) {
  if (!searchedNames)
    searchedNames.reset(new llvm::StringSet<>());
  return searchedNames->insert(name).second;
}

/// Add an unresolved wild card import into this scope.
void ASTDecl::addUnresolvedWildcardImport(
    UnresolvedWildcardImport unresolvedImport) {
  // Lazy allocate the storage.
  if (!unresolvedWildcardImports)
    unresolvedWildcardImports.reset(new UnresolvedWildcardImportsType());
  else {
    // If we are already tracking this entity, mark it as superseded so we can
    // place it last. The last unresolved wildcard statement always wins:
    //   from a import *
    //   from b import *
    //   from a import *
    for (UnresolvedWildcardImport &import : *unresolvedWildcardImports)
      if (import.moduleName == unresolvedImport.moduleName)
        import.isSuperseded = true;
  }
  unresolvedWildcardImports->emplace_back(std::move(unresolvedImport));
}

/// Mangle a parameter name for the given parameter scope and scope depth. Due
/// to the use of depth, the mangling doesn't change when the order of function
/// declarations change, so we have hash stability.
static StringAttr mangleParamNameImpl(const Twine &name, size_t depth,
                                      ASTDecl *paramScope) {
  MLIRContext *ctx = paramScope->getContext();

  // Top level funcs/structs are the most common, so we want to simplify the
  // mangling for that case. Many tests (and real world code too) has a single
  // parameter in a scope, so we also try to make that case nicer.
  std::string suffix = "`";
  if (depth != 1)
    suffix.append(llvm::utostr(depth) + 'x');
  if (size_t id = paramScope->getNextUniqueID(); id != 0)
    suffix.append(llvm::utostr(id));

  return StringAttr::get(ctx, name + suffix);
}

StringAttr ASTDecl::mangleUserDefinedParamName(StringAttr name) {
  bool hasCollision = false;
  auto [paramScope, depth] =
      getNearestParamScopeAndDepth(this, [&](const ASTDecl *curScope) {
        hasCollision =
            hasCollision || !curScope->lookupInCurrentScope(name).empty();
      });
  if (!hasCollision)
    return name;

  return mangleParamNameImpl(name.strref(), depth, paramScope);
}

StringAttr ASTDecl::mangleParamName(const Twine &name) {
  // This function always mangles, so no need to check for collisions.
  auto [paramScope, depth] =
      getNearestParamScopeAndDepth(this, [&](const ASTDecl *) {});
  return mangleParamNameImpl(name, depth, paramScope);
}

void ASTDecl::dump() const {
  // The value is either an operation or a type of MLIR `Value`.
  if (auto *op = getIfOperation()) {
    // Print without verifying, since IR could be in an invalid state.
    op->print(llvm::errs(), mlir::OpPrintingFlags().printGenericOpForm());
    llvm::errs() << "\n";
  } else if (auto cv = getIfIRValue()) {
    cv.dump();
  } else if (auto witness = getIfWitness()) {
    llvm::errs() << "witness ";
    if (resolvedness < DeclResolvedness::signature) {
      llvm::errs() << "<unresolved>";
    } else {
      auto resolvedType = witness->getWitnessEntry();
      resolvedType.witnessName.print(llvm::errs());
      llvm::errs() << ": ";
      resolvedType.witnessType.print(llvm::errs());
    }
    llvm::errs() << "\n";
  } else {
    llvm::errs() << "<null decl>\n";
  }
}

ASTType ASTDecl::getIfTypeValue() const {
  if (auto cv = getIfIRValue().getIfPValue())
    return cv.getIfTypeValue();
  return {};
}

std::optional<StringRef> ASTDecl::getUserNameIfOperation() const {
  if (Operation *op = getIfOperation())
    if (auto decl = dyn_cast<ASTDeclInterface>(op))
      return decl.getDeclName().getValue();
  return {};
}

/// Return the SymbolRefAttr for a declaration, including all scoping that may
/// be needed, making it unique for every declaration.  This returns null for
/// named values that do not have a declaration.
SymbolRefAttr ASTDecl::getSymbolRef() const {
  if (auto traitType = dyn_cast_if_present<TraitType>(getIfTypeValue())) {
    auto reducedSymbols =
        LIT::reduceTraitCompositionSymbols(getShared(), traitType.getSymbols());
    // If this is a single trait, return it.
    if (reducedSymbols.size() == 1)
      return reducedSymbols[0].getSymbol();
    return {};
  }

  auto op = dyn_cast_if_present<mlir::SymbolOpInterface>(getIfOperation());
  if (!op)
    return {};
  assert((!isa<FnOp>(op) || resolvedness >= DeclResolvedness::signature) &&
         "Functions don't have a symbol until their signatures are resolved");
  return getFullyResolvedSymbolRef(op);
}

/// Given an MLIR op for a struct declaration, return the self type.
Type ASTDecl::computeSelfTypeForStruct(StructDeclOp structOp) {
  SmallVector<TypedAttr> parameters;
  for (auto decl : structOp.getParams()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    parameters.push_back(ParamDeclRefAttr::get(decl));
  }

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return structOp.bindReference(parameters);
}

Type ASTDecl::computeSelfTypeForTrait(TraitDeclOp traitOp) {
  // The last parameter to the trait is the 'T' parameter which (when everything
  // gets instantiated) resolves to the final type the trait is instantiated on.
  return ASTType(ParamDeclRefAttr::get(traitOp.getParamsAttr().back()));
}

void ASTDecl::findExtensionsInScopeForStruct(
    SymbolRefAttr targetStruct, llvm::SmallPtrSetImpl<ASTDecl *> &results,
    std::optional<TraitSymbolAttr> filterTrait) {
  if (!declsInScope)
    return;

  // Fast path: any scope that has extensions also registers them under the
  // aggregate name "extension:" (see the ExtensionDeclOp case in
  // SharedState::addDeclsForOp and the module-import paths in DeclResolver).
  // This is a hot conformance-check query and the vast majority of scopes have
  // no extensions at all, so bail out with a single pointer-keyed lookup on the
  // pre-interned marker before building any per-struct name.
  if (declsInScope->find(shared.extensionsScopeMarker) == declsInScope->end())
    return;

  // Extensions targeting this struct are registered under "extension:<leaf>".
  // The bucket is keyed by the target's leaf name only, so distinct structs
  // that share a leaf name land together and the exact-symbol check below still
  // filters them.
  SmallString<64> extensionName(shared.extensionsScopeMarker.getValue());
  extensionName += targetStruct.getLeafReference().getValue();
  auto it = declsInScope->find(StringAttr::get(getContext(), extensionName));
  if (it == declsInScope->end())
    return;

  for (ASTDecl *decl : it->second) {
    auto extOp = dyn_cast_or_null<ExtensionDeclOp>(decl->getIfOperation());
    if (!extOp || !extOp.getTargetStructAttr() ||
        extOp.getTargetStructAttr() != targetStruct)
      continue;

    // If no trait filter specified, add this extension.
    if (!filterTrait.has_value()) {
      results.insert(decl);
      continue;
    }

    // Extension doesn't have canonicalTrait computed yet - skip it. This
    // happens during error conditions or early parsing phases.
    if (!extOp.getCanonicalTrait())
      continue;

    // Use the extension's canonicalTrait (flattened hierarchy) to check
    // whether it implements the filter trait.
    for (TraitSymbolAttr symbol :
         extOp.getCanonicalTrait().value().getSymbols()) {
      if (symbol == filterTrait.value()) {
        results.insert(decl);
        break; // Found it, no need to check more symbols.
      }
    }
  }
}
