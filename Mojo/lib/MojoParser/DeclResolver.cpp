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
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/DeclResolver.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DocString.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "MojoUtils.h"
#include "ParserBase.h"
#include "Traits.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

#include <deque>

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// DiagnosticDeclContextChanger
//===----------------------------------------------------------------------===//

DeclResolver::DiagnosticDeclContextChanger::DiagnosticDeclContextChanger(
    ASTDecl *declToUse) {
  if (!declToUse)
    return;
  auto &shared = declToUse->getShared();
  resolver = &*shared.declResolver;
  prevDiagnosticDeclContext = resolver->diagnosticDeclContext;
  resolver->diagnosticDeclContext = declToUse;
}
DeclResolver::DiagnosticDeclContextChanger::~DiagnosticDeclContextChanger() {
  if (!resolver)
    return;
  resolver->diagnosticDeclContext = prevDiagnosticDeclContext;
}

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

// Declarations (e.g. module, class, function) are parsed in multiple phases
// to increase laziness of the parse as well as make circular references
// possible.
//
// This ensures that the forward references between peer declarations are
// handled correctly as well as circular references, for example in mutually
// recursive functions and code like this:
//
//   def foo():
//     def bar():
//       print(x)
//     x = 42
//     bar()
//   foo()

DeclResolver::DeclResolver(SharedState &state) : SharedStateUser(state) {}

DeclResolver::~DeclResolver() {
  // Run the destructors on all the ASTDecl objects to make sure any
  // transitively allocated data is released.
  for (ASTDecl *decl : parsedDeclList)
    decl->~ASTDecl();
}

//===----------------------------------------------------------------------===//
// Decl Constructors

ASTDecl &DeclResolver::addDecl(DeclIRValue irValue, SMLoc loc,
                               StringAttr baseName, ASTDecl *parentDecl,
                               LexerCursor cursor, LexerCursor endCursor,
                               ssize_t indentation) {
  ASTDecl &decl = createUnlistedDecl(irValue, loc, parentDecl, cursor,
                                     endCursor, indentation);
  // If this has a parent and a name, insert it into the parents name table so
  // name lookup will resolve it.  If it doesn't, then we're done.
  if (baseName)
    attachDeclToParentNameTable(&decl, baseName);
  return decl;
}

ASTDecl &DeclResolver::addBytecodeDecl(Operation *op, StringAttr baseName,
                                       ASTDecl *parentDecl,
                                       DeclResolvedness resolvedness) {
  ASTDecl &decl =
      addDecl(op, shared.diags.convertLocToSMLoc(op->getLoc()), baseName,
              parentDecl, LexerCursor(), LexerCursor(), /*indentation=*/-1);
  decl.loadedFromBytecode = true;
  decl.resolvedness = resolvedness;
  return decl;
}

ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal,
                                            StringAttr name, SMLoc loc,
                                            ASTDecl *parentDecl) {
  auto &decl =
      addDecl(declVal, loc, name, parentDecl, LexerCursor(), LexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::body;
  return decl;
}

ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal, StringRef name,
                                            llvm::SMLoc loc,
                                            ASTDecl *parentDecl) {
  return addFullyResolvedDecl(declVal, StringAttr::get(getContext(), name), loc,
                              parentDecl);
}

ASTDecl &DeclResolver::addErroneousDecl(StringRef baseName, llvm::SMLoc loc,
                                        ASTDecl *parentDecl, bool unlisted) {
  // Use a dummy attribute representation for the error.
  BoolAttr dummyAttr = BoolAttr::get(parentDecl->getContext(), true);
  if (unlisted) {
    ASTDecl &errDecl =
        createUnlistedDecl(PValue(dummyAttr), loc, parentDecl, LexerCursor(),
                           LexerCursor(), /*indentation=*/0);
    errDecl.resolvedness = DeclResolvedness::body;
    errDecl.setErroneous();
    return errDecl;
  }
  ASTDecl &errDecl =
      addFullyResolvedDecl(PValue(dummyAttr), baseName, loc, parentDecl);
  errDecl.setErroneous();
  return errDecl;
}

ASTDecl &DeclResolver::createUnlistedDecl(DeclIRValue irValue, SMLoc loc,
                                          ASTDecl *parentDecl,
                                          LexerCursor cursor,
                                          LexerCursor endCursor,
                                          ssize_t indentation) {
  ASTDecl *decl = shared.allocPersistent<ASTDecl>(
      shared, irValue, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this is a declaration which has a TypeCheckErrorType, then all
  // references to it are invalid.
  if (auto cv = decl->getIfIRValue())
    if (cv.getRValueType().isTypeCheckErrorType())
      decl->setErroneous();

  return *decl;
}

ASTDecl &DeclResolver::createUnlistedDecl(Operation *declOp, SMLoc loc,
                                          ASTDecl *parentDecl,
                                          LexerCursor cursor,
                                          LexerCursor endCursor,
                                          ssize_t indentation) {
  return createUnlistedDecl(DeclIRValue(declOp), loc, parentDecl, cursor,
                            endCursor, indentation);
}

// Check whether we're merging in a bunch of ASTDecls that could all contribute
// to a single struct's namespace.
// For example, say we have these imports:
//     from module_a import MyStruct  # imports a struct
//     from module_b import MyStruct  # imports an extension for it
//     from module_c import MyStruct  # imports an extension for it
// and the first two are resolved. That means that these three entries all
// coexist under the name "MyStruct":
// - struct ASTDecl for module_a's MyStruct struct
// - extension ASTDecl for module_b's MyStruct extension
// - unresolved import for module_c's MyStruct extension
// Basically, any entries for things that might contribute to a single
// struct's namespace is allowed to coexist.
// TODO(MOCO-522): Arcana doc mention on how multiple extensions and one
// struct and multiple imports can all coexist with the same name, because
// struct extensions are importable via their target struct's name.
static LogicalResult
canMergeSingleNamespaceDecls(ArrayRef<ASTDecl *> incoming,
                             ArrayRef<ASTDecl *> existing) {
  // Check if all declarations (both incoming and existing) could contribute
  // to a single struct's namespace.
  bool structFound = false;

  for (ASTDecl *decl : llvm::concat<ASTDecl *const>(existing, incoming)) {
    auto op = decl->getIfOperation();
    bool couldContribute = isa_and_nonnull<StructDeclOp>(op) ||
                           isa_and_nonnull<ExtensionDeclOp>(op) ||
                           isa_and_nonnull<UnresolvedImportOp>(op);
    if (!couldContribute)
      return failure();
    if (isa_and_nonnull<StructDeclOp>(op)) {
      if (structFound) {
        // User is trying to add a second struct with the same name, fail.
        return failure();
      }
      structFound = true;
    }
  }

  return success();
}

void DeclResolver::attachDeclToParentNameTable(ASTDecl *decl, StringAttr name) {
  ASTDecl *parentDecl = decl->getParentDecl();

  // Lazy allocate declsInScope.
  if (!parentDecl->declsInScope)
    parentDecl->declsInScope.reset(new ASTDecl::DeclInScopeType());

  // Remember the named decl in the symbol table so it can be looked up.
  TinyPtrVector<ASTDecl *> &entries = (*parentDecl->declsInScope)[name];

  // Function support method overloading on input arguments.  Variables and
  // types cannot be overloaded because they have no inputs.  Well, we could
  // actually allow type overloading on parameters theoretically to support
  // T[4] and T[1,7] as different things, but let's no proactively add
  // complexity.
  if (isa_and_nonnull<FnOp>(decl->getIfOperation())) {
    // Verify that all previous entries are also functions.  Note that we can't
    // check the overload set is compatible with each other because the
    // signatures aren't all resolved.
    for (ASTDecl *previous : entries) {
      if (!isa_and_nonnull<FnOp>(previous->getIfOperation())) {
        auto diag = emitError(decl->getLoc(), "invalid redefinition of ")
                    << name;
        diag.attachNote(*previous)
            << "cannot overload with this non-function definition";
        decl->setErroneous();
        previous->setErroneous();
        return;
      }
    }

    // Otherwise, we're good, charge forwards.
    entries.push_back(decl);
    // We don't uniquifyNameAndAddToParentSymbolTable here, that's done
    // elsewhere for functions.
    return;
  }

  // Structs and extensions can both have the same name in the same scope.
  // TODO(MOCO-522): Reference some arcana docs on this
  bool addingStruct = isa_and_nonnull<StructDeclOp>(decl->getIfOperation());
  bool addingExtension =
      isa_and_nonnull<ExtensionDeclOp>(decl->getIfOperation());
  if (addingStruct || addingExtension) {
    // Verify that all previous entries are also structs or extensions.  Note
    // that we can't check the overload set is compatible with each other
    // because the signatures aren't all resolved.
    for (ASTDecl *previous : entries) {
      bool previousIsStruct =
          isa_and_nonnull<StructDeclOp>(previous->getIfOperation());
      bool previousIsExtension =
          isa_and_nonnull<ExtensionDeclOp>(previous->getIfOperation());
      bool previousIsImport =
          isa_and_nonnull<UnresolvedImportOp>(previous->getIfOperation());
      bool previousNotStructRelated =
          !previousIsStruct && !previousIsExtension && !previousIsImport;
      // This checks that we're not giving e.g. a function and a struct the same
      // name.
      if (previousNotStructRelated) {
        auto diag = emitError(decl->getLoc(), "cannot define ")
                    << (addingStruct ? "a struct" : "an extension")
                    << " here with name " << name;
        diag.attachNote(*previous)
            << "conflicts with this previous declaration";
        decl->setErroneous();
        previous->setErroneous();
        return;
      }
      // Check for import vs local struct conflicts
      // An imported declaration cannot coexist with a locally defined struct
      // because the import cannot be an extension of a struct we're currently
      // defining. Search "#12090" for an example.
      // TODO(MOCO-522): This deserves an arcana doc and a few references to it.
      if (addingStruct && previousIsImport) {
        auto diag =
            emitError(decl->getLoc(), "cannot define a struct here with name ")
            << name;
        diag.attachNote(*previous)
            << "conflicts with this previous declaration";
        decl->setErroneous();
        previous->setErroneous();
        return;
      }
      // This makes sure we're not adding two structs with the same name.
      if (addingStruct && previousIsStruct) {
        auto diag = emitError(decl->getLoc(), "invalid redefinition of ")
                    << name;
        diag.attachNote(*previous)
            << "conflicts with this previous struct declaration";
        decl->setErroneous();
        previous->setErroneous();
        return;
      }
    }

    // Otherwise, we're good, charge forwards.
    entries.push_back(decl);

    assert(dyn_cast_or_null<mlir::SymbolOpInterface>(decl->getIfOperation()));
    registerDeclSymbol(decl);
    return;
  }

  // For any other type of declaration, check for conflicts
  if (!entries.empty()) {
    // Check if we are adding an identical unresolved import.
    auto op = decl->getIfOperation();
    if (auto import = dyn_cast_or_null<UnresolvedImportOp>(op)) {
      // First check for duplicate imports
      for (ASTDecl *existing : entries) {
        if (auto prevImportOp = dyn_cast_or_null<UnresolvedImportOp>(
                existing->getIfOperation())) {
          if (import.getModulePathAttr() == prevImportOp.getModulePathAttr() &&
              import.getDeclNameAttr() == prevImportOp.getDeclNameAttr()) {
            // This is a duplicate UnresolvedImportOp, just ignore it.
            return;
          }
        }
      }
      // TODO(MOCO-522): Arcana docs mention for decls sharing a namespace.
      if (succeeded(canMergeSingleNamespaceDecls({decl}, entries))) {
        entries.push_back(decl);
        return;
      }
    }

    // This is a genuine redefinition error
    ASTDecl *existing = entries.back();
    auto diag = emitError(decl->getLoc(), "invalid redefinition of ") << name;
    diag.attachNote(*existing) << "previous definition here";

    // Mark the existing decl and this one as erroneous so uses of either
    // don't create confusing errors.
    decl->setErroneous();
    for (ASTDecl *previous : entries)
      previous->setErroneous();
    return;
  }

  // This is the first declaration with this name
  entries.push_back(decl);

  // Register symbol with the parent symbol table.
  // Functions don't have symbols until they are fully resolved, but decls
  // inside functions cannot be accessed anyways.
  registerDeclSymbol(decl);
}

void DeclResolver::registerDeclSymbol(ASTDecl *decl) {
  if (auto symbolDecl =
          dyn_cast_or_null<mlir::SymbolOpInterface>(decl->getIfOperation())) {
    shared.uniquifyNameAndAddToParentSymbolTable(symbolDecl.getOperation());
    // This symbol may have been renamed by the above
    // uniquifyNameAndAddToParentSymbolTable call.
    SymbolRefAttr symbol = decl->getSymbolRef();
    // This shouldn't trip because we uniqued it in the above
    // uniquifyNameAndAddToParentSymbolTable call.
    assert(!declForTypeSymbol.count(symbol) && "Symbol redefinition/collision");
    declForTypeSymbol[symbol] = decl;
  }
}

void DeclResolver::aliasDeclInParent(ASTDecl *decl, StringAttr aliasName) {
  ASTDecl *parentDecl = decl->getParentDecl();

  // Lazy allocate declsInScope.
  if (!parentDecl->declsInScope)
    parentDecl->declsInScope.reset(new ASTDecl::DeclInScopeType());

  // Add the decl to the parent's name table under the name aliasName.
  TinyPtrVector<ASTDecl *> &entries = (*parentDecl->declsInScope)[aliasName];

  // TODO(MOCO-522): Linear, seems expensive, maybe we can change declsInScope
  // to something like a linked hash map?
  if (!llvm::is_contained(entries, decl))
    entries.push_back(decl);

  // Note: We intentionally do NOT call uniquifyNameAndAddToParentSymbolTable
  // because the extension is already in the symbol table under its primary name
}

TraitType DeclResolver::getCanonicalTrait(TraitType trait) {
  if (TraitType canonical = traitCanonicalizationCache.lookup(trait))
    return canonical;
  SmallVector<TraitSymbolAttr> symbols(trait.getSymbols());
  DenseMap<TraitSymbolAttr, ConstraintAttr> constraintMap;
  if (trait.hasConstraints())
    for (auto [symbol, constraint] :
         llvm::zip_equal(symbols, trait.getConstraints()))
      constraintMap[symbol] = constraint;

  return traitCanonicalizationCache[trait] =
             getCanonicalTrait(symbols, constraintMap);
}

TraitType DeclResolver::getCanonicalTrait(
    SmallVectorImpl<TraitSymbolAttr> &symbols,
    const DenseMap<TraitSymbolAttr, ConstraintAttr> &constraintMap) {
  SmallVector<ConstraintAttr> constraints = {};
  if (!symbols.empty()) {
    constraints =
        canonicalizeTraitSymbolsAndConstraints(shared, symbols, constraintMap);
  }
  return TraitType::get(getContext(), symbols, constraints);
}

void DeclResolver::attachDeclToTraitCompositionDecl(
    ASTDecl *traitDecl, TraitSymbolAttr witnessFor,
    SmallVector<ASTDecl *> &&childDecl, StringAttr name) {
  // Lazy allocate declsInScope.
  if (!traitDecl->declsInScope)
    traitDecl->declsInScope.reset(new ASTDecl::DeclInScopeType());

  auto loc = childDecl.front()->getLoc(); // Just use the first location.
  ASTDecl *witnessDecl = &createUnlistedDecl(
      DeclIRValue(WitnessDecl{std::move(childDecl), witnessFor}), loc,
      /*parentDecl=*/traitDecl, LexerCursor(), LexerCursor(),
      /*indentation=*/-1);
  witnessDecl->resolvedness = DeclResolvedness::unparsed;

  (*traitDecl->declsInScope)[name].push_back(witnessDecl);
}

//===----------------------------------------------------------------------===//
// Import Resolution

void DeclResolver::aliasDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                              llvm::SMLoc aliasLoc, ASTDecl &context) {
  (void)aliasDeclsImpl(decls, name, aliasLoc, context);
}

LogicalResult DeclResolver::tryAliasDecls(ArrayRef<ASTDecl *> decls,
                                          StringAttr name, llvm::SMLoc aliasLoc,
                                          ASTDecl &context) {
  return aliasDeclsImpl(decls, name, aliasLoc, context,
                        /*emitDiagnostics=*/false);
}

LogicalResult
DeclResolver::aliasImportDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                               StringAttr declName, ImportPathAttr moduleName,
                               llvm::SMLoc aliasLoc, ASTDecl &context,
                               bool allowMultipleWithSameName) {
  return aliasDeclsImpl(decls, name, aliasLoc, context,
                        /*emitDiagnostics=*/true, moduleName, declName,
                        allowMultipleWithSameName);
}

// Check whether the incoming decls conflict with existing decls under the same
// name, applying the same rules as `attachDeclToParentNameTable` does for local
// declarations: functions may overload each other (deprecated when they come
// from different origins), but a function and a non-function (struct, alias,
// MLIR type, …) under the same name is an error, as are two distinct
// non-functions.
LogicalResult DeclResolver::checkImportNamingConflict(
    ArrayRef<ASTDecl *> incoming, ArrayRef<ASTDecl *> existing, StringAttr name,
    llvm::SMLoc aliasLoc, bool emitDiagnostics) {
  // Single pass over each set to find a representative def and non-def decl
  // (skipping UnresolvedImportOps whose type is not yet known).
  //
  // By module naming rules, each set has at most one non-function element
  // (a module cannot declare e.g. both a struct and an alias under the same
  // name). This lets us classify both sets in O(N+M) and dispatch directly,
  // avoiding an O(N*M) nested loop over two potentially large overload sets.
  struct DeclKinds {
    ASTDecl *fn = nullptr, *nonFn = nullptr;
  };
  auto classify = [](ArrayRef<ASTDecl *> decls,
                     ASTDecl *skip = nullptr) -> DeclKinds {
    DeclKinds result;
    for (ASTDecl *d : decls) {
      if (isa_and_nonnull<UnresolvedImportOp>(d->getIfOperation()))
        continue;
      if (d == skip)
        continue;
      if (isa_and_nonnull<FnOp>(d->getIfOperation())) {
        if (!result
                 .fn) // any representative def suffices for conflict detection
          result.fn = d;
      } else {
        result.nonFn = d; // at most one non-def per set (module naming rules)
      }
    }
    return result;
  };

  auto [incomingFn, incomingNonFn] = classify(incoming);
  // Skip incomingNonFn when scanning the existing set: if the user wrote
  // `from mod_a import Foo` twice, the first resolution already placed the
  // struct into `existing`, so the same ASTDecl* appears in both arrays.
  // Without the skip, we would compare the decl against itself and
  // incorrectly diagnose a "struct vs. struct" conflict.
  auto [existingFn, existingNonFn] = classify(existing, incomingNonFn);

  // Determine the conflicting pair, if any.
  ASTDecl *conflictA = nullptr, *conflictB = nullptr;
  if (incomingNonFn && existingFn) {
    // Non-function being imported conflicts with an existing function.
    conflictA = existingFn;
    conflictB = incomingNonFn;
  } else if (incomingFn && existingNonFn) {
    // Function being imported conflicts with an existing non-function.
    conflictA = existingNonFn;
    conflictB = incomingFn;
  } else if (incomingNonFn && existingNonFn) {
    // Two non-functions: compatible only if they form a single struct namespace
    // (one struct + its extensions). Two structs, two aliases, etc. conflict.
    if (failed(
            canMergeSingleNamespaceDecls({incomingNonFn}, {existingNonFn}))) {
      conflictA = existingNonFn;
      conflictB = incomingNonFn;
    }
  } else if (incomingFn && existingFn && incomingFn != existingFn) {
    // An overload set resolves from one place only.
    conflictA = existingFn;
    conflictB = incomingFn;
  }

  if (!conflictA)
    return success();

  if (emitDiagnostics) {
    auto diag = emitError(aliasLoc, "import of ") << name << " is ambiguous";
    diag.attachNote(*conflictA) << name << " declared here";
    diag.attachNote(*conflictB) << name << " also declared here";
  }
  return failure();
}

LogicalResult DeclResolver::aliasDeclsImpl(
    ArrayRef<ASTDecl *> decls, StringAttr name, llvm::SMLoc aliasLoc,
    ASTDecl &context, bool emitDiagnostics, ImportPathAttr modulePath,
    StringAttr declNameInModule, bool allowMultipleWithSameName) {
  // Check to see if the decl is an import. We create new decls within the
  // context for these instead of aliasing, because import decls lazily replace
  // themselves with new decls (depending on what gets imported). That
  // replacement is only known when the import decl is referenced (and thus
  // resolved), so we can't alias the import directly.
  ASTDecl *frontDecl = decls.front();
  if (auto importOp =
          dyn_cast_or_null<UnresolvedImportOp>(frontDecl->getIfOperation())) {
    // If the import is overlapping with an existing declaration, let it slide.
    // FIXME: This is assuming that the import would resolve to the same decl.
    if (ArrayRef<ASTDecl *> decls = context.lookupInCurrentScope(name);
        !decls.empty())
      return success();

    ASTDecl &importDecl = addDecl(
        frontDecl->getIfOperation(), frontDecl->getLoc(), name, &context,
        frontDecl->getCursor(), frontDecl->getCursor(), /*indentation=*/-1);
    return success(!importDecl.isErroneous());
  }

  // Lazy allocate declsInScope.
  if (!context.declsInScope)
    context.declsInScope.reset(new ASTDecl::DeclInScopeType());

  auto [it, inserted] =
      context.declsInScope->insert({name, TinyPtrVector<ASTDecl *>(decls)});
  // It succeeded and there was nothing in this scope by that name already, so
  // we're done.
  if (inserted)
    return success();
  // If we got here, it failed, there's already entries here by that name.
  TinyPtrVector<ASTDecl *> &entries = it->second;

  // If we get here, then we've hit an overlap. This is likely because we're
  // seeing the import statement that's already here, and it's conflicting with
  // the new entries we're bringing in.
  // Check to see if that's the case, and if so, replace the unresolved import
  // with the real decls from the target module.
  //
  // The `modulePath` argument tells us which module we're importing from, and
  // is only present when we're resolving an import (not just creating an
  // alias).
  // Here, we look for that import.
  // TODO(MOCO-522): This seems weird. This function shouldn't be making
  // assumptions about what modulePath's existence means. Possibly rename
  // modulePath or find some better way to represent this, or last resort, make
  // it some arcana.
  if (modulePath) {
    // Find and remove all matching imports (in case of duplicate imports).
    // Keep in mind, the user may have imported the module twice, so we have to
    // remove all matching imports (see test MSWGHRI).
    bool foundMatchingImport = false;
    for (int i = entries.size() - 1; i >= 0; --i) {
      if (auto importOp = dyn_cast_or_null<UnresolvedImportOp>(
              entries[i]->getIfOperation());
          importOp && importOp.getModulePathAttr() == modulePath &&
          importOp.getDeclNameAttr() == declNameInModule) {
        // Mark the import we're replacing as resolved in case anyone sees it
        // (which would be weird, since we're about to remove it, but just in
        // case).
        entries[i]->resolvedness = DeclResolvedness::body;
        // Remove this matching import. We'll replace it with the real decls
        // further below.
        entries.erase(entries.begin() + i);
        foundMatchingImport = true;
      }
    }

    // TODO(MOCO-522): It feels like this function is doing a few too many
    // things in too many odd cases, should split and revisit this abstraction.
    bool shouldAdd = false;
    if (foundMatchingImport) {
      // Sure enough, we found an importOp that matches the module and decl
      // name, let's replace the import with the real decls.
      if (failed(checkImportNamingConflict(decls, entries, name, aliasLoc,
                                           emitDiagnostics)))
        return failure();
      shouldAdd = true;
    } else {
      // No placeholder was removed, this can happen if someone is calling
      // aliasDeclsImpl for things that were already imported, for example if
      // we're importing a bunch of extensions when we've already imported them
      // in the past.
      // Now, check if the new decls can coexist with the existing ones.
      if (allowMultipleWithSameName)
        if (succeeded(canMergeSingleNamespaceDecls(decls, entries)))
          shouldAdd = true;
    }
    if (shouldAdd) {
      // Add new decls, avoiding duplicates.
      // TODO(MOCO-522): Quadratic loop, maybe we can change declsInScope to
      // something like a linked hash map?
      for (ASTDecl *decl : decls)
        if (!llvm::is_contained(entries, decl))
          entries.push_back(decl);
    }

    return success();
  }

  // TODO(MOCO-522): Arcana docs mention for decls sharing a namespace.
  if (succeeded(canMergeSingleNamespaceDecls(decls, entries))) {
    // Add new decls, avoiding duplicates.
    // TODO(MOCO-522): Quadratic loop, maybe we can change declsInScope to
    // something like a linked hash map?
    for (ASTDecl *decl : decls)
      if (!llvm::is_contained(entries, decl))
        entries.push_back(decl);
    return success();
  }

  ASTDecl *existing = entries.back();

  // If the decls are functions, try to merge them into the existing set.
  if (isa_and_nonnull<FnOp>(frontDecl->getIfOperation()) &&
      isa_and_nonnull<FnOp>(existing->getIfOperation())) {
    // Check that none of the decls are already in the set.
    auto canMergeDecl = [&](ASTDecl *decl) {
      FnOp declOp = cast<FnOp>(decl->getIfOperation());
      return llvm::all_of(entries, [&](ASTDecl *existing) {
        if (failed(resolve(*existing, DeclResolvedness::signature, aliasLoc)))
          return false;
        FnOp existingOp = cast<FnOp>(existing->getIfOperation());

        FnTypeGeneratorType declSignature = declOp.getFullSignature();
        FnTypeGeneratorType existingSignature = existingOp.getFullSignature();
        // If the argument types match exactly *and* the parameter
        // types match exactly, then we don't want to merge this decl into the
        // set. We also need to remove the by-ref result type from the
        // input types, so that aliasing is strictly based on the actual
        // inputs.
        auto getActualArgs =
            [](FnTypeGeneratorType signature) -> ArrayRef<Type> {
          ArrayRef<Type> inputTypes = signature.getArguments();
          // Drop the trailing result slots. Memory-only functions and throwing
          // functions each add a result slot.
          inputTypes = inputTypes.drop_back(signature.hasMemoryOnlyResult() +
                                            signature.isThrows());
          return inputTypes;
        };

        if (getActualArgs(declSignature) == getActualArgs(existingSignature) &&
            declSignature.getInputParamTypes() ==
                existingSignature.getInputParamTypes())
          return false;

        // We can merge the decl into the set.
        return true;
      });
    };
    if (llvm::all_of(decls, canMergeDecl)) {
      // We don't have to check for duplicates here because canMergeDecl
      // already detects duplicates.
      for (ASTDecl *decl : decls)
        entries.push_back(decl);
      return success();
    }
  }

  // Rejecting overlap is conservative and not what python does, but we can
  // relax this in the future when we know what the right policy should be.
  if (emitDiagnostics) {
    auto diag = emitError(aliasLoc, "invalid redefinition of ") << name;
    diag.attachNote(*existing) << "previous definition here";

    for (ASTDecl *previous : it->second)
      previous->setErroneous();
  }
  return failure();
}

ASTDecl &DeclResolver::createImportOp(ASTDecl &dest, mlir::OpBuilder &builder,
                                      StringAttr name,
                                      ImportPathAttr modulePath,
                                      mlir::Location loc) {
  auto importOp = ImportOp::create(builder, loc,
                                   /*sym_name=*/name, modulePath);
  SMLoc smloc = shared.diags.convertLocToSMLoc(loc);
  ASTDecl &importDecl =
      addDecl(static_cast<Operation *>(importOp), smloc, name, &dest,
              LexerCursor(), LexerCursor(), /*indentation=*/-1);
  // ImportOp has no body to parse — mark as fully resolved so that name
  // lookup through parent scopes doesn't trip the resolvedness assertion.
  importDecl.resolvedness = DeclResolvedness::body;
  return importDecl;
}

ImportPathAttr DeclResolver::getAbsoluteModuleName(ASTDecl &moduleDecl) {
  SymbolRefAttr ref = moduleDecl.getSymbolRef();
  if (!ref)
    return {};
  SmallVector<StringRef> components;
  components.push_back(ref.getRootReference().getValue());
  for (FlatSymbolRefAttr nested : ref.getNestedReferences())
    components.push_back(nested.getValue());
  return ImportPathAttr::get(getContext(), /*relativeLevel=*/0, components);
}

FailureOr<ASTDecl *> DeclResolver::bodyResolvePackageInit(ASTDecl &package,
                                                          SMLoc loc) {
  // Not a package — nothing to do.
  auto packageOp = dyn_cast_or_null<PackageOp>(package.getIfOperation());
  if (!packageOp)
    return nullptr;
  // The package's scope is empty; its __init__ holds the package's symbols.
  if (!shared.hasNestedModule(packageOp, "__init__"))
    return nullptr;
  ASTDecl &initDecl = shared.importModule(
      SharedState::ImportPath({"__init__"}, /*relativeLevel=*/1), packageOp,
      loc);
  if (initDecl.isErroneous())
    return nullptr;
  // If __init__'s body is itself mid-resolution - e.g. we are resolving a
  // statement that lives inside __init__ - do not recurse into it. Return null
  // so the caller treats the package scope as empty and falls back to
  // resolving the name as a submodule from the filesystem.
  //
  // Note: this guards the case where __init__'s body is still resolving. The
  // related guard in importDeclFromModule handles the narrower lazy case where
  // __init__'s body is fully resolved but a specific `from . import sub`
  // binding within it is the import currently in flight.
  if (isAlreadyProcessing(initDecl))
    return nullptr;
  if (failed(resolveBody(initDecl, loc)))
    return failure();
  return &initDecl;
}

namespace {

/// Walks a prebuilt standard-library package tree to find the unique package
/// whose public surface exposes a queried name, for the missing-import
/// suggestion. See DeclResolver::findUniqueStdlibImportFor for the full
/// specification of what is and isn't resolved.
class StdlibImportSearch {
public:
  StdlibImportSearch(SharedState &shared, StringRef name)
      : shared(shared), name(name),
        initName(StringAttr::get(shared.getContext(), "__init__")) {}

  /// Walk the tree rooted at `stdPackage` and return the package path to
  /// suggest for `name`, or nullopt if nothing matches or the matches are a
  /// genuine ambiguity (see `uniqueImportPath`).
  std::optional<std::string> run(PackageOp stdPackage) {
    CompilerTimeTraceScope timeScope("findUniqueStdlibImportFor.walk");
    Worklist worklist;
    worklist.emplace_back(stdPackage,
                          stdPackage.getSymNameAttr().getValue().str());
    while (!worklist.empty()) {
      auto [packageOp, packagePath] = worklist.pop_back_val();
      visitPackage(packageOp, packagePath, worklist);
    }
    return uniqueImportPath();
  }

private:
  using Worklist = SmallVector<std::pair<PackageOp, std::string>>;

  /// How a package's `__init__` surfaces `name`. `Native` = the package owns
  /// it: declared in `__init__`, re-exported from the package's own subtree, or
  /// pulled in by a relative wildcard. `Foreign` = the package only re-exports
  /// another package's symbol (a convenience aggregator, e.g. `std.ffi`
  /// re-exporting `std.os.abort`). Native owners are preferred over foreign
  /// re-exporters when choosing what to suggest (see `uniqueImportPath`).
  enum class MatchKind { Native, Foreign };

  /// A package's direct children, split for the surface check: its `__init__`
  /// module (the package's own surface) and its other child modules/packages
  /// keyed by leaf name (a `from .X import *` can name one of these as `X`).
  /// `__init__` is the package's own surface, never named by a wildcard, so it
  /// is kept out of `byLeaf`. `byLeaf` holds a mix of `FileModuleOp`s and
  /// `PackageOp`s, so its values stay the common `Operation *`.
  struct PackageChildren {
    FileModuleOp initModule = nullptr;
    llvm::StringMap<Operation *> byLeaf;
  };

  // Materialization convention: every method that iterates an op's regions
  // (indexChildren, initExposesName, moduleExposesName, findInit) materializes
  // that op first; callers do not pre-materialize. `materialize` is a cheap
  // no-op once an op is materialized, so the overlapping calls cost nothing.

  /// Inspect one package: record a match if its `__init__` exposes `name`,
  /// tagged native vs foreign so `uniqueImportPath` can prefer the owner.
  void visitPackage(PackageOp packageOp, StringRef packagePath,
                    Worklist &worklist) {
    PackageChildren children = indexChildren(packageOp, packagePath, worklist);
    if (children.initModule)
      if (std::optional<MatchKind> kind = initExposesName(
              children.initModule, children.byLeaf, packagePath))
        recordMatch(packagePath, *kind);
  }

  /// Index `packageOp`'s direct children (see `PackageChildren`) and, as a side
  /// effect, enqueue its sub-packages onto `worklist` for the walk to visit.
  PackageChildren indexChildren(PackageOp packageOp, StringRef packagePath,
                                Worklist &worklist) {
    materialize(packageOp);
    PackageChildren children;
    for (Region &region : packageOp->getRegions())
      for (Operation &op : region.getOps()) {
        auto fileModule = dyn_cast<FileModuleOp>(&op);
        auto subPackage = dyn_cast<PackageOp>(&op);
        if (!fileModule && !subPackage)
          continue;
        StringRef leaf = fileModule ? fileModule.getSymNameAttr().getValue()
                                    : subPackage.getSymNameAttr().getValue();
        if (fileModule && isInitModule(fileModule)) {
          children.initModule = fileModule;
        } else {
          // Leaf names are unique among a package's children
          assert(!children.byLeaf.contains(leaf) &&
                 "duplicate leaf name among a package's children");
          children.byLeaf[leaf] = &op;
        }
        if (subPackage)
          worklist.emplace_back(subPackage,
                                (packagePath + "." + leaf.str()).str());
      }
    return children;
  }

  /// Does a package's `__init__` expose `name` — via a direct declaration or a
  /// re-export — and if so, how? Returns `Native` when the name is owned here
  /// (declared, re-exported from the package's own subtree, or pulled in by a
  /// relative wildcard), `Foreign` when it is only a convenience re-export of
  /// another package's symbol, or nullopt when not exposed. A native exposure
  /// outranks a foreign one within the same `__init__`.
  std::optional<MatchKind>
  initExposesName(FileModuleOp initModule,
                  const llvm::StringMap<Operation *> &childByLeaf,
                  StringRef packagePath) {
    materialize(initModule);
    bool sawForeign = false;
    for (Region &region : initModule->getRegions()) {
      for (Operation &op : region.getOps()) {
        if (auto wild = dyn_cast<UnresolvedWildcardImportOp>(&op)) {
          // Only single-component relative wildcards (`from .X import *`) are
          // supported. Multi-component (`.a.b`), absolute (`pkg.foo`), and bare
          // (`.`) forms are skipped here: they name modules that are not direct
          // children, and matching them by leaf name alone would false-match an
          // unrelated direct child that happens to share the leaf.
          StringRef singleComponent =
              singleComponentRelativeChild(wild.getModulePathAttr());
          if (singleComponent.empty())
            continue;
          auto it = childByLeaf.find(singleComponent);
          // A relative wildcard pulls from the package's own child, so a hit
          // means the name is owned here: native.
          if (it != childByLeaf.end() && wildcardSourceExposesName(it->second))
            return MatchKind::Native;
        } else if (std::optional<MatchKind> kind =
                       classifyOpExposure(op, packagePath)) {
          if (*kind == MatchKind::Native)
            return MatchKind::Native;
          sawForeign = true;
        }
      }
    }
    return sawForeign ? std::optional(MatchKind::Foreign) : std::nullopt;
  }

  /// True if the `from .X import *` imports `name`.
  /// Checks `X` itself if it is a module, or
  ///  `X`'s `__init__` if it is a package.
  /// Does not follow wildcards inside `X` (one level deep).
  bool wildcardSourceExposesName(Operation *sourceOp) {
    if (auto fileModule = dyn_cast<FileModuleOp>(sourceOp))
      return moduleExposesName(fileModule);
    if (auto subPackage = dyn_cast<PackageOp>(sourceOp))
      if (FileModuleOp init = findInit(subPackage))
        return moduleExposesName(init);
    return false;
  }

  /// Scan a module's ops for a direct declaration or explicit re-export of
  /// `name`; wildcard re-exports inside the module are ignored.
  bool moduleExposesName(FileModuleOp modOp) {
    materialize(modOp);
    for (Region &region : modOp->getRegions())
      for (Operation &op : region.getOps())
        if (opExposesName(op))
          return true;
    return false;
  }

  /// True if a single op is an explicit re-export or a direct declaration of
  /// `name` (wildcard re-exports are handled by `initExposesName`). Used to
  /// scan a wildcard's target module, where only exposure (not native/foreign)
  /// is needed.
  bool opExposesName(Operation &op) const {
    if (auto imp = dyn_cast<ImportOp>(&op))
      return imp.getSymNameAttr().getValue() == name;
    if (auto unresolved = dyn_cast<UnresolvedImportOp>(&op))
      return unresolved.getImportNameAttr().getValue() == name;
    return declMatches(&op);
  }

  /// Like `opExposesName`, but for a package `__init__` op: also classifies the
  /// exposure. A direct declaration is native. A re-export is native when its
  /// source module lives in `packagePath`'s own subtree and foreign when it
  /// re-exports another package's symbol. Returns nullopt when `op` does not
  /// expose `name`.
  std::optional<MatchKind> classifyOpExposure(Operation &op,
                                              StringRef packagePath) const {
    if (auto imp = dyn_cast<ImportOp>(&op)) {
      if (imp.getSymNameAttr().getValue() != name)
        return std::nullopt;
      return sourceKind(imp.getModulePathAttr(), packagePath);
    }
    if (auto unresolved = dyn_cast<UnresolvedImportOp>(&op)) {
      if (unresolved.getImportNameAttr().getValue() != name)
        return std::nullopt;
      return sourceKind(unresolved.getModulePathAttr(), packagePath);
    }
    return declMatches(&op) ? std::optional(MatchKind::Native) : std::nullopt;
  }

  /// Classify a re-export by where its source module lives relative to the
  /// re-exporting package: within the package's own subtree — a relative import
  /// (`.x`), or an absolute path equal to or under `packagePath` — is native;
  /// anything else re-exports a different package's symbol and is foreign.
  MatchKind sourceKind(ImportPathAttr moduleName, StringRef packagePath) const {
    if (moduleName.getRelativeLevel() > 0)
      return MatchKind::Native;
    std::string dottedName =
        SharedState::ImportPath::fromAttr(moduleName).toDottedString();
    return isDottedPrefix(packagePath, dottedName) ? MatchKind::Native
                                                   : MatchKind::Foreign;
  }

  /// True if `op` is a top-level fn/struct/trait/alias declaration of `name`.
  /// `getDeclName()` yields the user-visible (demangled) name for each kind.
  bool declMatches(Operation *op) const {
    if (!isa<FnOp, StructDeclOp, TraitDeclOp, AliasDeclOp>(op))
      return false;
    return cast<ASTDeclInterface>(op).getDeclName().getValue() == name;
  }

  /// The `__init__` file-module child of a package, or null.
  FileModuleOp findInit(PackageOp packageOp) {
    materialize(packageOp);
    for (Region &region : packageOp->getRegions())
      for (Operation &op : region.getOps())
        if (auto fileModule = dyn_cast<FileModuleOp>(&op))
          if (isInitModule(fileModule))
            return fileModule;
    return {};
  }

  /// True if `fileModule` is a package's `__init__` (its public surface).
  bool isInitModule(FileModuleOp fileModule) const {
    return fileModule.getSymNameAttr() == initName;
  }

  void materialize(Operation *op) { shared.materializePrecompiledStdlibOp(op); }

  void recordMatch(StringRef modulePath, MatchKind kind) {
    (kind == MatchKind::Native ? nativeMatches : foreignMatches)
        .push_back(modulePath.str());
  }

  /// Reduce the collected matches to a single import path to suggest, or
  /// nullopt. Packages that own `name` natively outrank packages that only
  /// re-export it from elsewhere (a convenience aggregator like `std.ffi`), so
  /// a foreign re-export never wins when a native owner exists. Within the
  /// chosen set: a lone match is suggested directly; several matches are
  /// suggestable only if they lie on one ancestor->descendant chain (each path
  /// a dotted prefix of the next), which is what an upward re-export through a
  /// package subtree produces (a symbol defined in `std.a.b.c` and re-exported
  /// by `std.a.b` and `std.a`) — then the shortest (outermost, most public)
  /// path is the canonical import. Matches on diverging branches (two siblings)
  /// are a genuine ambiguity that we cannot resolve, and yield nullopt.
  std::optional<std::string> uniqueImportPath() {
    SmallVectorImpl<std::string> &candidates =
        !nativeMatches.empty() ? nativeMatches : foreignMatches;
    if (candidates.empty())
      return std::nullopt;
    // Order by depth (then lexicographically, for a deterministic result). A
    // single chain is then exactly the case where each path is a dotted prefix
    // of the next, and `candidates.front()` is the shortest.
    llvm::sort(candidates, [](StringRef a, StringRef b) {
      return std::make_tuple(a.count('.'), a) <
             std::make_tuple(b.count('.'), b);
    });
    for (size_t i = 1, e = candidates.size(); i < e; ++i)
      if (!isDottedPrefix(candidates[i - 1], candidates[i]))
        return std::nullopt;
    return candidates.front();
  }

  /// True if `prefix` is a component-wise (dotted) prefix of `path`: equal, or
  /// `path` continues with a `.` immediately after `prefix` (so `std.mem` is
  /// not a prefix of `std.memory`, but `std.a` is a prefix of `std.a.b`).
  static bool isDottedPrefix(StringRef prefix, StringRef path) {
    if (path == prefix)
      return true;
    return path.starts_with(prefix) && path.size() > prefix.size() &&
           path[prefix.size()] == '.';
  }

  /// If `moduleName` is a single-component relative reference (`.X`, exactly
  /// one leading dot and one component), return the component `X`; otherwise
  /// the empty string. Rejects multi-component (`.a.b`), absolute (`a.b`), and
  /// bare (`.`) forms so they are never matched against direct children by
  /// leaf name.
  static StringRef singleComponentRelativeChild(ImportPathAttr moduleName) {
    if (moduleName.getRelativeLevel() != 1 ||
        moduleName.getComponents().size() != 1)
      return {};
    return moduleName.getComponents().front().getValue();
  }

  SharedState &shared;
  StringRef name;
  StringAttr initName;
  /// Dotted paths of packages whose `__init__` surface exposes `name`, split by
  /// how they expose it (see `MatchKind`). Collected during the walk and
  /// reduced by `uniqueImportPath`, which prefers native owners over foreign
  /// re-exporters.
  SmallVector<std::string> nativeMatches;
  SmallVector<std::string> foreignMatches;
};

} // namespace

// Suggest the standard-library import for an unresolved unqualified `name`:
// finds the "canonical" (see below) std *package* whose public surface includes
// `name` and returns its dotted path (e.g. "std.memory") for a "did you mean to
// import ..." diagnostic, or nullopt when there is no canonical package. Runs
// on the error path of an already-failing compile, and is not cached (the tree
// is re-walked once per unresolved name).
//
// "Canonical" means: among the packages that expose `name`, prefer the ones
// that own it *natively* (declare it, re-export it from their own subtree, or
// pull it in via a relative wildcard) over packages that merely re-export it
// from another package (a convenience aggregator like `std.ffi` re-exporting
// `std.os.abort`). Then, within the preferred set, the canonical package is
// either (1) the only one, or (2) when several lie on one ancestor->descendant
// chain, the shortest (most public) path. When the preferred set still has
// packages on diverging branches, there is no canonical package and no
// suggestion is made (see the "Ambiguity" case below).
//
// A package's public surface is whatever its `__init__.mojo` exposes. The walk
// visits each std package and tests that surface for `name`. This is the
// specification of which forms of exposure are recognized; each lists the
// scenario, an example, and what the implementation does with it.
//
// Supported (a match produces the suggestion):
//
//   - Direct declaration in the package `__init__`.
//       `def name(): ...` in pkg/__init__.mojo          -> suggests `pkg`
//     Impl: reads the decl name off the op and compares it to `name`.
//
//   - Explicit re-export in the package `__init__`.
//       `from .sub import name` in pkg/__init__.mojo     -> suggests `pkg`
//     Impl: whether this op matches turns only on the bound import name; the
//     re-exported module's path then decides native vs foreign (see the "Name
//     defined ... re-exported from another" case below).
//
//   - Single-component relative wildcard re-export in the package `__init__`.
//       `from .sub import *`, `name` declared in `sub`   -> suggests `pkg`
//     Impl: maps the wildcard's leaf (`sub`) to a direct child module (or
//     sub-package) of the package, then scans that child's surface for `name`.
//
//   - Name defined in one package and re-exported from another.
//       `abort` is defined in `std.os`, and std/ffi/__init__.mojo also
//       re-exports it via `from std.os import abort`       -> suggests `std.os`
//     The package that defines the name (native) wins and is suggested, over a
//     package that just re-exports it (foreign). A re-export still counts as
//     "defining" when its source is inside the package's own subtree, so a
//     parent re-exporting from its own child (`std.a` with `from .b import
//     name` or `from std.a.b import name`) is native to `std.a` — not foreign.
//     Only re-exporting from a *different* package (`std.ffi` reaching into
//     `std.os`) is foreign. If a parent and its child both define the name, the
//     shortest wins (the ancestor->descendant chain case above); if only
//     re-exporters expose it, they are the fallback — still a suggestion.
//
//     Impl: `classifyOpExposure` marks a re-export foreign when its source
//     module is outside the package's own subtree; `uniqueImportPath` drops the
//     foreign matches whenever any native match exists.
//
// Not supported (no suggestion; `name` is simply not found):
//
//   - Source-built `std` (stdlib developers only).
//     Impl: bails immediately. A source std materializes its submodules lazily,
//     so they are not all present in the IR and the structural walk would see
//     an incomplete tree. A source-specific traversal is intentionally not
//     built — significant complexity for a path end users never hit.
//
//   - Multi-component relative wildcard re-export.
//       `from .sub.deeper import *`
//     Impl: rejected up front by `singleComponentRelativeChild`; only
//     single-component relative wildcards (`from .X import *`) are resolved.
//     Matching `.sub.deeper` by its leaf (`deeper`) alone would false-match an
//     unrelated direct child that happens to be named `deeper`.
//
//   - Absolute wildcard re-export.
//       `from std.foo import *`
//     Impl: same guard — an absolute name has no leading dot, so it is not a
//     single-component relative wildcard and is skipped (avoiding the same
//     leaf-collision false match as above).
//
//   - Nested (transitive) wildcard re-export.
//       `from .sub import *` where `sub` itself does `from .deeper import *`
//     Impl: wildcard resolution is one level deep; it scans `sub` directly but
//     does not follow a wildcard found inside `sub`.
//
//   - Ambiguity: packages on diverging branches expose `name`.
//     Impl: within the preferred (native, else foreign) set, `uniqueImportPath`
//     suggests the shortest only if the packages form a single
//     ancestor->descendant chain (the same name re-exported upward through one
//     subtree). Packages on diverging branches (e.g. two sibling owners, like
//     `std.subprocess.run` vs `std.benchmark.run`) cannot be disambiguated and
//     are suppressed.
//
//   - Private query: `name` is underscore-prefixed.
//     Impl: rejected up front, before the walk.
std::optional<std::string>
DeclResolver::findUniqueStdlibImportFor(StringRef name) {
  // Private query names are never suggested (see header).
  if (name.empty() || isInternalName(name))
    return std::nullopt;

  // Bytecode `std` only; null for a source build (see header).
  PackageOp stdPackage = shared.getPrecompiledStdlibPackage();
  if (!stdPackage)
    return std::nullopt;
  return StdlibImportSearch(shared, name).run(stdPackage);
}

LogicalResult DeclResolver::importDeclFromModule(
    ASTDecl &dest, PackageOp currentPackage, ImportPathAttr moduleName,
    StringAttr sourceName, StringAttr destName, SMLoc loc, SMLoc sourceNameLoc,
    SMLoc destNameLoc, bool resolveTarget) {

  auto modulePath = SharedState::ImportPath::fromAttr(moduleName);
  ASTDecl &module = shared.importModule(modulePath, currentPackage, loc);
  shared.notifyListenerOnModuleImport(module, modulePath, loc);

  // A relative self-import written inside a package's __init__ (`from . import
  // sub`) would, via the package->__init__ redirect, look up the submodule in
  // the very __init__ we are currently resolving and find this same import - a
  // spurious self recursion.
  //
  // This is the lazy-import counterpart to bodyResolvePackageInit's
  // `isAlreadyProcessing(initDecl)` guard: that one fires while __init__'s body
  // is still resolving; this one fires once the body is resolved but a specific
  // re-export binding inside it is being resolved on demand.
  bool selfReferential = false;
  if (auto initOrFail = bodyResolvePackageInit(module, loc);
      succeeded(initOrFail) && *initOrFail == &dest) {
    for (ASTDecl *d : dest.lookupInCurrentScope(sourceName))
      if (isAlreadyProcessing(*d)) {
        selfReferential = true;
        break;
      }
  }

  // Check to see if the module has the construct we are importing.
  LookupResult result =
      selfReferential
          ? LookupResult::getFailure({})
          : shared.lookupAndResolveDecl(sourceName, sourceNameLoc, module,
                                        /*searchParentScopes=*/false,
                                        /*resolveTarget=*/resolveTarget);
  if (result.isErroneous())
    return failure();
  SmallVector<ASTDecl *> results;
  if (result.isFailure()) {
    // The name was not bound in the module's (or, for a package, its
    // __init__'s) scope. It may instead be a submodule of the package -
    // resolve it lazily from the filesystem.
    if (ASTDecl *sub =
            shared.tryImportSubModule(module, sourceName, sourceNameLoc)) {
      results.push_back(sub);
    } else {
      StringRef name =
          cast_or_null<mlir::SymbolOpInterface>(module.getIfOperation())
              .getName();
      StringRef declType = isa_and_nonnull<PackageOp>(module.getIfOperation())
                               ? "package"
                               : "module";
      emitError(sourceNameLoc, declType + " '" + name + "' does not contain '" +
                                   sourceName.getValue() + "'");
      return failure();
    }
  } else {
    // Copy the entries out of the scope's symbol table. The lookups below
    // (re-export resolution, extension imports) can resolve further decls and
    // invalidate the underlying storage, leaving a dangling reference.
    results.assign(result.getIfSuccess().begin(), result.getIfSuccess().end());
  }
  assert(!results.empty() && "other cases handled above");

  shared.notifyListenerOnRef(results, sourceName, sourceNameLoc);
  shared.notifyListenerOnRef(results, destName, destNameLoc);

  // If what we imported is a *submodule* (not a re-exported symbol), bind an
  // ImportOp over it rather than the raw module, so component access through
  // the imported name (`sub.foo`) is gated like any other import. The import
  // re-resolves the name with no package context, so it must be absolute:
  // `a.b.sub` if it came from an absolute `from a.b import sub`, or if it came
  // from a relative import `from . import sub` where the current package is
  // `a.b`.
  ImportPathAttr gatePath;
  if (results.size() == 1 && isa_and_nonnull<PackageOp, FileModuleOp>(
                                 results.front()->getIfOperation())) {
    if (moduleName.getRelativeLevel() > 0) {
      gatePath = getAbsoluteModuleName(*results.front());
    } else {
      SharedState::ImportPath submodulePath = modulePath;
      submodulePath.components.push_back(sourceName.getValue());
      gatePath = submodulePath.toAttr(getContext());
    }
  }
  if (gatePath) {
    // Bind an ImportOp under `destName` over the imported submodule, and
    // reuse the from-import placeholder's ASTDecl for it. The ASTDecl outlives
    // this resolution (a later resolve-body pass dereferences it), so it must
    // end up pointing at a live op; re-pointing it also keeps name lookups for
    // `destName` from returning the now-superseded UnresolvedImportOp.
    ASTDecl *placeholderDecl = nullptr;
    if (dest.declsInScope) {
      auto it = dest.declsInScope->find(destName);
      if (it != dest.declsInScope->end())
        for (ASTDecl *d : it->second) {
          auto imp = dyn_cast_or_null<UnresolvedImportOp>(d->getIfOperation());
          if (imp && imp.getModulePathAttr() == moduleName &&
              imp.getDeclNameAttr() == sourceName) {
            placeholderDecl = d;
            break;
          }
        }
    }
    if (placeholderDecl) {
      Operation *oldOp = placeholderDecl->getIfOperation();
      OpBuilder fromBuilder(oldOp);
      auto gateOp =
          ImportOp::create(fromBuilder, shared.translateLocation(destNameLoc),
                           /*sym_name=*/destName, gatePath);
      placeholderDecl->setIRValue(gateOp.getOperation());
      placeholderDecl->resolvedness = DeclResolvedness::body;
      registerDeclSymbol(placeholderDecl);
      // The placeholder op is now superseded by the gate and has no live use.
      if (oldOp && oldOp != gateOp.getOperation())
        deadImportPlaceholders.insert(oldOp);
    } else {
      // No placeholder to reuse (defensive): bind a fresh gate decl.
      OpBuilder fromBuilder = dest.getDeclEndBuilder();
      createImportOp(dest, fromBuilder, destName, gatePath,
                     shared.translateLocation(destNameLoc));
    }
  } else if (failed(aliasImportDecls(results, destName, sourceName, moduleName,
                                     destNameLoc, dest, false))) {
    // Import the desired declaration (struct, function, etc.) the user asked
    // for.
    return failure();
  }

  // Also look for extensions in the source module.
  // When importing any declaration from a module, import all extensions from
  // that module so they're available in the destination scope.
  // All extensions known to their parents as e.g. `extension:MyStruct` but
  // also as `extension:` so asking for `extension:` will get all extensions.
  // The aggregate `extension:` entry is a union over every wildcard import in
  // the source scope, so resolve them all first. A lazy lookup would stop at
  // the first wildcard that provides any extension and miss the rest. A
  // failing wildcard reports at its own import site; it shouldn't fail this
  // explicit import.
  if (FailureOr<ASTDecl *> srcInit =
          bodyResolvePackageInit(module, sourceNameLoc);
      succeeded(srcInit)) {
    expandWildcardsForName(*srcInit ? **srcInit : module,
                           shared.extensionsScopeMarker);
  }
  StringAttr extensionNameAttr = shared.extensionsScopeMarker;
  auto requestedModuleExts =
      shared.lookupAndResolveDecl(extensionNameAttr, sourceNameLoc, module,
                                  /*searchParentScopes=*/false,
                                  /*resolveTarget=*/false);
  if (requestedModuleExts.isSuccess()) {
    ArrayRef<ASTDecl *> allExtensions = requestedModuleExts.getIfSuccess();
    if (!allExtensions.empty()) {
      shared.notifyListenerOnRef(allExtensions, extensionNameAttr,
                                 sourceNameLoc);
      shared.notifyListenerOnRef(allExtensions, extensionNameAttr, destNameLoc);

      // Import under "extension:" for finding all extensions in a module
      if (failed(aliasImportDecls(allExtensions, extensionNameAttr,
                                  extensionNameAttr, moduleName, destNameLoc,
                                  dest, true))) {
        emitError(destNameLoc, "failed to import extensions from module '" +
                                   modulePath.toDottedString() + "'");
        return failure();
      }
      // Now that we have all the extensions, go through each one and register
      // it under its specific name (e.g. "extension:SIMD") so
      // collectTypeAndExtensions can find them.
      for (ASTDecl *extensionDecl : allExtensions) {
        auto extOp =
            dyn_cast_or_null<ExtensionDeclOp>(extensionDecl->getIfOperation());
        if (!extOp)
          continue;
        auto targetStructName = extOp.getTargetStructName().value();
        StringAttr specificExtensionName = StringAttr::get(
            getContext(), shared.extensionsScopeMarker.getValue().str() +
                              targetStructName.str());
        if (failed(aliasImportDecls({extensionDecl}, specificExtensionName,
                                    extensionNameAttr, moduleName, destNameLoc,
                                    dest, true))) {
          emitError(destNameLoc, "failed to import extension for '" +
                                     targetStructName + "' from module '" +
                                     modulePath.toDottedString() + "'");
          return failure();
        }
      }
    }
  }

  return success();
}

LogicalResult DeclResolver::importWildcardDeclsFromModule(
    ASTDecl &context, const UnresolvedWildcardImport &unresolvedImport,
    StringAttr onlyName) {
  assert((!onlyName || !isInternalName(onlyName)) &&
         "callers must filter internal names; wildcards never provide them");
  ImportPathAttr moduleName = unresolvedImport.moduleName;
  SMLoc loc = unresolvedImport.importLoc;
  auto modulePath = SharedState::ImportPath::fromAttr(moduleName);
  PackageOp currentPackage =
      dyn_cast_or_null<PackageOp>(context.getIfOperation());
  if (!currentPackage && context.getIfOperation())
    currentPackage = context.getIfOperation()->getParentOfType<PackageOp>();

  // Make sure the module has been resolved.
  ASTDecl &module = shared.importModule(modulePath, currentPackage, loc);
  if (failed(resolveBody(module, loc)))
    return failure();

  // For packages, wildcard-import from __init__'s scope: it holds all
  // re-exported symbols and any sibling-module stubs. A PackageOp's own scope
  // is always empty.
  FailureOr<ASTDecl *> initOrFailure = bodyResolvePackageInit(module, loc);
  if (failed(initOrFailure))
    return failure();
  ASTDecl &iterScope = *initOrFailure ? **initOrFailure : module;

  // Imports a wildcard decl into the current scope. Returns true on failure.
  auto importWildcardDecl = [this, moduleName, &context,
                             loc](StringAttr name,
                                  ArrayRef<ASTDecl *> decls) -> LogicalResult {
    // Ignore erroneous children, which have nothing in them.
    if (decls.empty())
      return success();

    auto shouldImportWildcardDecl = [](ASTDecl *decl) {
      auto structOp = dyn_cast_or_null<StructDeclOp>(decl->getIfOperation());
      if (!structOp || !structOp.isSynthetic() || !structOp.getDefinesClosure())
        return true;
      return false;
    };

    SmallVector<ASTDecl *> filteredDecls;
    llvm::copy_if(decls, std::back_inserter(filteredDecls),
                  shouldImportWildcardDecl);
    if (filteredDecls.empty())
      return success();

    return aliasImportDecls(filteredDecls, name, name, moduleName, loc, context,
                            false);
  };

  // Resolve pending wildcard imports in the scope we are about to iterate
  // (optionally only for the name we're interested in).
  LogicalResult result = success();
  if (onlyName) {
    expandWildcardsForName(iterScope, onlyName);

    auto decls = iterScope.lookupInCurrentScope(onlyName);

    result = importWildcardDecl(onlyName, decls);
  } else {
    if (failed(resolveAllWildcardImports(iterScope)))
      return failure();

    for (const auto &[name, decls] : iterScope.getDeclsInScope()) {
      // Wildcard imports don't import decls with a leading '_'.
      if (isInternalName(name))
        continue;

      // A name whose decls are all disabled is a miss.
      if (LLVM_UNLIKELY(iterScope.hasDisabledDecls) &&
          llvm::all_of(decls,
                       [](ASTDecl *decl) { return decl->isDisabled(); })) {
        continue;
      }

      if (failed(importWildcardDecl(name, decls)))
        result = failure();
    }
  }

  // Also import all extensions from the source module, similar to what
  // importDeclFromModule does. This ensures that when doing wildcard imports,
  // extensions are available in the destination scope.
  // Extensions are registered under "extension:" so we can find all of them.
  StringAttr extensionNameAttr = shared.extensionsScopeMarker;
  auto moduleExtensions =
      shared.lookupAndResolveDecl(extensionNameAttr, loc, module,
                                  /*searchParentScopes=*/false,
                                  /*resolveTarget=*/false);
  if (moduleExtensions.isSuccess()) {
    ArrayRef<ASTDecl *> allExtensions = moduleExtensions.getIfSuccess();
    if (!allExtensions.empty()) {
      shared.notifyListenerOnRef(allExtensions, extensionNameAttr, loc);

      // Import under "extension:" for finding all extensions in a module
      if (failed(aliasImportDecls(allExtensions, extensionNameAttr,
                                  extensionNameAttr, moduleName, loc, context,
                                  true))) {
        emitError(loc, "failed to import extensions from module '" +
                           modulePath.toDottedString() + "'");
        return failure();
      }

      // Now register each extension under its specific name (e.g.
      // "extension:SIMD") so collectTypeAndExtensions can find them.
      for (ASTDecl *extensionDecl : allExtensions) {
        auto extOp =
            dyn_cast_or_null<ExtensionDeclOp>(extensionDecl->getIfOperation());
        if (!extOp)
          continue;
        auto targetStructName = extOp.getTargetStructName();
        if (!targetStructName)
          continue;
        StringAttr specificExtensionName = StringAttr::get(
            getContext(), shared.extensionsScopeMarker.getValue().str() +
                              targetStructName.value().str());
        if (failed(aliasImportDecls({extensionDecl}, specificExtensionName,
                                    extensionNameAttr, moduleName, loc, context,
                                    true))) {
          emitError(loc, "failed to import extension for '" +
                             targetStructName.value() + "' from module '" +
                             modulePath.toDottedString() + "'");
          return failure();
        }
      }
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Decl Resolution

LogicalResult DeclResolver::resolve(ASTDecl &decl, DeclResolvedness howResolved,
                                    SMLoc loc) {
  // If decl is already resolved enough, we're done.
  if (decl.resolvedness >= howResolved) {
    // If decl is busted, then return failure.
    return success(!decl.isErroneous());
  }

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (failed(declsCurrentlyProcessing.insert(&decl, loc))) {
    auto diag =
        emitError(decl.getLoc(),
                  "attempt to resolve a recursive reference to declaration");

    auto addDeclName = [&](ASTDecl *decl) {
      std::optional<StringRef> name = decl->getUserNameIfOperation();
      if (!name)
        return;
      diag << " '";
      if (auto structOp = dyn_cast_or_null<StructDeclOp>(
              decl->getParentDecl()->getIfOperation()))
        diag << structOp.getDeclName().getValue() << ".";
      diag << *name << "'";
    };

    addDeclName(&decl);
    diag.attachNote(loc) << "referenced from here";

    // Include a stack trace of notes showing why this is being cyclicly
    // resolved.
    for (ASTDecl *prev : llvm::reverse(declsCurrentlyProcessing.stack)) {
      // Bottom out when we find the declaration in question.
      diag.attachNote(*prev) << "by declaration";
      addDeclName(prev);

      diag.attachNote(declsCurrentlyProcessing.map[prev])
          << "referenced through this use";
      if (prev == &decl)
        break;
    }
    decl.setErroneous();
    return failure();
  }

  // Handle decls that are loaded from bytecode. These decls are not parsed like
  // decls originating from source files.
  if (decl.loadedFromBytecode) {
    if (failed(shared.resolveDeclFromBytecode(decl, howResolved)))
      decl.setErroneous();

    declsCurrentlyProcessing.pop();
    return success(!decl.isErroneous());
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signature) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    if (auto declOp = decl.getIfOperation()) {
      TypeSwitch<Operation &>(*declOp)
          .Case<FnOp, StructDeclOp, StructFieldOp, TraitDeclOp, ExtensionDeclOp,
                AliasDeclOp>([&](auto op) {
            // If this is a synthetic decl, resolve it specially.
            if (decl.getCursor().isInvalid()) {
              if constexpr (std::is_same_v<FnOp, decltype(op)>) {
                if (failed(resolveSyntheticSignature(op, decl)))
                  decl.setErroneous();
                return;
              }
              if constexpr (std::is_same_v<AliasDeclOp, decltype(op)>) {
                if (failed(resolveSyntheticSignature(op, decl)))
                  decl.setErroneous();
                return;
              }
            }

            // Generate pretty stack traces if a crash happens in this scope.
            CrashReporter crashReporter(decl.getLoc(),
                                        "resolving decl signature", shared);

            // Resolve the signature: on a parse error, we note that the
            // decl is malformed and should not be referenced to silence
            // downstream errors.
            Lexer lexer(shared.diags, decl.getCursor());
            if (failed(resolveSignature(op, lexer, decl)))
              decl.setErroneous();
            decl.getCursor() = lexer.getCursor();
          })
          .Case<UnresolvedImportOp>([&](auto op) {
            // Resolve the signature: on a parse error, we note that the decl
            // is malformed and should not be referenced to silence downstream
            // errors.
            if (failed(resolveSignature(op, decl)))
              decl.setErroneous();
          })
          .Case<LIT::FileModuleOp, ModuleOp, PackageOp, ImportOp,
                UnresolvedWildcardImportOp>([&](auto op) { /*Nothing*/ })
          .Default([&](Operation &attr) {
            llvm_unreachable(
                "do not know how to resolve the signature of this decl!");
          });
    } else if (auto typeValue = decl.getIfTypeValue()) {
      auto traitType = dyn_cast_or_null<TraitType>(decl.getIfTypeValue());
      assert(traitType && "do not know how to resolve the signature of this "
                          "decl!");
      if (failed(resolveSignature(traitType, decl)))
        decl.setErroneous();
    } else if (auto witness = decl.getIfWitness()) {
      if (failed(resolveSignature(witness, decl)))
        decl.setErroneous();
    } else {
      llvm_unreachable(
          "do not know how to resolve the signature of this decl!");
    }
    // Never regress resolvedness. In the case of non inlined nested functions,
    // the body is fully resolved when the signature is resolved in order
    // to identify the value of 'capturing'
    if (decl.resolvedness != DeclResolvedness::body)
      decl.resolvedness = DeclResolvedness::signature;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (decl.resolvedness < DeclResolvedness::body &&
      howResolved == DeclResolvedness::body) {
    auto checkEndOfBodyCursor = [&](Lexer &lexer) {
      // If the final parse of the declaration didn't match the initial
      // parse, report an error about unrecognized tokens at end of
      // declaration.
      if (!decl.isMatchingEndCursor(lexer.getCursor()) && !decl.isErroneous()) {
        if (lexer.getToken().isAny(Token::kw_def, Token::kw_struct,
                                   Token::kw_trait, Token::kw_class,
                                   Token::kw_var)) {
          lexer.emitTokenError(
              "definition isn't on its own line at the correct "
              "indentation");
        } else if (lexer.getToken().is(Token::eof)) {
          lexer.emitTokenError(
                   "internal error: decl parsing skipped beyond end "
                   "of declaration")
                  .attachNote(decl.getLoc())
              << "declaration started here";
        } else {
          lexer.emitTokenError("unknown tokens at the end of a declaration");
        }
      }
    };

    // Mark the body as already resolved so that name lookup can be performed
    // in the decl during resolution.
    //    decl.resolvedness = DeclResolvedness::body;

    // Handle each operation that can be name bound.
    if (decl.isErroneous()) {
      // If the decl is already erroneous, trying to process further may crash
      // or cause spurious error messages.
    } else if (auto declOp = decl.getIfOperation()) {
      TypeSwitch<Operation &>(*declOp)
          .Case<FileModuleOp, FnOp, StructDeclOp, StructFieldOp,
                ExtensionDeclOp, TraitDeclOp, AliasDeclOp>([&](auto op) {
            // A deferred source module opens + lexes its file lazily here, on
            // first body resolution, mirroring how a bytecode child
            // materializes from its reader. This sets up its parse cursor.
            if constexpr (std::is_same_v<FileModuleOp, decltype(op)>) {
              if (decl.getCursor().isInvalid() &&
                  failed(shared.materializeDeferredModule(decl, loc))) {
                decl.setErroneous();
                return;
              }
            }

            // If this is a synthetic decl, complete it specially.
            if (decl.getCursor().isInvalid()) {
              if constexpr (std::is_same_v<FnOp, decltype(op)>) {
                if (op.isSynthetic() && failed(resolveSyntheticBody(op, decl)))
                  decl.setErroneous();
                return;
              }
            }

            // Generate pretty stack traces if a crash happens in this scope.
            CrashReporter crashReporter(decl.getLoc(), "resolving decl body",
                                        shared);

            // Parse the body of the declaration from the correct point.
            Lexer lexer(shared.diags, decl.getCursor());
            if (resolveBody(op, lexer, decl))
              return;

            checkEndOfBodyCursor(lexer);
          })
          .Case<ConformanceOp>([&](auto op) {
            if (failed(resolveBody(op, decl)))
              decl.setErroneous();
          })
          .Case<PackageOp>([&](auto op) { (void)resolveBody(op, decl); })
          .Case<ModuleOp, ImportOp, UnresolvedImportOp,
                UnresolvedWildcardImportOp>([&](auto op) { /*Nothing*/ })
          .Default([&](Operation &attr) {
            llvm_unreachable(
                "do not know how to resolve the body of this decl!");
          });
    } else if (auto typeVal = decl.getIfTypeValue()) {
      auto traitType = dyn_cast_or_null<TraitType>(decl.getIfTypeValue());
      assert(traitType && "do not know how to resolve the body of this decl!");
      if (failed(resolveBody(traitType, decl)))
        decl.setErroneous();
    } else if (auto witness = decl.getIfWitness()) {
      if (failed(resolveBody(witness, decl)))
        decl.setErroneous();
    } else {
      llvm_unreachable("do not know how to resolve the body of this decl!");
    }

    if (decl.resolvedness == DeclResolvedness::signature)
      decl.resolvedness = DeclResolvedness::body;
  }

  declsCurrentlyProcessing.pop();
  // If decl is busted, then return failure.
  return success(!decl.isErroneous());
}

void DeclResolver::resolveAllWithin(ASTDecl &decl) {
  std::deque<ASTDecl *> worklist{&decl};
  while (!worklist.empty()) {
    ASTDecl *declIt = worklist.back();
    worklist.pop_back();

    if (declIt->isDisabled())
      continue;

    (void)resolveBody(*declIt, declIt->getLoc());

    for (auto &[name, decls] : declIt->getDeclsInScope()) {
      for (ASTDecl *child : decls) {
        if (child->getParentDecl() == declIt)
          worklist.push_front(child);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Top-Level Decl Resolution

void DeclResolver::resolveReferencedDecls() {
  // Deinitable and its members will be referenced by
  // CheckLifetimes, so make sure to resolve it.
  if (ASTDecl *traitDecl = shared.lookupBuiltinTrait("Deinitable", SMLoc()))
    resolveAllWithin(*traitDecl);

  // Iteratively resolve all of the parsed decls that got referenced outside
  // the main container (typically stdlib/library declarations).
  llvm::SetVector<ASTDecl *> deferredDecls;
  size_t parsedDeclIt = 0;
  do {
    // Resolve all of the newly parsed decls that got referenced.
    for (; parsedDeclIt != parsedDeclList.size(); ++parsedDeclIt) {
      ASTDecl &decl = *parsedDeclList[parsedDeclIt];

      // If the decl was never touched and we pulled it in from bytecode, treat
      // it as unreachable and don't resolve it now.
      if (decl.resolvedness == DeclResolvedness::unparsed) {
        // Some decls always need to be resolved if their parents were resolved,
        // allowlist the decls that we can safely ignore when unparsed.
        if (isa_and_nonnull<FnOp, FileModuleOp, PackageOp, ImportOp,
                            UnresolvedImportOp, UnresolvedWildcardImportOp,
                            StructDeclOp, TraitDeclOp, AliasDeclOp>(
                decl.getIfOperation())) {
          deferredDecls.insert(&decl);
          continue;
        }
      }

      (void)resolveBody(decl, decl.getLoc());
    }

    // After resolving the newly parsed decls, make sure we resolve any
    // previously parsed decls that are newly referenced.
    bool resolvedAnything = false;
    do {
      resolvedAnything = false;
      for (ASTDecl *decl : deferredDecls) {
        // Fully resolve this decl if it was only midway resolved during normal
        // parsing resolution.
        if (decl->resolvedness == DeclResolvedness::signature) {
          (void)resolveBody(*decl, decl->getLoc());
          resolvedAnything = true;
        }
      }
    } while (resolvedAnything);
  } while (parsedDeclIt != parsedDeclList.size());
}

void DeclResolver::resolveAllReferencedFrom(ASTDecl &decl,
                                            bool eraseUnparsedDecls) {
  CompilerTimeTraceScope traceScope("resolveAllReferencedFrom", [&] {
    return decl.getUserNameIfOperation().value_or("").str();
  });

  // The first stage is to fully resolve all of the decls recursively defined
  // within the main container. These decls provide the anchor for resolution.
  std::deque<ASTDecl *> worklist({&decl});
  while (!worklist.empty()) {
    ASTDecl *declIt = worklist.back();
    worklist.pop_back();

    // Resolve the decl.
    (void)resolveBody(*declIt, declIt->getLoc());

    if (declIt->isDisabled())
      continue;

    // When validating doc strings, we wish to only validate those defined on
    // decl in the main container. As this point the main container decl has
    // been fully resolved, so it's an opportune time to validate.
    validateDocString(*declIt);

    // If this is a package, resolve all of the modules within it as a pre-step.
    // Normally these get lazily resolved, but if we're forcing pulling them in,
    // we need to do it now.
    if (auto packageOp =
            dyn_cast_or_null<PackageOp>(declIt->getIfOperation())) {
      for (ASTDecl *sub : shared.getNestedModuleDecls(packageOp)) {
        (void)resolveBody(*sub, declIt->getLoc());
        worklist.push_front(sub);
      }
    }

    // Traverse the children. We don't resolve alias children, these will be
    // resolved separately if they actually got referenced.
    for (auto &[_, decls] : declIt->getDeclsInScope()) {
      for (ASTDecl *decl : decls)
        if (decl->getParentDecl() == declIt)
          worklist.push_front(decl);
    }
  }

  // After all of the children within `decl` have been fully resolved,
  // iteratively resolve all of the outside decls that got referenced.
  // Skip when errors have already been emitted: resolving library/stdlib
  // declarations is expensive (potentially the entire stdlib) and
  // unnecessary when compilation will fail anyway.
  if (!shared.diags.isErrorEmitted())
    resolveReferencedDecls();

  // Erase unresolved operations from source.
  if (eraseUnparsedDecls) {
    for (ASTDecl *decl : parsedDeclList) {
      // During trait body resolution we create decls that point to parent
      // trait decl's FnOps. In order to avoid double frees later on in this
      // loop bail early if we come across such a decl.
      if (decl->getCursor().isInvalid())
        continue;
      if (decl->resolvedness == DeclResolvedness::unparsed &&
          !decl->loadedFromBytecode)
        if (Operation *op = decl->getIfOperation()) {
          if (!isa<UnresolvedImportOp>(op)) {
            op->erase();
            decl->setIRValue(nullptr);
          }
        }
    }

    // Erase from-import placeholders that resolved to a submodule and were
    // superseded by an ImportOp (see importDeclFromModule). This runs
    // after all resolution, so any sibling decl surfaced over the same op has
    // already been re-pointed at its own gate; null out any decl that still
    // references one before erasing it, to avoid a dangling ASTDecl.
    if (!deadImportPlaceholders.empty()) {
      for (ASTDecl *decl : parsedDeclList)
        if (deadImportPlaceholders.contains(decl->getIfOperation()))
          decl->setIRValue(nullptr);
      for (Operation *op : deadImportPlaceholders)
        op->erase();
      deadImportPlaceholders.clear();
    }
  }
}

void DeclResolver::expandWildcardsForName(ASTDecl &scope, StringAttr name,
                                          bool stopOnFirstHit) {
  // Wildcard imports don't import internal names.
  if (name.empty() || isInternalName(name))
    return;
  // Newest first. Re-read the list each step: expanding one wildcard runs
  // arbitrary resolution, which may grow it.
  for (size_t i = scope.getUnresolvedWildcardImports().size(); i > 0; --i) {
    UnresolvedWildcardImport &wildcard =
        scope.getUnresolvedWildcardImports()[i - 1];
    if (wildcard.isSuperseded || !wildcard.markSearched(name.getValue()))
      continue;
    // On failure keep going: another wildcard may still provide the name.
    if (failed(importWildcardDeclsFromModule(scope, wildcard, name)))
      continue;
    if (stopOnFirstHit && !scope.lookupInCurrentScope(name).empty())
      return;
  }
}

LogicalResult DeclResolver::resolveAllWildcardImports(ASTDecl &scope) {
  // Resolve wildcard imports from last to first, thereby meaning the last one
  // "wins" in terms of shadowing; subsequent colliding decls won't be brought
  // into scope. The list is append-only, so indices stay valid, but expanding a
  // wildcard runs arbitrary resolution that may append to it and reallocate:
  // re-read the list every step, and mark an entry drained before expanding it
  // so a re-entrant call can't expand it twice.
  size_t i = scope.getUnresolvedWildcardImports().size();
  while (i-- > 0) {
    UnresolvedWildcardImport &wildcard =
        scope.getUnresolvedWildcardImports()[i];
    if (wildcard.isSuperseded)
      continue;
    wildcard.isSuperseded = true;
    UnresolvedWildcardImport drained{wildcard.moduleName, wildcard.importLoc};
    size_t before = scope.getUnresolvedWildcardImports().size();
    if (failed(importWildcardDeclsFromModule(scope, drained)))
      return failure();
    // Anything appended is newer, so it must be expanded first.
    if (size_t after = scope.getUnresolvedWildcardImports().size();
        after != before)
      i = after;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Symbol-ASTDecl Mapping

ASTDecl &DeclResolver::getDeclForTypeSymbol(SymbolRefAttr symbol) const {
  auto it = declForTypeSymbol.find(symbol);
  if (it == declForTypeSymbol.end()) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << "unknown decl symbol: " << symbol;
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return *it->second;
}

ASTDecl *
DeclResolver::getDeclForTypeSymbolIfExists(SymbolRefAttr symbol) const {
  auto it = declForTypeSymbol.find(symbol);
  return it != declForTypeSymbol.end() ? it->second : nullptr;
}

ASTDecl *DeclResolver::getDeclForFuncSymbol(SymbolRefAttr attr) const {
  auto it = declForFuncSymbol.find(attr);
  return it != declForFuncSymbol.end() ? it->second : nullptr;
}

Operation *DeclResolver::finalizeFuncSignature(FnOp funcOp, ASTDecl &decl) {
  // Install it in the symbol table and check for redefinition while doing so.
  Operation *existing = shared.uniquifyNameAndAddToParentSymbolTable(funcOp);
  // Remember the mapping from its fully mangled symbol so we can find its AST
  // representation and body from IR references.
  // NOTE: this has to run after `uniquifyNameAndAddToParentSymbolTable` as the
  // call above might update the symbol name when there is a name collision.
  declForFuncSymbol[getFullyResolvedSymbolRef(funcOp)] = &decl;
  return existing;
}

ASTDecl *DeclResolver::getTraitDecl(TraitType trait) {
  // This is the invariant that we want to enforce, if the assertion triggered,
  // there must be something wrong elsewhere.
  assert(getCanonicalTrait(trait) == trait &&
         "trait type should always be canonicalized");

  // Check if the canonicalized trait type has a hit.
  if (auto it = canonicalTraitCompositionDecls.find(trait);
      it != canonicalTraitCompositionDecls.end())
    return it->second;

  // Otherwise, create a new decl and register for the canonical trait type.
  // Trait compositions are anonymous declarations and do not have a source
  // location themselves. Conformance errors will be routed to its member decls.
  ASTDecl *decl = &createUnlistedDecl(DeclIRValue(trait), /*loc=*/{},
                                      /*parentDecl=*/nullptr, LexerCursor(),
                                      LexerCursor(), /*indentation=*/-1);

  // Initialize the decl to signature-resolved since we do not have anything to
  // do for the signature resolve phase.
  decl->resolvedness = DeclResolvedness::signature;
  canonicalTraitCompositionDecls[trait] = decl;
  return decl;
}

//===----------------------------------------------------------------------===//
// Export Handling

void DeclResolver::registerAndCheckExport(StringRef aliasName, SMLoc loc) {
  auto [it, inserted] = exportedSymbolNames.try_emplace(aliasName, loc);
  if (!inserted) {
    auto diag = emitError(loc, "invalid re-export of ") << aliasName;
    diag.attachNote(it->second) << "previous export here";
    return;
  }
}

void DeclResolver::exportMain(ASTDecl &funcDecl) {
  FnOp userMainFn = cast_or_null<FnOp>(funcDecl.getIfOperation());
  FnTypeGeneratorType userMainSignature = userMainFn.getFuncTypeGenerator();
  ASTDecl *containingDecl = funcDecl.getParentDecl();
  SMLoc loc = funcDecl.getLoc();

  // The type of main function described by the given func decl.
  enum MainKind {
    // A non-raising function that returns None.
    kNonRaisingNoneMain,
    // A raising function that returns None.
    kRaisingNoneMain,
  };
  MainKind mainKind = kNonRaisingNoneMain;

  // Validate that main has the expected signature.
  if (!userMainSignature.getInputParamTypes().empty()) {
    shared.emitError(loc, "'main()' does not accept parameters; "
                          "remove the square brackets");
    return;
  }
  ASTType userResultType(userMainFn.getUserResultType());
  ArrayRef<Type> argTypes = userMainSignature.getArguments();

  // Process a main returning none.
  if (userResultType.isNoneType()) {
    if (userMainSignature.isThrows()) {
      mainKind = kRaisingNoneMain;
      // Drop the error from the argument list.
      argTypes = argTypes.drop_front(2);
    }

    // Process a main returning object.
  } else {
    shared.emitError(
        loc, "'main()' does not return a value; remove the return type");
    return;
  }
  if (!argTypes.empty()) {
    shared.emitError(loc, "'main()' does not accept arguments; remove them");
    return;
  }

  // Validate that we aren't in a package, defining a `main` within a package
  // is not fully supported.
  if (userMainFn->getParentOfType<PackageOp>()) {
    shared.emitError(loc, "'main()' is not supported within packages");
    return;
  }

  // Utility for resolving a decl within the Startup module.
  ASTDecl &startupModule =
      shared.importModule({"std", "builtin", "_startup"},
                          /*currentPackage=*/nullptr, funcDecl.getLoc());
  auto resolveStartDecl = [&](StringRef name) -> ASTDecl * {
    auto result = shared.lookupAndResolveDecl(
        name, funcDecl.getLoc(), startupModule, /*searchParentScopes=*/false);
    if (result.getIfSuccess().empty()) {
      if (result.isFailure()) {
        shared.emitError(funcDecl.getLoc(),
                         "unable to resolve `Builtin.Startup` module when "
                         "exporting 'main'");
      }
      return nullptr;
    }
    ASTDecl *decl = result.getIfSuccess().front();
    if (failed(resolveBody(*decl, decl->getLoc())))
      return nullptr;
    return result.getIfSuccess().front();
  };

  // Generate a shim for main that handles parsing command line arguments,
  // capturing uncaught exceptions, and returning the exit code. The shim
  // defines a C-ABI compatible function that sets up the mojo runtime.
  OpBuilder builder = containingDecl->getDeclEndBuilder();

  // The Startup module provides a stubbed out shim for us to use, so pull that
  // in.
  ASTDecl *mainShimProtoDecl = resolveStartDecl("__mojo_main_prototype");
  if (!mainShimProtoDecl)
    return;
  FnOp mainShimProtoFn =
      cast_or_null<FnOp>(mainShimProtoDecl->getIfOperation());

  // Builder function.
  StringAttr mainAttr = StringAttr::get(getContext(), "main");
  auto shimMainFn = cast<FnOp>(builder.clone(*mainShimProtoFn));
  shimMainFn.setSymNameAttr(mainAttr);
  shimMainFn.setLinkageNameAttr(
      LinkageNameAttr::get(shimMainFn->getContext(), "main"));
  shimMainFn.setExported();
  shimMainFn.getBody()->clear();
  // Set the C ABI effect
  auto sigGen = shimMainFn.getFuncTypeGenerator();
  auto body = sigGen.getBody();
  auto newBody = body.getWithFnEffects(body.getFnEffects().setCABI(true));
  shimMainFn.setFuncTypeGenerator(FnTypeGeneratorType::get(
      sigGen.getInputParamTypes(), newBody, sigGen.getParamListAttrs()));

  // Populate the body of the shim. For this we designate the internal
  // implementation to one of the wrapper helpers in the Startup module,
  // depending on how the user specified their main function.
  StringRef mainWrapperName;
  switch (mainKind) {
  case kNonRaisingNoneMain:
    mainWrapperName = "__wrap_and_execute_main";
    break;
  case kRaisingNoneMain:
    mainWrapperName = "__wrap_and_execute_raising_main";
    break;
  }
  ASTDecl *mainWrapperDecl = resolveStartDecl(mainWrapperName);
  if (!mainWrapperDecl)
    return;
  FnOp mainWrapperFn = cast_or_null<FnOp>(mainWrapperDecl->getIfOperation());
  FnTypeGeneratorType mainWrapperSigGen = mainWrapperFn.getFuncTypeGenerator();

  // Generate a reference to the main wrapper function, which expects the user
  // main to be provided via an parameter.
  FuncType mainWrapperSig = mainWrapperSigGen.getBody();
  FnMetaOriginDataAttr mainWrapperFnMeta =
      cast<FnMetaOriginDataAttr>(mainWrapperSig.getMetadata());
  auto strippedMainWrapperFnMeta = FnMetaOriginDataAttr::get(
      getContext(), mainWrapperFnMeta.getNumImplicitOriginDecls(),
      mainWrapperFnMeta.getCaptureOrigins(),
      mainWrapperFnMeta.getIsNestedOriginsReadOnly(),
      mainWrapperFnMeta.getDefinesInteriorOrigins());
  auto strippedMainWrapperSig = FuncType::get(
      mainWrapperSig.getValues(), mainWrapperSig.getArgConventions(),
      mainWrapperSig.getFnEffects(), strippedMainWrapperFnMeta,
      mainWrapperSig.getArgListAttrs());
  SymbolConstantAttr wrapperFnRef = SymbolConstantAttr::get(
      getFullyResolvedSymbolRef(mainWrapperFn),
      GeneratorType::get(/*inputParamTypes=*/{}, strippedMainWrapperSig,
                         /*metadata=*/PogListAttr::get(getContext())),
      {SymbolConstantAttr::get(getFullyResolvedSymbolRef(userMainFn),
                               userMainSignature)});

  auto shimBodyBuilder = ImplicitLocOpBuilder::atBlockBegin(
      shimMainFn->getLoc(), shimMainFn.getBody());
  Value wrappedCallResult =
      CallOp::create(
          shimBodyBuilder, mainWrapperSigGen.getUserResultType(), wrapperFnRef,
          /*originParams=*/ArrayRef<TypedAttr>(), shimMainFn.getArguments())
          .getResult(0);

  // Align sugar if needed.
  if (wrappedCallResult.getType() != shimMainFn.getArgumentTypes()[0]) {
    assert(
        isEqualCanon(wrappedCallResult.getType(),
                     shimMainFn.getArgumentTypes()[0]) &&
        "wrapped call result type does not match shim main fn argument type");
    wrappedCallResult = RebindOp::create(
        shimBodyBuilder, shimMainFn.getArgumentTypes()[0], wrappedCallResult);
  }

  IREmitter::emitNormalReturn(shimBodyBuilder, wrappedCallResult);

  exportedSymbolNames.insert({mainAttr, funcDecl.getLoc()});
}

//===----------------------------------------------------------------------===//
// Decl Helpers

static void printConstraints(llvm::raw_ostream &os,
                             ArrayRef<ConstraintAttr> constraints) {
  if (constraints.empty())
    return;
  os << '{';
  llvm::interleave(
      constraints, os,
      [&](ConstraintAttr constraint) {
        ASTType::printParam(os, constraint.getProposition(), /*diags=*/{});
      },
      ",");
  os << '}';
}

StringAttr DeclResolver::getMangledName(StringAttr baseName, ASTDecl &container,
                                        FnTypeGeneratorType signatureGen) {
  // Compute the full signature of the decl to ensure dependent parameters from
  // a parent decl are name-erased in the mangled name.
  FnTypeGeneratorType fullSig =
      LIT::getFullSignature(container.getIfOperation(), signatureGen);

  SmallString<64> mangledName(baseName.getValue().begin(),
                              baseName.getValue().end());
  llvm::raw_svector_ostream os(mangledName);
  // Don't include parent parameters in the mangling.
  ArrayRef<Type> params = fullSig.getInputParamTypes().take_back(
      signatureGen.getInputParamTypes().size());
  if (!params.empty()) {
    size_t numSkipped = fullSig.getInputParamTypes().size() - params.size();
    os << '[';
    llvm::interleave(
        llvm::enumerate(params), os,
        [&](auto typeAndIdx) {
          auto [idx, implType] = typeAndIdx;
          ASTType type = implType;
          if (fullSig.getParamListAttrs().isPosVarArg(idx + numSkipped)) {
            os << "*";
            type = type.getParameterListInfo().elementType;
          }
          os << type.getAsString(/*diags=*/{});
        },
        ",");
    os << ']';
  }

  mangledName += '(';
  for (auto [argNo, conventionX, argTypeX] :
       llvm::enumerate(fullSig.getArgConventions(), fullSig.getArguments())) {
    auto convention = conventionX;
    ASTType argType = argTypeX;

    // We do not mangle results into the signature.
    if (isResultSlot(convention))
      continue;

    // Update the mangled name for this argument.
    if (argNo != 0)
      mangledName += ",";

    // Required keyword arguments can be overloaded on.
    if (fullSig.getArgListAttrs().getPassingKind(argNo) == PassingKind::KwOnly)
      mangledName += fullSig.getArgName(argNo).str() + ":";

    // If this had adjustments added to it because of its argument convention /
    // variadic state, strip them off.
    unsigned numStars = 0;
    if (fullSig.isPosVarArg(argNo)) {
      argType = ASTType(fullSig.getIfVariadicListOrPack(argNo))
                    .getVariadicListInfo()
                    .elementType;
      convention = fullSig.getVariadicConvention(argNo);
      numStars = 1;
    } else if (fullSig.isPack(argNo)) {
      TypedAttr packVariadic = ASTType(fullSig.getIfVariadicListOrPack(argNo))
                                   .getVariadicPackInfo()
                                   .typeList;
      mangledName += '*';
      ASTType::printParam(os, packVariadic, /*diags=*/{});
      continue;
    } else if (fullSig.isKwVarArg(argNo)) {
      // TODO: Propagate convention correctly.
      convention = ArgConvention::ImmReg;
      argType = argType.getKwargsDictRefValueType();
      numStars = 2;
    } else {
      argType = RefType::stripRefConvention(argType, convention);
    }
    mangledName += argType.getAsString(/*ctx=*/{nullptr});

    // Add suffix to disambiguate overloadable conventions.
    switch (convention) {
    case ArgConvention::OwnedReg:
      llvm_unreachable("not used by the parser");
    case ArgConvention::OwnedMem:
    case ArgConvention::DeinitMem:
      mangledName += '$';
      break;
    case ArgConvention::ImmReg:
    case ArgConvention::ImmMem:
    case ArgConvention::Mut:
      break;
    case ArgConvention::Ref:
    case ArgConvention::MutRef:
      mangledName += '%';
      break;
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
      llvm_unreachable("byref_result should be skipped");
    }

    while (numStars--)
      mangledName += '*';
  }
  mangledName += ')';

  // Add def constraints to the mangled name.
  printConstraints(os, fullSig.getParamListAttrs().getBodyConstraints());

  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(mangledName.begin(), mangledName.end(), '@', '_');
  return StringAttr::get(baseName.getContext(), mangledName);
}
