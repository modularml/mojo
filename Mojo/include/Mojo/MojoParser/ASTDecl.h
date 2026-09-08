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
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_ASTDECL_H
#define KGEN_MOJOPARSER_ASTDECL_H

#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/Support/TriBool.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TinyPtrVector.h"

namespace M::KGEN {
class ConstraintAttr;
class ParamDeclRefAttr;
} // end namespace M::KGEN

namespace M::KGEN::LIT {
class StructType;
class DocStringAttr;
class DocString;
class TraitDeclOp;
class TraitType;

// TODO(MOCO-4712): This should just be a CValue variant, we should
// simplify how trait witness are created in general, then it should be merged
// with the CValue variants when we have a IR representation for the witness
// decl.
struct WitnessDecl {
  // This is the merged witness non-function decls with the same witness name,
  // they might conflict each other if their types are not reconcilable.
  using UnresolvedDecls = SmallVector<ASTDecl *>;

  // This is the resolved witness entry.
  struct ResolvedType {
    StringAttr witnessName;
    Type witnessType;

    // Some extra information that is not available via a fnType. Meaningless
    // for non-function witnesses.
    ImplicitConversionKind implicitConversion = ImplicitConversionKind::None;
    bool isStaticMethod = false;
  };

  UnresolvedDecls getDecls() const { return cast<UnresolvedDecls>(storage); }
  ResolvedType getWitnessEntry() const { return cast<ResolvedType>(storage); }

  // Depending on whether the decl is fully resolved, it could be either be a
  // array lof decls that it depends on, or it could be a resolved witness type.
  SmartVariant<ResolvedType, UnresolvedDecls> storage;

  // This is the trait symbol that the decl witnessed.
  TraitSymbolAttr traitSymbol;
};

using DeclIRValue =
    SmartVariant<Operation *, WitnessDecl, CValue, std::nullopt_t>;

struct UnresolvedWildcardImport {
  ImportPathAttr moduleName;
  SMLoc importLoc;
  /// Whether or not the wildcard import has been superseded by a later one (in
  /// source order) or has been drained through resolution.
  bool isSuperseded = false;

  /// The names that this wildcard has already been searched for.
  std::unique_ptr<llvm::StringSet<>> searchedNames = nullptr;

  /// Mark this wildcard as searched for `name`, returning false if it already
  /// was.
  bool markSearched(StringRef name);
};

/// This is the AST representation (as opposed to the MLIR representation) of a
/// declaration in a program.  These maintain type checking and other
/// information that is irrelevant by the time the parser has created a complete
/// and correct IR for a program.  Declarations often have other declarations
/// nested inside of them, forming "scopes" and supporting name lookup.
///
/// Declarations in Mojo work the same way as in Python: scopes are nested
/// and are defined when a builtin, module, class/struct, or function definition
/// is introduced.  Mojo (like Python) allows forward references to values
/// before they are defined, so the parser works in multiple phases where it
/// notices a declaration but does not parse its body until it is demanded.
class ASTDecl {
public:
  /// ASTDecl's always have a notion of where they came from.
  SharedState &getShared() const { return shared; }
  MLIRContext *getContext() const { return shared.getContext(); }

  CValue getIfIRValue() const { return dyn_cast<CValue>(irValue); }

  /// If the IRValue is an Operation*, return it, otherwise return null.
  /// This is used for things like Module, StructDecl, Func, or ParamDecl.
  Operation *getIfOperation() const { return dyn_cast<Operation *>(irValue); }
  void setIRValue(DeclIRValue value) { irValue = value; }

  WitnessDecl *getIfWitness() const {
    if (isa<WitnessDecl>(irValue))
      return &cast<WitnessDecl>(irValue);
    return nullptr;
  }

  /// Return true if this decl is callable: a `FnOp`, or a witness for one.
  bool isCallableDecl() const;

  /// Return the full signature of this callable decl.
  FnTypeGeneratorType getDeclFullSignature() const;

  /// Return true if this decl is a static method.
  bool isStaticMethodDecl() const;

  /// Return how this decl may be used as an implicit conversion.
  ImplicitConversionKind getDeclImplicitConversionKind() const;

  // When handling things like default trait method, we might insert placeholder
  // ASTDecl for default implementation that later become invalid after body
  // resolving the struct (e.g., when there is a user-provided overload), this
  // that case, we  need to mark the ASTDecl to be invalid. We cannot simply
  // detach it from the parent decl as it corrupt a lot of loop iteration over
  // ASTDecls.
  void markDisabled() {
    assert(getCursor().isInvalid() && "should only disable synthetic ASTDecl");
    resolvedness = DeclResolvedness::body;
    irValue = std::nullopt;
    // Let the parent scope's name lookups know they must now filter disabled
    // decls; scopes without any stay on the fast path.
    if (parentDecl)
      parentDecl->hasDisabledDecls = true;
  }
  bool isDisabled() const { return isa<std::nullopt_t>(irValue); }

  /// Get the user-printable name of the declaration if it has one.
  ///  This removes any mangling (e.g. for parameters).
  std::optional<StringRef> getUserNameIfOperation() const;

  /// If the IRValue is a concrete type, return it as an ASTType.
  ASTType getIfTypeValue() const;

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG llvm::SMLoc
  getLoc() const {
    return loc;
  }
  void setLoc(llvm::SMLoc newLoc) { loc = newLoc; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG ASTDecl *
  getParentDecl() const {
    return parentDecl;
  }

  /// Get the nearest decl backed by one of the given operations. This can
  /// return itself, a parent decl, or null if no such decl is found.
  template <typename... OpTs>
  ASTDecl *getNearestDeclOfType() {
    ASTDecl *cur = this;
    while (cur && !isa_and_nonnull<OpTs...>(cur->getIfOperation()))
      cur = cur->getParentDecl();
    return cur;
  }

  /// Return the indentation of the introducer token or -1 if it wasn't on the
  /// start of line.
  ssize_t getIndentation() const { return indentation; }

  /// This cursor holds the location the parser should resume for the next phase
  /// of resolution.  For example, after initial scanning of a 'def', this will
  /// be on the def token.  After processing the signature, this will be after
  /// the colon.
  LexerCursor &getCursor() { return cursor; }

  /// Set the parse cursor and matching end-cursor together. Used when a decl's
  /// source is materialized after the decl is created.
  void setParseCursor(const LexerCursor &cursor, const LexerCursor &endCursor) {
    this->cursor = cursor;
    this->endCursorState = endCursor.getState();
  }

  /// Return true if the end of the speculatively scanned decl matches the
  /// specified cursor.
  bool isMatchingEndCursor(const LexerCursor &cursor) const {
    return endCursorState == cursor.getState();
  }

  /// Return the SymbolRefAttr for a declaration, including all scoping that may
  /// be needed, making it unique for every declaration.  This returns null for
  /// named values that do not have a declaration.
  SymbolRefAttr getSymbolRef() const;

  /// Return the builder at the end of the region that the decl contains.
  OpBuilder getDeclEndBuilder() {
    Operation *op = getIfOperation();
    assert(op && op->getNumRegions() == 1 &&
           "can't get builder for this ASTDecl");
    return OpBuilder::atBlockEnd(&op->getRegion(0).front());
  }

  /// This return the 'Self' type for a struct or trait, which includes
  /// parameters bound to references to the struct parameter declarations.
  ASTType getTypeDeclSelf() const {
    assert(resolvedness != DeclResolvedness::unparsed &&
           "signature must be resolved to get a resolved type");
    return typeDeclSelf;
  }
  void setTypeDeclSelf(ASTType type) {
    assert(type && "Cannot set null types");
    typeDeclSelf = type;
  }

  /// Given an MLIR op for a struct declaration, return the self type.
  static Type computeSelfTypeForStruct(StructDeclOp structOp);

  /// Given an MLIR op for a trait declaration, return the self type.
  static Type computeSelfTypeForTrait(TraitDeclOp traitOp);

  /// Add an unresolved wild card import into this scope.
  void addUnresolvedWildcardImport(UnresolvedWildcardImport unresolvedImport);

  /// Record that `name` was imported with @stable(recursive=True) into this
  /// scope, suppressing stability warnings for that name and its members.
  void addRecursivelyStableName(mlir::StringAttr name);

  /// Return true if `name` is covered by a @stable(recursive=True) import
  /// override in this scope or any enclosing scope.
  bool hasRecursivelyStableName(mlir::StringAttr name) const;

  /// Return true if `typeDecl`'s user-visible name is covered by a
  /// @stable(recursive=True) import override in this scope. Returns false if
  /// `typeDecl` is null or has no user-visible name.
  bool hasRecursivelyStableType(const ASTDecl *typeDecl) const;

  /// Record that `name` was imported with @__doc_inline into this scope. Doc
  /// generation documents such a name here, using the target's declaration.
  void addDocInlineName(mlir::StringAttr name);

  /// Return true if `name` was imported into this scope with @__doc_inline.
  bool isDocInlineName(mlir::StringAttr name) const;

  /// Return the doc string for this decl, or nullptr if there isn't one.
  DocStringAttr getDocString() const;

  /// Return the parsed `DocString` for this decl if available.
  std::optional<DocString> getParsedDocString() const;

  /// Given a decl for a struct or trait type, check if this type conforms
  /// to the specified trait type. Returns a 3-state result:
  /// - `yes` if the type definitely conforms
  /// - `no` if the type definitely does not conform
  /// - `unknown` if conformance depends on constraints that cannot be
  ///   evaluated statically (e.g., generic parameters)
  ///
  /// If concreteType is provided, its parameter bindings are used to evaluate
  /// conditional trait conformances. If callerAssumptions is non-empty, those
  /// where-clause assumptions are used to prove unfoldable constraints.
  TriBool doesNominalTypeConformTo(TraitType trait, ASTType concreteType,
                                   ArrayRef<ConstraintAttr> callerAssumptions);

  /// Same, additionally collecting the problematic constraints so a diagnosing
  /// caller can name them.
  ConstraintResult doesNominalTypeConformToWithDetails(
      TraitType trait, ASTType concreteType,
      ArrayRef<ConstraintAttr> callerAssumptions);

  /// Find all extensions in this scope that target a specific struct.
  /// If filterTrait is provided, only returns extensions that implement that
  /// specific trait.
  void findExtensionsInScopeForStruct(
      SymbolRefAttr targetStruct, llvm::SmallPtrSetImpl<ASTDecl *> &results,
      std::optional<TraitSymbolAttr> filterTrait = std::nullopt);

  /// Collect all declarations that could contribute to a type's namespace.
  /// Examples:
  /// - If there's a struct Spaceship with two extensions, this will return all
  ///   three of those declarations.
  /// - If there's a trait Sporkable with two extensions, this will return all
  ///   three of those declarations.
  /// This is useful for collecting declarations in which we can search for
  /// methods for a given type, e.g. `my_ship.some_method()`, since the method
  /// might be in any of those declarations.
  llvm::SmallVector<ASTDecl *, 4> collectTypeAndExtensions(ASTType type,
                                                           llvm::SMLoc callLoc);

  /// If this is a method of a struct or trait, return the decl for the struct
  /// or trait.
  ASTDecl *tryGetMethodParentDecl() const;

  /// Whether this decl was loaded from bytecode.
  bool isLoadedFromBytecode() const { return loadedFromBytecode; }

  //===--------------------------------------------------------------------===//
  // Name lookup
  //===--------------------------------------------------------------------===//

  /// Look up a name in this declaration's scope only: return null on failure.
  ArrayRef<ASTDecl *> lookupInCurrentScope(StringAttr name) const;
  ArrayRef<ASTDecl *> lookupInCurrentScope(StringRef name) const;

  /// Perform a lookup in this declaration's scope and all parent scopes,
  /// returning the nearest target or empty if nothing is found.
  ArrayRef<ASTDecl *> lookup(StringAttr name) const {
    const ASTDecl *curScope = this;
    while (curScope) {
      ArrayRef<ASTDecl *> result = curScope->lookupInCurrentScope(name);
      if (!result.empty())
        return result;
      curScope = curScope->parentDecl;
    }
    return {};
  }

  /// Return an iterable set of declarations in this scope.  For lookups, use
  /// lookup or lookupInCurrentScope.
  ArrayRef<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>>
  getDeclsInScope() const {
    if (declsInScope)
      return ArrayRef(declsInScope->begin(), declsInScope->end());
    return {};
  }

  /// Given a reference to a parameter, look at this declaration and enclosing
  /// scopes to find the ASTDecl that defines it (e.g. the enclosing function,
  /// struct or trait).  This can return null if not found.
  std::tuple<const ASTDecl *, ArrayRef<ParamDeclAttr>, size_t>
  lookupParamReference(ParamDeclRefAttr paramRef) const;

  //===--------------------------------------------------------------------===//
  // Other State management.
  //===--------------------------------------------------------------------===//

  /// Indicate that the decl has reference errors.
  void setErroneous();
  /// Return true if the decl has reference errors.
  bool isErroneous() const { return hasReferenceError; }

  /// Return any decorators that need to be processed as part of body resolution
  /// phase for a decl.
  ArrayRef<ExprNode *> getBodyDecorators() const;

  /// During signature resolution, this is called with any decorators that need
  /// to persist until body resolution.
  void setBodyDecorators(ArrayRef<ExprNode *> decorators);

  /// Check if the given name collides with an existing user declared parameter
  /// name in the scope, and if so, uniquely mangle it by postpending a backtick
  /// ("`"), scope depth, and a unique ID.
  StringAttr mangleUserDefinedParamName(StringAttr name);

  /// Create a unique parameter name by postpending a backtick ("`"), scope
  /// depth, and a unique ID.
  StringAttr mangleParamName(const Twine &name);

  /// Move the children decls of `src` (along with their constraints) into this
  /// decl. This is useful when a temporary decl needs to be created for parsing
  /// subexpressions but whose children will be inherited later by a decl being
  /// resolved.
  void takeDecls(ASTDecl &src);

  /// Anonymous origins, closure impl structs, and potentially other names are
  /// uniqued to avoid collisions. This returns an ID that is unique to this
  /// ASTDecl instance and help generate such names.
  unsigned getNextUniqueID() { return counter++; }

  /// Mark this ASTDecl as an explicit parameter scope so that parameter
  /// mangling can be applied for in-flight types.
  void setExplicitParamScope() { isExplicitParamScope = true; }
  bool getIsExplicitParamScope() const { return isExplicitParamScope; }

  /// Get the map of trait conformance lineage for this decl. This is lazily
  /// initialized because it is only needed for structs.
  DenseMap<TraitSymbolAttr, std::pair<TraitSymbolAttr, SMLoc>> *
  getTraitConformanceLineage(bool createIfMissing = false);

  /// Get the set of assumptions for this decl and all its parent decls.
  void getKnownAssumptionsIncludingParents(
      SmallVectorImpl<ConstraintAttr> &assumptions) const;

  /// If scope is non-null, returns its assumptions (including parents);
  /// otherwise returns an empty vector.
  static llvm::SmallVector<ConstraintAttr>
  getAssumptionsFromScope(const ASTDecl *scope);

  /// Insert a set of assumptions into this decl.
  void insertKnownAssumptions(ArrayRef<ConstraintAttr> assumptions);

  /// Dump the underlying IR value.
  void dump() const;

  /// The pending wildcard imports into this scope.
  MutableArrayRef<UnresolvedWildcardImport> getUnresolvedWildcardImports() {
    if (!unresolvedWildcardImports)
      return {};
    return *unresolvedWildcardImports;
  }

private:
  friend class DeclResolver;
  friend class ModuleLoader;
  friend class SharedState;
  ASTDecl(SharedState &shared, DeclIRValue irValue, llvm::SMLoc loc,
          ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
          ssize_t indentation);
  ASTDecl(const ASTDecl &) = delete;
  ASTDecl &operator=(const ASTDecl &) = delete;

private:
  /// The Mojo shared state this decl is associated with.
  SharedState &shared;

  /// This is the IRValue or MLIR operation that this decl corresponds to if it
  /// has one.
  DeclIRValue irValue;

  /// This is the source location of the declaration, used for diagnostics and
  /// debug information.
  llvm::SMLoc loc;

  /// For a type declaration like a struct, this is the type of 'self' in a
  /// member.  This is only valid after signature resolution.
  ASTType typeDeclSelf;

  /// This the parent scope that should continue name lookup, or null for the
  /// top scope.
  ASTDecl *parentDecl;

  /// This is the cursor that points to the next part of declaration to continue
  /// parsing as the declaration is progressively resolved.
  LexerCursor cursor;

  /// This is the lexer cursor state for the first token /after/ the
  /// declaration.  This is used to make sure that bits of a declaration are not
  /// skipped in the early parse and not processes in the later parse.
  const char *endCursorState;

  /// This is the indentation level of the introducer keyword, useful for
  /// parsing the body of the declaration.  If the declaration was not at the
  /// start of a line or this is the top level module, then this is set to -1.
  ssize_t indentation;

public:
  /// This keeps track of what level of type checking this declaration has been
  /// through.  It is maintained by DeclResolver.
  DeclResolvedness resolvedness : 3; // Starts as DeclResolvedness::unparsed

private:
  /// When a bytecode decl depends on a source decl's children, we have to parse
  /// the signatures of all the children to register them in the symbol table.
  /// Cache this process using a flag on the decl.
  bool referencedFromBytecode : 1;

  /// True if any decl in this scope has been disabled, so name lookup must
  /// filter tombstones. Cache this using a flag on the decl.
  bool hasDisabledDecls : 1;

  /// This is set to true when an error is detected and reported about this
  /// declaration that could cause references to it to cause spurious downstream
  /// errors.  For example, "var x : SomeUndeclaredType" will cause errors for
  /// every reference to 'x' because the type will be bogus.
  bool hasReferenceError : 1;

  /// This is set to true if there is an entry for body-decorators in a
  /// backing hashtable.  Clients should use "getBodyDecorators().
  bool hasBodyDecorators : 1;

  /// This is set to true when the declaration was loaded from bytecode, not
  /// parsed from a textual source file. These declarations behave differently
  /// than source decls, and e.g., do not resolve in the same way as source
  /// decls.
  bool loadedFromBytecode : 1;

  /// True if this ASTDecl explicitly opts in to being treated as a parameter
  /// scope by `mangleParamName` even though its IR value isn't a
  /// `DeclInterface` op
  bool isExplicitParamScope : 1;

  /// The counter to allow the generation of unique IDs for this ASTDecl.
  unsigned counter = 0;

  /// These are the declarations defined within this scope.  This is lazily
  /// allocated the first time something is added, because it the vast majority
  /// of decls (leaves in the tree) don't need it.  (5-12% need it).
  ///
  /// This includes "owned" child decls whose `parentDecl` points back to this
  /// decl, as well as "inherited" child decls whose `parentDecl` points to
  /// other decls (as is the case for trait composition decls, see STCASTD).
  ///
  /// It also includes entries that are imported from other scopes. When an
  /// UnresolvedImportOp is resolved, it will be replaced with pointers to
  /// ASTDecls that are *owned* by other ASTDecls.
  ///
  /// It also sometimes includes the same ASTDecl twice, under different names.
  /// For example, an `__extension MyStruct` will be known to its parent ASTDecl
  /// as "extension:MyStruct" as well as "extension:" (the latter name is shared
  /// by all extensions).
  /// TODO(MOCO-2674): Have a review for this "extension:" choice, in case
  /// others have concerns about it.
  ///
  /// TODO(MOCO-522): Arcana docs on all the above, and on how importing works
  /// in general.
  ///
  /// TODO: Properly model inherited vs. owned decls.
  using DeclInScopeType = llvm::MapVector<StringAttr, TinyPtrVector<ASTDecl *>>;
  std::unique_ptr<DeclInScopeType> declsInScope;

  /// A set of modules with unresolved wildcard imports into this decl. This is
  /// lazily initialized because it is very rarely needed (~0.6% of all decls).
  /// This list only ever grows; UnresolvedWildcardImports are instead flagged
  /// as superseded once resolved or once a later import of the same module
  /// replaces it.
  using UnresolvedWildcardImportsType =
      llvm::SmallVector<UnresolvedWildcardImport>;
  std::unique_ptr<UnresolvedWildcardImportsType> unresolvedWildcardImports;

  /// Lazily-allocated set of import names that were imported with
  /// @stable(recursive=True) into this scope.  These names suppress stability
  /// warnings for the named binding and all member accesses through it.
  std::unique_ptr<llvm::DenseSet<mlir::StringAttr>> recursivelyStableNames;

  /// Lazily-allocated set of import names brought into this scope with
  /// @__doc_inline.
  std::unique_ptr<llvm::DenseSet<mlir::StringAttr>> docInlineNames;

  /// A map from each trait symbol that a struct conforms to, to the first
  /// symbol that explicitly inherits from it. This provides better diagnostics
  /// when a struct does not conform to a trait. This is lazily initialized
  /// because it is only needed for structs.
  using TraitConformanceLineageType =
      DenseMap<TraitSymbolAttr, std::pair<TraitSymbolAttr, SMLoc>>;
  std::unique_ptr<TraitConformanceLineageType> traitConformanceLineage;

  /// This is the set of constraints that can be assumed to be true inside
  /// this declaration. It may reference parameter declarations from this
  /// ASTDecl or its parent ASTDecls.
  std::unique_ptr<llvm::SetVector<ConstraintAttr>> knownAssumptions;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_ASTDECL_H
