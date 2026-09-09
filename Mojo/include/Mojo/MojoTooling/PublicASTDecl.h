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

#ifndef KGEN_MOJOTOOLING_PublicASTDecl_H
#define KGEN_MOJOTOOLING_PublicASTDecl_H

#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/SignatureModel.h"
#include "Mojo/MojoTooling/TypeMetadata.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>
namespace llvm {
namespace json {
class Object;
} // namespace json
} // namespace llvm

namespace M {

using KGEN::TypeMetadata;

namespace KGEN {
enum class PassingKind : uint32_t;
enum class VariadicKind : uint32_t;
namespace LIT {
class ASTDecl;
class SharedState;
class FnTypeGeneratorType;
} // namespace LIT
} // namespace KGEN

class MojoASTDeclRef;
class MojoParserContext;

namespace {
/// Typical max trait composition size in Mojo APIs.
/// Currently only 1 case exceeds this: builtin.Comparable with 5 traits.
static constexpr size_t kTypicalTraitCompositionSize = 4;

/// Typical max parameter count for Mojo functions, structs, and aliases.
/// Currently about 98% of declarations use ≤2 parameters.
static constexpr size_t kTypicalParameterCount = 2;

/// Typical max argument count for Mojo functions.
/// Currently about 95% of functions use ≤2 arguments.
static constexpr size_t kTypicalArgumentCount = 2;
} // namespace
class PublicDecl;
class MojoASTTypeRef;
enum class PublicDeclKind;

/// Rendered `where` conditions of a decl's conditional trait conformances,
/// keyed by the conformed trait's symbol. Unconditional conformances are
/// absent. The key is the symbol rather than the trait's short name because
/// distinct symbols can share one name.
using ConformanceConditionMap =
    llvm::DenseMap<KGEN::TraitSymbolAttr, std::string>;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// This class is a wrapper around an internal Mojo AST declaration.
class MojoASTDeclRef {
public:
  MojoASTDeclRef(KGEN::LIT::ASTDecl *decl = nullptr) : decl(decl) {}

  /// Returns the operation corresponding to this decl if there is one, nullptr
  /// otherwise. The returned operation should only be used for introspection,
  /// it should not be modified in any way.
  Operation *getIfOperation() const;

  /// Returns if the AST declaration is valid.
  operator bool() const { return decl != nullptr; }

  /// Access the underlying AST decl.
  KGEN::LIT::ASTDecl &operator*() { return *decl; }
  KGEN::LIT::ASTDecl *operator->() { return decl; }
  const KGEN::LIT::ASTDecl &operator*() const { return *decl; }
  const KGEN::LIT::ASTDecl *operator->() const { return decl; }
  bool operator==(const MojoASTDeclRef &other) const {
    return decl == other.decl;
  }

  /// Returns the type corresponding to this declaration. If not available, this
  /// returns an invalid `MojoASTTypeRef`.
  MojoASTTypeRef getType() const;

  /// Get the name of this declaration if available.
  std::optional<StringRef> getName() const;

  /// Get the deprecation warning on this declaration if available.
  std::optional<StringRef> getDeprecationWarning() const;

  /// Get the location of the start token of this decl. It might not be the
  /// identifier.
  llvm::SMLoc getLoc() const;

  /// Get the parent MojoASTDeclRef of this decl.
  MojoASTDeclRef getParent() const;

  /// Get the SharedState for this decl if non-null.
  KGEN::LIT::SharedState *getShared() const;

  /// Get a PublicDecl that can be used for more easily inspecting the metadata
  /// of this decl. It supports aliases, modules, functions, structs, arguments,
  /// struct fields and variables.
  std::unique_ptr<PublicDecl> getDecl() const;

  /// Return the approximate kind of decl that would be created by would created
  /// by `getDecl`, if any. This isn't guaranteed to be the exact decl kind, but
  /// it can be used to provide a fast indicator.
  std::optional<PublicDeclKind> getApproximateDeclKind() const;

  //===--------------------------------------------------------------------===//
  // Children
  //===--------------------------------------------------------------------===//

  /// This class represents an individual child entry. It contains the name of
  /// the child declaration, and the group of declarations that share the same
  /// name.
  class ChildEntry {
  public:
    /// Return the name of this entry.
    StringRef getName() const { return name; }

    /// Return the declarations within this entry.
    auto getDecls() const {
      return llvm::map_range(rawEntries, [](KGEN::LIT::ASTDecl *entry) {
        return MojoASTDeclRef(entry);
      });
    }

  private:
    friend MojoASTDeclRef;

    /// Constructs a new child entry.
    ChildEntry(StringRef name, ArrayRef<KGEN::LIT::ASTDecl *> rawEntries)
        : name(name), rawEntries(rawEntries) {}

    /// The name of this entry.
    StringRef name;

    /// The raw entry array.
    ArrayRef<KGEN::LIT::ASTDecl *> rawEntries;
  };

  /// This class defines an iterator over the children of a declaration.
  class ChildIterator
      : public llvm::indexed_accessor_iterator<ChildIterator,
                                               KGEN::LIT::ASTDecl *, ChildEntry,
                                               ChildEntry, ChildEntry> {
  public:
    /// Accesses the entry at the current position.
    ChildEntry operator*() const;

  private:
    friend MojoASTDeclRef;

    /// Constructs a new iterator.
    ChildIterator(MojoASTDeclRef decl, size_t index);
  };

  /// Return the children of this declaration.
  llvm::iterator_range<ChildIterator> getChildren() const;

private:
  using ApproximatePublicDeclKind = std::optional<PublicDeclKind>;
  using PublicDeclInstance = std::unique_ptr<PublicDecl>;

  /// The documentation for this method is in the corresponding cpp file.
  template <typename ResultType, typename PublicDeclT, typename... DeclArgs>
  ResultType createPublicDecl(DeclArgs &&...declArgs) const;

  /// The documentation for this method is in the corresponding cpp file.
  template <typename ResultType>
  ResultType getDeclImpl() const;

  /// Allow MojoParserContext to access the internal impl.
  friend class MojoParserContext;

  /// The ASTDecl being referenced.
  KGEN::LIT::ASTDecl *decl;
};

//===----------------------------------------------------------------------===//
// MojoASTTypeRef
//===----------------------------------------------------------------------===//

/// This class provides a wrapper around an internal Mojo AST type.
class MojoASTTypeRef {
public:
  MojoASTTypeRef() : MojoASTTypeRef(nullptr) {}
  MojoASTTypeRef(KGEN::LIT::ASTType type) : type(type) {}
  MojoASTTypeRef(Type type) : type(type) {}
  MojoASTTypeRef(const void *impl) : type(Type::getFromOpaquePointer(impl)) {}

  /// Returns if the AST declaration is valid.
  operator bool() const { return bool(type); }

  /// Returns a readable string representation of this type.
  std::string getAsString(KGEN::LIT::SharedState &shared) const;

  /// If the current type is a reference, return the type of the pointee. This
  /// aborts if the current type isn't a reference.
  MojoASTTypeRef getReferenceElementType() const;

  /// Return the MLIR type associated with this
  Type getMLIRType() const;

private:
  // Return the decl that defined this type.
  MojoASTDeclRef getDecl(KGEN::LIT::SharedState &sharedState);

  /// Allow MojoParserContext to access the internal implementation.
  friend class MojoParserContext;

  /// The internal implementation of the AST type.
  KGEN::LIT::ASTType type;
};

//===----------------------------------------------------------------------===//
// PublicDecls
//
// The following classes provide simple and JSON-serializable representations of
// the most common Mojo decls. They also include structured documentation from
// the corresponding DocStrings and the ability to summarize themselves as code
// snippets.
//
// The API for these decls is expected to be somewhat stable because other tools
// rely on it, hence its `Public` name.
//===----------------------------------------------------------------------===//

/// Stability information extracted from the @stable decorator on a decl.
/// isStable=false means no @stable decorator. sinceVersion is empty when
/// the decorator was bare (@stable with no arguments).
/// isStabilityTracked=true means the symbol is in a package that has opted
/// into stability tracking (currently only "std").
struct StableInfo {
  bool isStable = false;
  bool isStabilityTracked = false;
  std::string sinceVersion;
};

/// The different kinds of public decls.
enum class PublicDeclKind {
  DK_PublicAliasDecl,
  DK_PublicArgumentDecl,
  DK_PublicFunctionDecl,
  DK_PublicModuleDecl,
  DK_PublicPackageDecl,
  DK_PublicParameterDecl,
  DK_PublicStructDecl,
  DK_PublicStructFieldDecl,
  DK_PublicTraitDecl,
  DK_PublicVariableDecl,
};

/// Base class of all public decls.
class PublicDecl {
public:
  virtual ~PublicDecl() = default;

  /// Generate a correct piece of code that summarizes this decl.
  virtual std::string getDeclarationSnippet(MojoParserContext &ctx) const = 0;

  /// Get the name of the decl. It might be empty.
  StringRef getName() const { return name; }

  /// Get a string representation of the kind of decl, e.g., 'variable',
  /// 'function', etc.
  StringRef getKindAsString() const;

  static StringRef getKindAsString(PublicDeclKind kind);

  /// Serialize the fields in this decl to JSON.
  virtual llvm::json::Object toJSON(MojoParserContext &ctx) const = 0;

  /// Return a nicely formatted markdown docstring of this declaration. It might
  /// be empty if no docstring is available.
  virtual std::string getMarkdownDocString() const { return {}; }

  /// Return a nicely formatted markdown blob containing the declaration snippet
  /// and doc string of the decl.
  std::string getFullMarkdownString(MojoParserContext &ctx) const;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  PublicDeclKind getKind() const { return kind; }

protected:
  PublicDecl(PublicDeclKind kind, StringRef name) : kind(kind), name(name) {}

private:
  PublicDeclKind kind;
  StringRef name;
};

/// Decl for `var`.
class PublicVariableDecl : public PublicDecl {
public:
  /// Get the type string of this variable
  StringRef getType() const { return type; }

  /// Return if this variable is global.
  bool isGlobal() const { return isGlobalVariable; }

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "variable",
  ///    "name": string,
  ///    "type": string,
  ///    "path": string
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicVariableDecl;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicVariableDecl(MojoASTDeclRef declRef);

  std::string type;
  TypeMetadata typeMetadata;
  bool isGlobalVariable;
  StringRef deprecated;
};

/// Decl for parameters of structs or functions.
class PublicParameterDecl : public PublicDecl {
public:
  /// Build a documented parameter from a fully-resolved `ParameterInfo`
  /// (which carries the type string with any conforms_to trait bounds
  /// already merged in). Type metadata for doc cross-linking is computed
  /// once during construction.
  PublicParameterDecl(KGEN::ParameterInfo info,
                      KGEN::LIT::SharedState &sharedState,
                      const MojoASTDeclRef *currentDeclContext = nullptr);

  KGEN::PassingKind getPassingKind() const { return info.passingKind; }

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  /// Get constraints of this declaration as string. It might be empty.
  StringRef getConstraints() const { return info.constraints; }

  /// The underlying signature-shape view of this parameter, used by both the
  /// rendering algorithm and JSON serialization.
  const KGEN::ParameterInfo &getInfo() const { return info; }

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "parameter",
  ///    "description": string,
  ///    "name": string,
  ///    "type": string,
  ///    "path": string,
  ///    "passingKind": string,
  ///    "defaultValue": string?
  ///    "constraints": string?
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicParameterDecl;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  KGEN::ParameterInfo info;
  TypeMetadata typeMetadata;
  SmallVector<TypeMetadata, kTypicalTraitCompositionSize> traitMetadata;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// Decl for function arguments, including varargs arguments.
class PublicArgumentDecl : public PublicDecl {
public:
  /// The convention this argument is passed with. Aliased to the shared
  /// signature model's `ArgumentConvention` so callers can use either name.
  using Convention = KGEN::ArgumentConvention;

  /// Build a documented argument from a fully-resolved `ArgumentInfo`.
  PublicArgumentDecl(KGEN::ArgumentInfo info,
                     KGEN::LIT::SharedState &sharedState,
                     const MojoASTDeclRef *currentDeclContext = nullptr);

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  Convention getConvention() const { return info.convention; }

  KGEN::PassingKind getPassingKind() const { return info.passingKind; }
  void setPassingKind(KGEN::PassingKind kind) { info.passingKind = kind; }

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  /// The underlying signature-shape view of this argument, used by both the
  /// rendering algorithm and JSON serialization.
  const KGEN::ArgumentInfo &getInfo() const { return info; }
  KGEN::ArgumentInfo &getInfo() { return info; }

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "argument",
  ///    "name": string,
  ///    "description": string,
  ///    "convention": string, // "imm", "mut", "var"
  ///    "type": string,
  ///    "path": string,
  ///    "passingKind": string,
  ///    "defaultValue": string?
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicArgumentDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  KGEN::ArgumentInfo info;
  TypeMetadata typeMetadata;
  SmallVector<TypeMetadata, kTypicalTraitCompositionSize> traitMetadata;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// Decl for alias.
class PublicAliasDecl : public PublicDecl {
public:
  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the declaration snippet for the alias. The positions of parameters
  /// within the printed snippet may be extracted via `parameterOffsets`, which
  /// may be null if parameter offsets are not desired.
  std::string getDeclarationSnippet(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const;

  std::string getMarkdownDocString() const override;

  /// Return the parameters of this alias.
  ArrayRef<PublicParameterDecl> getParameters() const { return parameters; }

  StringRef getValue() const { return value; }

  /// Return the type of this alias if available.
  StringRef getType() const { return type; }

  /// Return if this alias is global.
  bool isGlobal() const { return isGlobalAlias; }

  /// Populate the parameter descriptions from a docstring.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  /// Generate a string for the signature of this alias, given its
  /// components. The positions of parameters within the printed signature may
  /// be extracted via the optionally null `parameterOffsets`.
  std::string getSignature(MojoParserContext &ctx,
                           SmallVectorImpl<std::pair<unsigned, unsigned>>
                               *parameterOffsets = nullptr) const;

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "alias",
  ///    "name": string,
  ///    "description": string,
  ///    "summary": string,
  ///    "parameters": PublicParameterDecl[],
  ///    "signature": string,
  ///    "value": string,
  ///    "type": string
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicAliasDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicAliasDecl(MojoASTDeclRef declRef);

  /// The underlying decl. Retained so that `getSignature` can delegate to the
  /// op-driven signature printer in MojoParser.
  MojoASTDeclRef decl;
  SmallVector<PublicParameterDecl, kTypicalParameterCount> parameters;
  std::string aliasConstraints;
  std::string value;
  std::string type;
  std::string docPath;
  bool isGlobalAlias;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string description;
  StableInfo stableInfo;
  std::string summary;
};

/// Decl for functions of any kind, including closures, instance methods, fn
/// functions, def functions, etc.
class PublicFunctionDecl : public PublicDecl {
public:
  /// Return true if this is an async function.
  bool isAsync() const { return isAsyncFlag; }

  /// Return true if this is an __init__ function decorated with the @implicit
  /// decorator
  bool isImplicitConversion() const { return isImplicitConversionFlag; }

  /// Return true if this is a static struct method, i.e., marked with @static.
  bool isStatic() const { return isStaticFlag; }

  /// Return true if this is a non-static struct method.
  bool isMethod() const { return isMethodFlag; }

  /// Return true if this trait method has a default implementation.
  bool hasDefaultImplementation() const { return isDefaultImplFlag; }

  /// Return the list of arguments of this function.
  ArrayRef<PublicArgumentDecl> getArguments() const { return args; }
  const PublicArgumentDecl &getArgument(size_t i) const { return args[i]; }

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the declaration snippet for the function. The positions of parameters
  /// and arguments within the printed signature may be extracted via the
  /// optional `parameterOffsets` and `argumentOffsets`. nullptr should be
  /// provided if collecting the offsets isn't desired.
  std::string getDeclarationSnippet(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets =
          nullptr) const;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// Return the parameters of this function.
  ArrayRef<PublicParameterDecl> getParameters() const { return parameters; }

  // TODO: always return the type, including None. The doc and snippet
  // generation will need to be updated accordingly.
  /// Return the type of this function if it is not None.
  std::optional<StringRef> getReturnType() const { return returnType; }

  /// Generate a string for the signature of this function, given its
  /// components. The positions of parameters, arguments, and the return type
  /// within the printed signature may be extracted via the optionally null
  /// `parameterOffsets`, `argumentOffsets`, and `returnOffset` parameters.
  std::string getSignature(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets =
          nullptr,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets = nullptr,
      unsigned *returnOffset = nullptr) const;

  /// Return true if this function raises.
  bool raises() const { return raisesFlag; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "function",
  ///   "name": string,
  ///   "args": PublicArgumentDecl[],
  ///   "constraints": string,
  ///   "description": string,
  ///   "hasDefaultImplementation": boolean,
  ///   "isImplicitConversion", boolean,
  ///   "isStatic": boolean,
  ///   "parameters": PublicParameterDecl[],
  ///   "raises": boolean,
  ///   "raisesDoc": string,
  ///   "returns": {"type": string, "path": string?, "doc": string?},
  ///   "signature": string, // E.g., "baz() -> Int"
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicFunctionDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicFunctionDecl(MojoASTDeclRef declRef);
  PublicFunctionDecl(MojoASTDeclRef declRef,
                     KGEN::LIT::FnTypeGeneratorType signature);

  /// Initialize the function decl with the given signature.
  void initFromSignature(MojoASTDeclRef declRef,
                         KGEN::LIT::FnTypeGeneratorType signature,
                         ArrayRef<Type> userArgTypes, Type userResultType);

  /// Augment this function decl with docstring documentation, as well as its
  /// parameters and args.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  /// The underlying decl. Retained so that `getSignature` can delegate to the
  /// op-driven signature printer in MojoParser when an `FnOp` is available.
  /// May be null if this `PublicFunctionDecl` was synthesized from a derived
  /// signature without a backing op.
  MojoASTDeclRef decl;
  SmallVector<PublicArgumentDecl, kTypicalArgumentCount> args;
  SmallVector<PublicParameterDecl, kTypicalParameterCount> parameters;
  std::string fnConstraints;
  // TODO: Convert this to MojoASTTypeRef.
  std::optional<std::string> returnType;

  //===----------------------------------------------------------------------===//
  // Effects and modifiers
  //===----------------------------------------------------------------------===//

  bool isAsyncFlag = false;
  bool isImplicitConversionFlag = false;
  bool isMethodFlag = false;
  bool isStaticFlag = false;
  bool raisesFlag = false;
  StableInfo stableInfo;
  bool isInit = false; // Is init or moveinit or copyinit.
  bool isDefaultImplFlag =
      false; // True if trait method has default implementation.

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string constraints;
  std::string description;
  std::string raisesDoc;
  std::string returnsDoc;
  std::string summary;
};

/// Decl for struct field.
class PublicStructFieldDecl : public PublicDecl {
public:
  PublicStructFieldDecl(StringRef name, StringRef type)
      : PublicDecl(PublicDeclKind::DK_PublicStructFieldDecl, name), type(type) {
  }

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  std::string getMarkdownDocString() const override;

  /// Return the type of this field.
  StringRef getType() const { return type; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "field",
  ///   "name": string,
  ///   "description": string,
  ///   "summary": string,
  ///   "type": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicStructFieldDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicStructFieldDecl(MojoASTDeclRef declRef);

  std::string type;
  std::string value;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
  std::string summary;
};

/// A collection of overloaded functions in the same scope.
class FunctionDeclOverloadSet {
public:
  /// Create a list of function overload sets by grouping the given functions by
  /// their name. It's assumed that the input list is sorted by name.
  static SmallVector<FunctionDeclOverloadSet, 2>
  fromSortedFunctions(SmallVector<PublicFunctionDecl, 2> &&functions);

  /// Get the common name of the functions in this overload set.
  StringRef getBaseName() const { return baseName; }

  /// Get the functions in this overload set.
  ArrayRef<PublicFunctionDecl> getFunctions() const { return functions; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "function",
  ///   "name": string,
  ///   "overloads": PublicFunctionDecl[]
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const;

private:
  FunctionDeclOverloadSet(StringRef baseName) : baseName(baseName) {}

  void append(PublicFunctionDecl function) {
    functions.push_back(std::move(function));
  }

  std::string baseName;
  SmallVector<PublicFunctionDecl, 2> functions;
};

// Decl for trait decls.
class PublicTraitDecl : public PublicDecl {
public:
  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  std::string getMarkdownDocString() const override;

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "trait",
  ///   "name": string,
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSet[],
  ///   "parentTraits": { "name": string, "path"?: string }[],
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicTraitDecl;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicTraitDecl(MojoASTDeclRef declRef);

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string description;
  StableInfo stableInfo;
  std::string summary;
  MojoASTDeclRef decl;
};

/// Decl for structs.
class PublicStructDecl : public PublicDecl {
public:
  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the declaration snippet for the struct. The positions of parameters
  /// within the printed snippet may be extracted via `parameterOffsets`, which
  /// may be null if parameter offsets are not desired.
  std::string getDeclarationSnippet(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const;

  std::string getMarkdownDocString() const override;

  /// Generate a string for the signature of this function, given its
  /// components. The positions of parameters within the printed signature may
  /// be extracted via the optionally null `parameterOffsets`.
  std::string getSignature(MojoParserContext &ctx,
                           SmallVectorImpl<std::pair<unsigned, unsigned>>
                               *parameterOffsets = nullptr) const;

  /// Return the parameters of this struct.
  ArrayRef<PublicParameterDecl> getParameters() const { return parameters; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "struct",
  ///   "name": string,
  ///   "aliases": PublicAliasDecl[],
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSet[],
  ///   "parameters": PublicParameterDecl[],
  ///   "parentTraits": {
  ///     "name": string,
  ///     "path"?: string,
  ///     // The `where` condition, present only for a conditional conformance
  ///     // such as `Writable where conforms_to(T, Writable)`.
  ///     "condition"?: string
  ///   }[],
  ///   "fields": PublicStructFieldDecl[],
  ///   "signature": string,
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicStructDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicStructDecl(MojoASTDeclRef declRef);

  /// Augment this struct decl with docstring documentation, as well as its
  /// parameters.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  SmallVector<PublicParameterDecl, kTypicalParameterCount> parameters;
  std::string structConstraints;

  ConformanceConditionMap conformanceConditions;

  KGEN::LIT::TypeConvention convention;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string constraints;
  std::string description;
  StableInfo stableInfo;
  std::string summary;
  MojoASTDeclRef decl;
};

/// Decl for module.
class PublicModuleDecl : public PublicDecl {
public:
  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "module",
  ///   "name": string,
  ///   "aliases": PublicAliasDecl[],
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSet[],
  ///   "structs": PublicStructDecl[],
  ///   "traits": PublicTraitDecl[],
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicModuleDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicModuleDecl(MojoASTDeclRef declRef);

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
  std::string summary;
  MojoASTDeclRef decl;
};

class PublicPackageDecl : public PublicDecl {
public:
  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "package",
  ///   "name": string,
  ///   "description": string,
  ///   "summary": string,
  ///   "modules": PublicModuleDecl[],
  ///   "packages": PublicPackageDecl[],
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicPackageDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PublicPackageDecl(MojoASTDeclRef declRef);

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
  std::string summary;
  MojoASTDeclRef decl;
};

} // namespace M

namespace llvm {
/// Cast from an MojoASTTypeRef to a mojo type.
template <typename T>
struct CastInfo<T, M::MojoASTTypeRef>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, M::MojoASTTypeRef,
                                     CastInfo<T, M::MojoASTTypeRef>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(M::MojoASTTypeRef astType) {
    if (!astType)
      return false;
    return T::classof(astType.getMLIRType());
  }

  static T doCast(M::MojoASTTypeRef astType) {
    return cast<T>(astType.getMLIRType());
  }
};

template <typename T>
struct CastInfo<T, const M::MojoASTTypeRef>
    : public ConstStrippingForwardingCast<T, const M::MojoASTTypeRef,
                                          CastInfo<T, M::MojoASTTypeRef>> {};

} // namespace llvm

#endif // KGEN_MOJOTOOLING_PublicASTDecl_H
