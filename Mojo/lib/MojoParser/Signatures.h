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
// This contains logic for parsing and type checking and IR building of
// signatures for structs, functions, and function types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_SIGNATURES_H
#define KGEN_MOJOPARSER_SIGNATURES_H

#include "DeferredTypingContext.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/Lexer.h"

#include <optional>

namespace mlir {
class raw_indented_ostream;
} // namespace mlir

namespace M::KGEN::LIT {
class ASTDecl;
class ASTType;
class IREmitter;
class ExprNode;
class FnTypeGeneratorType;
class ParserBase;

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

/// This specifies the handling of keyword arguments in a list.
enum class KWArgHandling {
  kInferred,            //< before a standalone '//'
  kPositionalOnly,      //< before a standalone '/'
  kPositionalOrKeyword, //< before a standalone '*'
  kKeywordOnly          //< after a standalone '*'
};

enum class KWArgMarkerInfo {
  kNotMarker,  //< This is a normal argument.
  kSlashSlash, //< This argument is a standalone '//' marker.
  kSlash,      //< This argument is a standalone '/' marker.
  kStar,       //< This argument is a standalone '*' marker.
};

enum class ArgListKind {
  kParamList,         //< parameter list like `[x: Int, y: Int]`
  kArgList,           //< argument list like `(x: Int, y: Int)`
  kFnTypeArgList,     //< def type, like `def (Int, y: Float)`
  kFnTypeParamList,   //< def type, like `def [Int, y: Float](x: Int)`
  kBareLambdaArgList, //< argument list like `lambda x, y: x+y`
};

enum class CaptureConvention : uint8_t {
  kConventionMove = 0,
  kConventionMut = 1,
  kConventionCopy = 2,
  kConventionRead = 3,
  kConventionRef = 4,
  kConventionTrivialCopy = 5,
  kConventionUnspecified = 6
};

/// Note that this type may be stored in bump pointer allocated state, so it
/// cannot have a destructor!
struct ParsedConstraint {
  SMLoc loc;
  ExprNode *propExpr;
  /// Optional user-provided failure message from `where (condition,
  /// "message")` syntax. Null if no message was written.
  StringAttr message;

  ParseResult parse(ParserBase &p);

  /// Split the parsed `where` expression into `propExpr` (the condition) and
  /// `message`. If `parsed` has the shape `(condition, "message")` -- a
  /// parenthesized two-element tuple whose second element is a string literal
  /// -- both fields are set; a tuple whose second element is not a string
  /// literal is a targeted error (only string-literal messages are supported).
  /// Any other expression becomes `propExpr` as the bare condition.
  ParseResult extractParenthesizedMessage(ParserBase &p, ExprNode *parsed);

  /// Print the constraint for debugging.
  void print(mlir::raw_indented_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

/// Parsing support for a function argument and parameter:
///
/// argument_list      ::= argument ("," argument)*
/// argument           ::= "/" | "*"
/// argument           ::= [argument_convention] [argument_variadic] identifier
///                        [argument_type] ["=" expression]
/// argument_convention ::= "var" | "imm" | "mut" | "out" | "deinit"
/// argument_variadic  ::= "*" | "**"
/// argument_type      ::= ":" star_expression
///
/// Note that this  type is stored in a bump pointer allocated ExprNode, so
/// it cannot have a destructor!
struct ParsedArgument {
  SMLoc loc;
  LexerCursor cursor;
  // Specify argument passing convention, e.g. owned/mut etc.
  enum PAArgConvention {
    kConventionUnspec = 0,      // Nothing specified
    kConventionMut = 1,         // mut x
    kConventionVar = 2,         // var x
    kConventionImm = 3,         // imm x
    kConventionRef = 4,         // ref [origin, addrspace] x
    kConventionOut = 5,         // out [addrspace] x
    kConventionDeinit = 6,      // deinit self
    kConventionByRefResult = 7, // No syntax: result slot
  } convention = kConventionUnspec;

  // After type checking, this will hold the KGEN convention to use.
  ArgConvention kgenConvention = ArgConvention(128);

  // if there is a variadic argument, the convention of the variadic argument
  // will be stored here.
  ArgConvention variadicArgConvention = ArgConvention(128);

  VariadicKind variadicKind = VariadicKind::None;
  StringAttr name;
  ExprNode *typeExpr = nullptr;
  ExprNode *initExpr = nullptr;
  // If this is a ref convention, this specifies the origin expression.
  ExprNode *refOriginExpr = nullptr;

  /// This gets set to true when there is a /diagnosed/ error that should
  /// prevent subsequent references to this argument.
  mutable bool isErroneous = false;

  KWArgHandling kwArgHandling = KWArgHandling::kPositionalOrKeyword;

  ParseResult parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                    ArgListKind kind);

  /// Map KWArgHandling to the PassingKind enum of the LIT dialect.
  PassingKind getKWArgHandlingAsPassingKind() const;

  /// Print the argument for debugging.
  void print(mlir::raw_indented_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

//===----------------------------------------------------------------------===//
// ParsedParamList
//===----------------------------------------------------------------------===//

/// This is all the state built up when parsing the parameter signature for a
/// parameterized declaration, (e.g. a function or struct).
class ParsedParamList {
public:
  /// The full ParsedArgument for each parameter.
  SmallVector<ParsedArgument> params;

  /// Trailing constraints specified with 'where' clauses after the signature.
  SmallVector<ParsedConstraint> bodyConstraints;

  /// Parse a parameter signature if present.
  ///
  /// param_signature    ::= "[" param_list ("->" param_result_types)? "]"
  /// param_list   ::= argument_list | "(" ")"
  /// param_result_types ::= expression ("," expression)*
  ParseResult parseParametersIfPresent(ParserBase &p, ArgListKind kind);

  /// Parse trailing constraints if present.
  ///
  /// constraint_clauses ::= ("where" expression)*
  /// A message is carried inside the expression as a parenthesized pair:
  /// `where ( condition , string_literal )`.
  ParseResult parseTrailingConstraintsIfPresent(ParserBase &p);
};

/// This contains the result state from type checking a parameter signature.
class TypeCheckedParamList {
private:
  /// Private constructor - use the static create method instead.
  TypeCheckedParamList(ASTDecl &declScope);

public:
  /// This is the declaration that we do name lookup against.
  ASTDecl &declScope;
  SharedState &shared;

  /// Type check each of the parameters from 'parsedParams' into their
  /// decomposed representation. Returns nullopt if type checking fails.
  static std::optional<TypeCheckedParamList>
  create(ParsedParamList &parsedParams, ASTDecl &declScope);

  /// Emit the trailing `where` clauses parsed into `emittedBodyConstraints`.
  /// This cannot be performed at `create` because the body constraints may
  /// reference function arguments (in the case of a function trailing
  /// constraint), which requires the function signature be parsed and its
  /// arguments already registered in `declScope`.
  void emitBodyConstraints();

  /// Get an PogListAttr for this parameter list.
  PogListAttr getParamListAttr() const;

  // These are the results of type checking 'params' in typeCheck.
  /// One ParamDeclAttr for each parameter being declared.
  SmallVector<ParamDeclAttr> paramDeclAttrs;
  SmallVector<StringAttr> names;
  SmallVector<PassingKind> passingKinds;
  SmallVector<VariadicKind> variadicKinds;
  SmallVector<SMLoc> locations;

  /// Default values for params.
  SmallVector<TypedAttr> defaults;

  /// Deferred typing context from emitting parameter declaration types.
  DeferredTypingContext deferredTypingContext;

  /// Constraints emitted for the body of the generator.
  SmallVector<ConstraintAttr> emittedBodyConstraints;

  /// Un-emitted body constraints that will be processed later during
  /// `emitBodyConstraints`.
  SmallVector<ParsedConstraint> stagedBodyConstraints;
};

//===----------------------------------------------------------------------===//
// ParsedArgumentList
//===----------------------------------------------------------------------===//

/// This is all the state built up when parsing a function signature.
class ParsedArgumentList {
public:
  /// Any arguments specified.
  SmallVector<ParsedArgument> parsedArgs;
  /// The result specifier if present. This is usually something like "Int" for
  /// () -> Int.  It can also be "out [o] Int" with convention=kConventionOut,
  /// and can also be "() -> ref [o] Int" with kConventionRef.
  ParsedArgument resultArg;
  FnEffects effects;

  /// Tracks the Mojo-only `thin` effect on function types.
  bool isThin = false;

  bool isExperimentalParamTrait = false;
  /// True if an explicit `abi(...)` effect was written on this function.
  /// Distinguishes `abi("Mojo")` (no-op on the type, but explicit) from the
  /// absence of any abi annotation.
  bool hasExplicitABI = false;
  ExprNode *thrownTypeExpr = nullptr;

  /// Parse an argument list, including the parentheses around them. This also
  /// parses 'raises' and other effects.
  ParseResult parseArgumentListAndEffects(ParserBase &p, ArgListKind kind);

  /// Parse the result specifier starting with a `->` if present.
  void parseResultIfPresent(ParserBase &p,
                            std::optional<size_t> stmtIndent = std::nullopt);

  /// Returns true when a function type should be interpreted as a closure
  /// trait rather than a thin function pointer or legacy capturing function.
  bool isClosureFunctionType() const {
    return !effects.isCapturing() && !isThin;
  }
};

/// This is all the state built up when parsing a capture signature.
class ParsedCaptureList {
public:
  /// Any arguments specified.
  SmallVector<std::tuple<StringRef, CaptureConvention, SMLoc>> parsedCaptures;

  /// True if a `{...}` capture list was written explicitly.
  bool hasExplicitCaptureList = false;

  /// default capture convention, if exists.
  std::optional<CaptureConvention> captureAllByConvention;

  /// Parse a capture list
  ParseResult parseCaptureList(ParserBase &p);

  /// Parse the result specifier starting with a `->` if present.
  void parseResultIfPresent(ParserBase &p,
                            std::optional<size_t> stmtIndent = std::nullopt);
};

/// This contains the result state from type checking a parameter signature.
class TypeCheckedFnSignature {
public:
  /// Emit the argument types, default values, and result type and determine
  /// the argument conventions.
  ///
  /// 'fnDecl' will be null when this is a function type, which doesn't have a
  /// declaration.
  TypeCheckedFnSignature(TypeCheckedParamList &paramList,
                         ParsedArgumentList &argList,
                         const ExprNode *originExpr, ASTDecl *fnDecl,
                         StringAttr baseName);
  TypeCheckedParamList &paramList;
  ParsedArgumentList &argList;

  /// This indicates whether the function is a special function like __init__.
  SpecialFunctionInfo fnInfo;

  /// For methods, this is the default type of Self.  For global functions this
  /// is null.
  ASTType selfType;

  /// Whether `@__allow_legacy_custom_self_type` was specified.
  bool allowCustomSelfType = false;

  // This is the type checked declared argument type, e.g. "String" or "Int".
  SmallVector<Type> argTypes;
  /// Default values for args.
  SmallVector<TypedAttr> defaults;
  ASTType resultType;

  /// This is the type checked argument types with argument conventions and
  /// origins applied, e.g. "!lit.ref<String>" or "!kgen.param_list<Int>"
  SmallVector<Type> fullArgTypes;
  SmallVector<ParamDeclAttr> implicitOriginDecls;

  /// This is the result type + variant for throwing functions.  This is what
  /// finally gets treated as the ABI for the function.
  ASTType fullResultType;

  /// This is an optional origin set parameter, representing the origins of
  /// the function captures.
  TypedAttr captureOrigins;
  /// Whether `@__unsafe_nested_origins_read_only` was specified:
  /// nested origins are not considered in exclusivity checking.
  /// TODO: Generalize this to mutation sets.
  bool isNestedOriginsReadOnly = false;
  /// Whether the function has an explicitly declared return type containing an
  /// interior origin, so calls establish those interior origins.
  bool definesInteriorOrigins = false;

  /// This performs any special checks over the declaration based on its name
  /// and whether it is a method.  This happens after decorator processing
  /// because that is how defs work in Python.
  ///
  /// If this function detects a problem, it marks the decl as erroneous and
  /// resets fnInfo. `name` is updated in place to the canonical spelling of
  /// the function's name (e.g. a deprecated '__del__' becomes '__deinit__'),
  /// so the caller's downstream use of the name (mangling, the symbol and
  /// source-name attributes) always sees the canonical form.
  void verifyFunctionNameBinding(ASTDecl &decl, StringAttr &name);

  /// In a method with a self argument, check to make sure it has the correct
  /// type and invariants. "isSelfResult" is true for initializers that return
  /// Self, they don't take it as an input argument.
  void checkSelfArgument(ASTDecl &decl, ASTType selfArgType,
                         const ParsedArgument &selfArg,
                         bool isSelfResult) const;

  /// Return a FunctionType with the specified argTypes and resultType.
  FunctionType getFunctionType() const;

  /// Form a LIT signature packaging up all the stuff we need to know about this
  /// type checked function.
  FnTypeGeneratorType getFnTypeGeneratorType() const;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_SIGNATURES_H
