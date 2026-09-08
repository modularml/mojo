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

#include "Mojo/MojoParser/DeclResolver.h"

#include "ClosureEmitter.h"
#include "DLValues.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/ModuleLoader.h"
#include "MojoUtils.h"
#include "ParserBase.h"
#include "ParserEvaluationContext.h"
#include "Signatures.h"
#include "StabilityMarkers.h"
#include "StructEmitter.h"
#include "Traits.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Filesystem/Paths.h"

#include "Mojo/LITDialect/LITUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// If the given ASTDecl represents an extension, return the ASTDecl for its
/// target struct. If the given ASTDecl represents a struct, return the struct
/// itself. Returns nullptr if this is neither a struct nor an extension.
static ASTDecl *getStructOrTargetStruct(ASTDecl &decl,
                                        DeclResolver &declResolver) {
  if (auto extensionOp =
          dyn_cast_or_null<ExtensionDeclOp>(decl.getIfOperation())) {
    auto targetStructRefMaybe = extensionOp.getTargetStruct();
    if (targetStructRefMaybe)
      return &declResolver.getDeclForTypeSymbol(*targetStructRefMaybe);
  } else if (isa_and_nonnull<StructDeclOp>(decl.getIfOperation())) {
    return &decl;
  }
  return nullptr;
}

/// Parse an expression and immediately resolve it to a type.  This returns
/// failure on parse error.
static ParseResult parseType(ParserBase &p, ASTType &result, ASTDecl &declScope,
                             std::optional<size_t> stmtIndent,
                             bool allowUnbound) {
  ExprNode *expr = nullptr;
  if (p.parseExpression(expr, stmtIndent))
    return failure();

  IREmitter emitter(declScope, EC_Type);
  result = emitter.emitExprType(expr, allowUnbound);
  if (!result)
    return failure();

  return success();
}

static LogicalResult resolveDefaultedOpFromTrait(DeclResolver &resolver,
                                                 Operation *defaultedOp,
                                                 ASTDecl *structDecl) {
  auto traitFnDecl = defaultedOp->getParentOfType<TraitDeclOp>();

  auto traitSymbol =
      TraitSymbolAttr::get(getFullyResolvedSymbolRef(traitFnDecl));
  auto conformanceDecl =
      structDecl->lookupInCurrentScope(traitSymbol.getFlattenedName());

  return resolver.resolveBody(*conformanceDecl.front(),
                              conformanceDecl.front()->getLoc());
}
//===----------------------------------------------------------------------===//
// Decorator Support
//===----------------------------------------------------------------------===//

namespace {
/// Decorators attached to a declaration may be "signature" decorators or "body"
/// decorators.
///
/// - Signature decorators are applied during the resolution of the signature of
///   a declaration before it is name bound.
/// - Body decorators are applied after the body of the declaration is fully
///   resolved.
///
/// This is the base class for handling decorators on declarations. Signature
/// decorators are processed first and then leftover decorators are persisted
/// until body resolution is complete via the SharedState.
class Decorators : public SharedStateUser {
public:
  /// Create a class to handle decorators for a decl. If `signatureOnly` is set,
  /// the class will reject any decorator not processed during signature
  /// resolution.
  Decorators(ASTDecl &decl, bool signatureOnly = false)
      : SharedStateUser(decl.getShared()), decl(decl),
        signatureOnly(signatureOnly) {}

  /// Handle the `@deprecated` decorator for all decls.
  LogicalResult handleDeprecated(ExprNode *expr, ASTDecl &decl);

  /// Handle the `@unavailable` decorator for all decls.
  LogicalResult handleUnavailable(ExprNode *expr, ASTDecl &decl);

  /// Handle the `@stable` decorator for decls that implement
  /// StabilityDecoratorInterface.
  LogicalResult handleStable(ExprNode *expr, ASTDecl &decl);

  /// Handle the `@doc_hidden` decorator for decls that implement
  /// DocHiddenDecoratorInterface (AliasDeclOp and StructFieldOp).
  LogicalResult handleDocHidden(ExprNode *expr, ASTDecl &decl);

  /// Process signature decorators on the declaration using the provided
  /// functor. The functor should return success if the decorator was processed
  /// as a signature decorator. Any leftover decorators are emitted and deferred
  /// as body decorators.
  void applySignatureDecorators(
      ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
      function_ref<LogicalResult(ExprNode *)> process = [](ExprNode *) {
        return failure();
      });

  /// Process body decorators on the declaration using the provided functor.
  /// The functor should return success if the decorator was processed as a
  /// body decorator. Any leftover decorators are emitted and set on the
  /// operation.
  void applyBodyDecorators(function_ref<LogicalResult(ExprNode *)> process);

private:
  /// Validate compiler decorators that are allowed to propagate.
  LogicalResult validateCompilerDecorator(TypedAttr attr);

  /// Parse a message-or-use decorator and forward the resulting (reason,
  /// optional replacement) pair to `setter`. Centralizes the duplicated
  /// logic shared by `handleDeprecated` and `handleUnavailable`.
  ///
  /// `validateDecl` is an optional extra check applied after the
  /// `StabilityDecoratorInterface` check. It should emit its own error and
  /// return `failure()` if the decl kind is not allowed for this decorator;
  /// returning `success()` lets parsing continue. When unset, any decl that
  /// implements `StabilityDecoratorInterface` is accepted.
  LogicalResult handleMessageDecorator(
      ExprNode *expr, ASTDecl &decl, StringRef name,
      function_ref<void(StabilityDecoratorInterface itf, StringAttr reason,
                        StringAttr replacement)>
          setter,
      function_ref<LogicalResult(Operation *op)> validateDecl = {});

  /// The declaration this class is applying decorators to.
  ASTDecl &decl;
  /// Whether only signature decorators are allowed.
  bool signatureOnly;
};
} // namespace

LogicalResult Decorators::handleMessageDecorator(
    ExprNode *expr, ASTDecl &decl, StringRef name,
    function_ref<void(StabilityDecoratorInterface itf, StringAttr reason,
                      StringAttr replacement)>
        setter,
    function_ref<LogicalResult(Operation *op)> validateDecl) {
  // Detect bare `@<name>` and complain that a message must be explicitly
  // specified.
  if (auto declRef = dyn_cast<DeclRefNode>(expr);
      declRef && declRef->spelling == name) {
    std::string message = llvm::formatv(
        "@{0} requires a reason message or a replacement symbol (with "
        "'use')",
        name);
    std::string placeholder =
        llvm::formatv("(\"insert {0} message here\")", name);
    shared.emitError(expr->getLoc(), message) << FixIt::insertAfterToken(
        expr->getRange().getEnd(), placeholder, shared.diags);
    return success();
  }

  auto callNode = dyn_cast<CallNode>(expr);
  if (!callNode)
    return failure();
  auto declRef = dyn_cast<DeclRefNode>(callNode->callee);
  if (!declRef || declRef->spelling != name)
    return failure();

  // From here on, we've matched @<name> - all paths must return success().

  // Check that the decl's operation implements StabilityDecoratorInterface.
  Operation *op = decl.getIfOperation();
  auto stabilityInterface =
      op ? dyn_cast<StabilityDecoratorInterface>(op) : nullptr;
  if (!stabilityInterface) {
    shared.emitError(expr->getLoc())
        << "@" << name << " decorator is not supported on this declaration";
    return success();
  }

  // Apply the decorator-specific decl-kind restriction (if any). The
  // validator is expected to emit its own diagnostic on rejection.
  if (validateDecl && failed(validateDecl(op)))
    return success();

  if (callNode->operands.size() != 1) {
    shared.emitError(
        expr->getLoc(),
        llvm::formatv(
            "@{0} accepts either a reason message or a replacement symbol "
            "(with 'use')",
            name));
    return success();
  }

  auto &arg = callNode->operands.front();
  // Handle a positional string, or a `reason=` keyword argument.
  if (arg.isPositional() || (arg.isKeyword() && arg.name == "reason")) {
    auto strExpr = dyn_cast<StringLiteralNode>(arg.expr);
    if (!strExpr) {
      shared.emitError(arg.expr->getLoc(),
                       "'reason' argument must be a string literal");
      return success();
    }
    setter(stabilityInterface,
           StringAttr::get(getContext(), strExpr->getValue()),
           /*replacement=*/{});
    return success();
  }

  // Handle a `use=symbol` keyword argument.
  if (arg.isKeyword() && arg.name == "use") {
    auto target = dyn_cast<DeclRefNode>(arg.expr);
    if (!target) {
      shared.emitError(arg.expr->getLoc(), "'use' must reference a symbol");
      return success();
    }

    // If the decorated decl is a method, look for a sibling member in the
    // same struct/trait/extension. For non-methods (top-level functions,
    // structs, aliases), use the standard lookup with searchParentScopes=true.
    ASTDecl *methodParent = decl.tryGetMethodParentDecl();
    ASTDecl *scope = methodParent ? methodParent : decl.getParentDecl();
    LookupResult lookup = shared.lookupAndResolveDecl(
        target->spelling, target->getLoc(), *scope,
        /*searchParentScopes=*/methodParent == nullptr);

    if (lookup.isErroneous())
      return success();

    if (lookup.getIfSuccess().empty()) {
      shared.emitError(target->getLoc(), "cannot reference unknown value '")
          << target->spelling << "'";
      return success();
    }

    std::string sourceName;
    if (auto sym = dyn_cast<mlir::SymbolOpInterface>(op)) {
      sourceName = sym.getName();
    } else if (auto fn = dyn_cast<FnOp>(op)) {
      sourceName = fn.getSourceName() ? fn.getSourceName()->str()
                                      : "<anonymous function>";
    } else if (auto alias = dyn_cast<AliasDeclOp>(op)) {
      sourceName = demangleParameterName(alias.getParamDecl().getName());
    } else {
      assert(false && "unhandled case");
      sourceName = "<unhandled case>";
    }

    auto reason = StringAttr::get(
        getContext(), llvm::formatv("'{0}' is {1}, use '{2}' instead",
                                    sourceName, name, target->spelling));
    auto replacement = StringAttr::get(getContext(), target->spelling);
    setter(stabilityInterface, reason, replacement);
    return success();
  }

  shared.emitError(
      expr->getLoc(),
      llvm::formatv("{0} must specify either a message or a symbol (with the "
                    "'use' argument)",
                    name));
  return success();
}

LogicalResult Decorators::handleDeprecated(ExprNode *expr, ASTDecl &decl) {
  return handleMessageDecorator(
      expr, decl, "deprecated",
      [](StabilityDecoratorInterface itf, StringAttr reason,
         StringAttr replacement) {
        itf.setDeprecationInfoAttr(
            DeprecationInfoAttr::get(reason, replacement));
      });
}

LogicalResult Decorators::handleUnavailable(ExprNode *expr, ASTDecl &decl) {
  return handleMessageDecorator(
      expr, decl, "unavailable",
      [](StabilityDecoratorInterface itf, StringAttr reason,
         StringAttr replacement) {
        itf.setUnavailableInfoAttr(
            UnavailableInfoAttr::get(reason, replacement));
      },
      // @unavailable is only allowed on functions and methods. Structs,
      // traits, comptime aliases, etc. are rejected.
      [&](Operation *op) -> LogicalResult {
        if (isa<FnOp>(op))
          return success();
        shared.emitError(
            expr->getLoc(),
            "@unavailable can only be applied to functions and methods");
        return failure();
      });
}

LogicalResult Decorators::handleStable(ExprNode *expr, ASTDecl &decl) {
  // Common logic applied once @stable has been matched on a supported decl.
  auto applyStable = [&](StringAttr sinceVersion = {}) -> LogicalResult {
    Operation *op = decl.getIfOperation();
    auto stabilityInterface =
        op ? dyn_cast<StabilityDecoratorInterface>(op) : nullptr;
    if (!stabilityInterface) {
      shared.emitError(
          expr->getLoc(),
          "@stable decorator is not supported on this declaration");
      return success();
    }
    // Check for @stable member in an unstable struct/trait - this is an error.
    if (checkStableMemberInUnstableParent(decl, expr->getLoc(), shared))
      return success(); // Handled (with error).
    stabilityInterface.setHasStableDecorator(true);
    if (sinceVersion)
      stabilityInterface.setStableSinceVersionAttr(sinceVersion);
    return success();
  };

  // Handle @stable(...) call form.
  if (auto callNode = dyn_cast<CallNode>(expr)) {
    auto callee = dyn_cast<DeclRefNode>(callNode->callee);
    if (!callee || callee->spelling != "stable")
      return failure(); // Not our decorator.

    // From here on, we've matched @stable(...) - all paths must return
    // success().

    if (callNode->operands.size() != 1) {
      shared.emitError(
          expr->getLoc(),
          "@stable accepts only one argument, either 'since' or 'recursive'");
      return success();
    }

    auto &arg = callNode->operands.front();
    if (!arg.isKeyword()) {
      shared.emitError(arg.expr->getLoc(),
                       "@stable requires a keyword argument ('since' or "
                       "'recursive'), not a positional argument");
      return success();
    }

    if (arg.name == "recursive") {
      auto *boolNode = dyn_cast<BoolLiteralNode>(arg.expr);
      if (!boolNode || !boolNode->value) {
        shared.emitError(arg.expr->getLoc(),
                         "'recursive' argument to @stable must be True");
        return success();
      }
      // recursive=True is only valid on import statements, and import
      // decorator handling is done eagerly in parseFromImportDecorators.
      shared.emitError(expr->getLoc(),
                       "@stable(recursive=True) is only valid on import "
                       "statements");
      return success();
    }

    if (arg.name != "since") {
      shared.emitError(arg.expr->getLoc(),
                       "@stable only accepts keyword arguments 'since' or "
                       "'recursive'");
      return success();
    }

    auto strExpr = dyn_cast<StringLiteralNode>(arg.expr);
    if (!strExpr) {
      shared.emitError(arg.expr->getLoc(),
                       "'since' argument must be a string literal");
      return success();
    }

    // Own the string to avoid a dangling StringRef (getValue() returns
    // std::string by value).
    std::string versionStr = strExpr->getValue();
    StringRef version = versionStr;
    // Validate relaxed semver: non-empty, starts with a digit, contains only
    // alphanumeric characters and dots (e.g. "1.0", "1.2.3", "2.0rc1").
    if (version.empty() || !llvm::isDigit(version.front()) ||
        !llvm::all_of(version,
                      [](char c) { return llvm::isAlnum(c) || c == '.'; })) {
      shared.emitError(arg.expr->getLoc(),
                       "'since' argument must be a valid version string "
                       "(e.g. \"1.0\", \"1.2.3\", \"2.0rc1\")");
      return success();
    }

    return applyStable(StringAttr::get(getContext(), version));
  }

  // Handle bare `@stable` decorator (no arguments).
  auto declRef = dyn_cast<DeclRefNode>(expr);
  if (!declRef || declRef->spelling != "stable")
    return failure(); // Not a @stable decorator, let other handlers try.

  // From here on, we've matched @stable - all paths must return success().
  return applyStable();
}

LogicalResult Decorators::handleDocHidden(ExprNode *expr, ASTDecl &decl) {
  // Only handle bare `@doc_hidden` (no arguments) on declaration types that
  // cannot use the full decorator resolution machinery (IREmitter) without
  // causing recursive scope lookup issues: AliasDeclOp and StructFieldOp.
  // For FnOp, StructDeclOp, etc., @doc_hidden is deferred to body processing.
  auto declRef = dyn_cast<DeclRefNode>(expr);
  if (!declRef || declRef->spelling != "doc_hidden")
    return failure(); // Not our decorator.

  Operation *op = decl.getIfOperation();
  if (auto aliasOp = dyn_cast_if_present<AliasDeclOp>(op)) {
    aliasOp.setHasDocHiddenDecoratorAttr(UnitAttr::get(aliasOp.getContext()));
    return success();
  }
  if (auto fieldOp = dyn_cast_if_present<StructFieldOp>(op)) {
    fieldOp.setIsDocHiddenAttr(UnitAttr::get(fieldOp.getContext()));
    return success();
  }
  // For other op types, fall through to body decorator processing.
  return failure();
}

void Decorators::applySignatureDecorators(
    ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
    function_ref<LogicalResult(ExprNode *)> process) {
  // Process decorators in the order they are seen. Collect body decorators to
  // be deferred.
  SmallVector<ExprNode *> bodyDecorators;
  for (auto &[decorator, _] : decoratorExprs) {
    if (succeeded(handleDeprecated(decorator, decl)) ||
        succeeded(handleUnavailable(decorator, decl)) ||
        succeeded(handleStable(decorator, decl)) ||
        succeeded(handleDocHidden(decorator, decl)) ||
        succeeded(process(decorator)))
      continue;
    bodyDecorators.push_back(decorator);
  }

  if (!bodyDecorators.empty() && signatureOnly) {
    shared.emitError(bodyDecorators.front()->getLoc(),
                     "decorator on this statement is unsupported; remove, "
                     "replace, or correct the decorator")
        << SourceRange(bodyDecorators.front()->getRangeStart(),
                       bodyDecorators.back()->getRangeEnd());
    return;
  }

  // Check for mutual exclusivity between @deprecated, @stable, and
  // @unavailable. At most one of these may appear on a single declaration.
  if (auto stabilityInterface =
          dyn_cast_if_present<StabilityDecoratorInterface>(
              decl.getIfOperation())) {
    // Use the first decorator's location for the error message.
    SMLoc errorLoc = decoratorExprs.empty()
                         ? decl.getLoc()
                         : decoratorExprs.front().first->getLoc();
    if (stabilityInterface.isStable() && stabilityInterface.isDeprecated())
      shared.emitError(errorLoc,
                       "@deprecated and @stable cannot be used together");
    if (stabilityInterface.isUnavailable() && stabilityInterface.isDeprecated())
      shared.emitError(errorLoc,
                       "@unavailable and @deprecated cannot be used together");
    if (stabilityInterface.isUnavailable() && stabilityInterface.isStable())
      shared.emitError(errorLoc,
                       "@unavailable and @stable cannot be used together");
  }

  // Defer the rest of the decorators through the shared state.
  decl.setBodyDecorators(bodyDecorators);
}

// Helper function to extract symbol name from a TypedAttr
static std::optional<StringRef> extractDecoratorName(TypedAttr attr) {
  // Helper lambda to extract name from a symbol reference
  auto extractFromSymbolRef = [](SymbolRefAttr ref) -> StringRef {
    StringRef name = ref.getLeafReference().getValue();
    return name.substr(0, name.find_first_of("(["));
  };

  if (auto cst = dyn_cast<SymbolConstantAttr>(attr))
    return extractFromSymbolRef(cst.getSymbol());
  if (auto fnLiteral = dyn_cast<FnLiteralTypeGeneratorType>(attr.getType()))
    return extractDecoratorName(fnLiteral.getSymbolConstantAttr());

  if (auto call = dyn_cast<ParamOperatorAttr>(attr)) {
    // Only process if it's an Apply operator with at least one operand
    if (call.getOpcode() != POC::Apply || call.getOperands().empty())
      return std::nullopt;

    if (auto firstOp = dyn_cast<SymbolConstantAttr>(call.getOperands().front()))
      return extractFromSymbolRef(firstOp.getSymbol());
  }

  return std::nullopt;
}

LogicalResult Decorators::validateCompilerDecorator(TypedAttr attr) {
  constexpr StringRef plainDre[] = {
      "doc_hidden",
      "lldb_formatter_wrapping_type",

      KGEN::kFnRegisterInternal,
      KGEN::kFnRegisterShapeFunction,
      "enforce_io_param",

      KGEN::kFnRegister,
      "elementwise",
      "view_kernel",
      "mutable",
  };

  auto symbolName = extractDecoratorName(attr);
  if (!symbolName)
    return failure();

  if (auto call = dyn_cast<ParamOperatorAttr>(attr)) {
    return success(call.getOpcode() == POC::Apply &&
                   llvm::is_contained(plainDre, *symbolName) &&
                   call.getOperands().size() <= 6);
  }

  return success(llvm::is_contained(plainDre, *symbolName));
}

void Decorators::applyBodyDecorators(
    function_ref<LogicalResult(ExprNode *)> process) {
  // Don't run decorators if the declaration is invalid.
  if (decl.isErroneous())
    return;

  SmallVector<ExprNode *> exprDecorators;
  for (auto decorator : decl.getBodyDecorators())
    if (failed(process(decorator)))
      exprDecorators.push_back(decorator);

  // Emit the expressions and persist the resulting PValue into the IR.
  // TODO: Emit an attempt to call the decorator value.
  SmallVector<TypedAttr> decoPValues;
  decoPValues.reserve(exprDecorators.size());
  IREmitter emitter(decl, EC_Decorator);
  for (auto *decorator : exprDecorators) {
    if (PValue decoVal = emitter.emitExprPValue(decorator, EC_Decorator)) {
      // DecoVal wants the symbol constant attr.
      if (auto fnLit = dyn_cast<FnLiteralTypeGeneratorType>(decoVal.getType()))
        decoVal = PValue(fnLit.getSymbolConstantAttr());

      if (failed(validateCompilerDecorator(decoVal))) {
        emitError(decorator->getLoc(), "unsupported compiler decorator")
            << decorator->getRange();
      }
      decoPValues.push_back(decoVal);
    }
  }

  cast<ASTDeclInterface>(decl.getIfOperation())
      .setDecoratorsAttr(DecoratorsAttr::get(getContext(), decoPValues));
}

//===----------------------------------------------------------------------===//
// Function Decl implementation
//===----------------------------------------------------------------------===//

static constexpr const StringLiteral kMainSymbolName = "main";

/// Try to set the linkage name on a FnOp. If a different linkage name is
/// already set, emit an error and return true.
static bool trySetLinkageName(SMLoc loc, ASTDecl &decl,
                              LinkageNameAttr linkageName) {
  auto fnOp = llvm::dyn_cast_if_present<FnOp>(decl.getIfOperation());
  if (!fnOp)
    return false;

  if (auto existing = fnOp.getLinkageNameAttr()) {
    if (existing != linkageName) {
      decl.getShared().emitError(loc)
          << "function has conflicting linkage name from a previous @__name or "
             "@export decorator";
      return true;
    }
    return false; // same value; no conflict
  }
  fnOp.setLinkageNameAttr(linkageName);
  return false;
}

namespace {
struct FnSigDecorators : public SharedStateUser {
  FnSigDecorators(ASTDecl &decl, ASTDecl &sigDecl, SharedState &shared,
                  StringRef baseName, TypeCheckedFnSignature &tcSignature)
      : SharedStateUser(shared), decl(decl), sigDecl(sigDecl),
        funcOp(cast_or_null<FnOp>(decl.getIfOperation())), baseName(baseName),
        tcSignature(tcSignature) {}

  /// Apply a function signature decorator.
  LogicalResult applyOne(ExprNode *decorator);
  /// Finalize application of all signature decorators.
  void finalize();

  static LogicalResult checkAlwaysInlineBuiltin(FnOp funcBody,
                                                SharedState &shared);

private:
  void applyImplicitDecorator(SMLoc decoratorLoc, const CallNode *callNode);
  void applyCopyOrMoveCapture(SMLoc decoratorLoc, const CallNode *callNode,
                              bool isMove, StringRef decorator);
  void applyExtern(SMLoc decoratorLoc, const CallNode *node);
  void applyExportLike(SMLoc loc, bool isExport, const CallNode *node,
                       IREmitter &emitter);
  void applyAlwaysInline(const CallNode *node);
  void applyLLVMMetadata(SMLoc decoratorLoc, const CallNode *node);

  void applyArgumentless(StringRef spelling, const CallNode *callNode,
                         function_ref<void()> applyImpl);

  ArrayAttr getLLVMMetadataArray(ArrayRef<Operand> operands);

  /// Register an LLVM arg metadata in the internal list to avoid churning mlir
  /// attributes as these arg metadata decorators are parsed. Must call finalize
  /// to actually apply metadata onto the function.
  void applyLLVMArgMetadata(SMLoc decoratorLoc, const CallNode *node);

  ASTDecl &decl;
  ASTDecl &sigDecl;
  FnOp funcOp;
  StringRef baseName;
  TypeCheckedFnSignature &tcSignature;

  /// The working list of LLVMArgMetadata. Either empty, or initialized to a
  /// list with the same length as the total number of function arguments on
  /// first use.
  SmallVector<Attribute> llvmArgMetadata;

  /// The working vector of the LLVMMetadata.
  SmallVector<Attribute> llvmMetadata;
};
} // namespace

/// This function verifies @always_inline("builtin") functions after the body of
/// the function has been parsed.
LogicalResult FnSigDecorators::checkAlwaysInlineBuiltin(FnOp fnOp,
                                                        SharedState &shared) {
  // To see if this is foldable, synthesize a bunch of argument values that we
  // can cram into the function and see if it balks.
  SmallVector<TypedAttr> operands;

  // Figure out the callee.  We synthesize a bound reference to the callee
  // making up nonsense parameter bindings.
  ParameterEvaluator evaluator = shared.getParameterEvaluator();
  SmallVector<TypedAttr> params;
  for (auto paramDecl : fnOp.collectAllParams(/*implOrigins*/ false)) {
    params.push_back(
        UnknownAttr::get(evaluator.getReboundType(paramDecl.getType())));
    evaluator.setDeclBinding(paramDecl, params.back());
  }
  auto paramValueArray = ParameterExprArrayAttr::get(fnOp.getContext(), params);
  operands.push_back(
      fnOp.getBoundReference(shared.getEvaluationContext(), paramValueArray));

  for (auto arg : fnOp.getBody()->getArguments())
    operands.push_back(
        UnknownAttr::get(evaluator.getReboundType(arg.getType())));

  if (shared.foldInlineBuiltinFunction(operands, fnOp.getLoc(), true))
    return success();
  return failure();
}

LogicalResult FnSigDecorators::applyOne(ExprNode *decorator) {
  const DeclRefNode *declRef = dyn_cast<DeclRefNode>(decorator);
  const CallNode *callNode = nullptr;
  if (!declRef) {
    callNode = dyn_cast<CallNode>(decorator);
    if (callNode)
      declRef = dyn_cast<DeclRefNode>(callNode->callee);

    if (!callNode || !declRef) {
      // Qualified decorator names like
      // @extensibility.register_shape_function(args) have an AttributeRefNode
      // callee. Fall through to applyBodyDecorators (same path as struct
      // decorators) where validateCompilerDecorator handles the whitelist
      // check.
      if (callNode && isa<AttributeRefNode>(callNode->callee))
        return failure();
      emitError(decorator->getLoc(), "invalid expression in decorator");
      decl.setErroneous();
      return failure();
    }
  }

  StringRef spelling = declRef->spelling;
  if (spelling == "export") {
    for (ParamDeclAttr paramDecl : tcSignature.paramList.paramDeclAttrs) {
      // Singleton values like origins are fine. They will be removed by
      // lowerlit before code generation.
      if (ASTType(paramDecl.getType()).isSingleton(shared))
        continue;

      emitError(decorator->getLoc(),
                "@export can not be applied on parametric functions");
      decl.setErroneous();
      return failure();
    }
    IREmitter emitter(sigDecl, EC_Decorator);
    applyExportLike(decorator->getLoc(), /*isExport=*/true, callNode, emitter);
  } else if (spelling == "__name") {
    IREmitter emitter(sigDecl, EC_Decorator);
    applyExportLike(decorator->getLoc(), /*isExport=*/false, callNode, emitter);
  } else if (spelling == "staticmethod") {
    applyArgumentless(spelling, callNode, [&]() {
      if (!decl.tryGetMethodParentDecl()) {
        emitError(declRef->getLoc(),
                  "only methods on structs may be declared static");
      }
    });
    // We set the staticmethod flag even on errors, since the user intention is
    // clear, and this will suppress errors about missing self arguments.
    funcOp.setIsStatic(true);
  } else if (spelling == "always_inline") {
    applyAlwaysInline(callNode);
  } else if (spelling == "no_inline") {
    applyArgumentless(spelling, callNode,
                      [&]() { funcOp.setInlineLevel(InlineLevel::Never); });
  } else if (spelling == "__parameter" || spelling == "parameter") {
    // Temporarily accept the legacy `@parameter` spelling with a deprecation
    // warning so the rename to `@__parameter` can land without breaking
    // out-of-tree and late-landing code in the same change.
    if (spelling == "parameter") {
      emitWarning(declRef->getLoc(),
                  "'@parameter' is deprecated; use '@__parameter'")
          << FixIt::replaceToken(declRef->getLoc(), "__parameter");
    }
    applyArgumentless(spelling, callNode,
                      [&]() { tcSignature.argList.effects.setCapturing(); });
  } else if (spelling == "__unsafe_nested_origins_read_only") {
    applyArgumentless(spelling, callNode,
                      [&]() { tcSignature.isNestedOriginsReadOnly = true; });
  } else if (spelling == "__allow_legacy_custom_self_type") {
    applyArgumentless(spelling, callNode,
                      [&]() { tcSignature.allowCustomSelfType = true; });
  } else if (spelling == "implicit") {
    applyImplicitDecorator(decorator->getLoc(), callNode);
  } else if (spelling == "extern") {
    applyExtern(decorator->getLoc(), callNode);
  } else if (spelling == "__move_capture") {
    applyCopyOrMoveCapture(decorator->getLoc(), callNode, /*isMove=*/true,
                           spelling);
  } else if (spelling == "__copy_capture") {
    applyCopyOrMoveCapture(decorator->getLoc(), callNode, /*isMove=*/false,
                           spelling);
  } else if (spelling == "__llvm_metadata") {
    applyLLVMMetadata(decorator->getLoc(), callNode);
  } else if (spelling == "__llvm_arg_metadata") {
    applyLLVMArgMetadata(decorator->getLoc(), callNode);
  } else {
    return failure();
  }

  return success();
}

void FnSigDecorators::applyImplicitDecorator(SMLoc decoratorLoc,
                                             const CallNode *callNode) {
  size_t numOperands = callNode ? callNode->operands.size() : 0;
  if (numOperands > 1) {
    emitError(callNode->getLoc())
        << "'@implicit' decorator takes 0 or 1 arguments, found "
        << numOperands;
    return;
  }

  ImplicitConversionKind conversionKind = ImplicitConversionKind::Implicit;
  if (numOperands == 1) {
    const Operand &operand = callNode->operands[0];

    auto *boolExpr = dyn_cast<BoolLiteralNode>(operand.expr);
    if (!boolExpr || !operand.isKeyword() || operand.name != "deprecated") {
      emitError(callNode->getLoc())
          << "'@implicit' may only have a keyword argument 'deprecated' with "
             "literal boolean value";
      return;
    }
    if (boolExpr->value)
      conversionKind = ImplicitConversionKind::Deprecated;
  }

  if (tcSignature.fnInfo.kind != SpecialFunctionKind::kInit) {
    emitError(decoratorLoc)
        << "'@implicit' may only be applied to '__init__' methods";
    return;
  }

  ArrayRef<ParsedArgument> args = tcSignature.argList.parsedArgs;

  // Drop any error and result slots, default arguments and variadics.
  // Things like `__init__(out x, x: T, y : T = 42).
  // Allow `__init__(out x, x: Int = 4)` which has a default.
  while (1) {
    if (args.empty())
      break;

    auto &lastArg = args.back();
    if (lastArg.convention == ParsedArgument::kConventionByRefResult) {
      args = args.drop_back();
      continue;
    }

    // Drop defaults and varargs so long as they aren't the last argument.
    if (args.size() > 1 && (lastArg.initExpr || // arg has a default.
                                                // vararg lists can be empty
                            lastArg.variadicKind != VariadicKind::None))
      args = args.drop_back();
    else
      break;
  }

  // We must have a positional argument to take the new value.
  if (args.size() != 1 ||
      (args[0].kwArgHandling != KWArgHandling::kPositionalOnly &&
       args[0].kwArgHandling != KWArgHandling::kPositionalOrKeyword)) {
    emitError(decl.getLoc()) << "'@implicit' initializers must accept a single "
                                "positional argument value";
    return;
  }
  funcOp.setImplicitConversion(conversionKind);
}

void FnSigDecorators::applyCopyOrMoveCapture(SMLoc decoratorLoc,
                                             const CallNode *callNode,
                                             bool isMove,
                                             StringRef decoratorSpelling) {
  if (!callNode || callNode->operands.empty()) {
    emitError(decoratorLoc, "'@")
        << decoratorSpelling << "' must have arguments";
    return;
  }

  const CallNode &node = *callNode;
  // HACK(#16110): Need to implement proper capture list syntax rather than rely
  // on a special decorator.
  for (const Operand &operand : node.operands) {
    auto *declRef = dyn_cast<DeclRefNode>(operand.expr);
    if (!declRef) {
      emitError(operand.getLoc(), "'@")
          << decoratorSpelling << "' expected a declaration";
      continue;
    }
    LookupResult lookup = shared.lookupAndResolveDecl(
        declRef->spelling, declRef->getLoc(), *decl.getParentDecl(),
        /*searchParentScopes=*/true);
    if (lookup.isErroneous())
      continue;

    ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
    if (decls.empty()) {
      emitError(declRef->getLoc(), "cannot capture unknown value '")
          << declRef->spelling << "'";
      continue;
    }
    if (decls.size() != 1) {
      emitError(declRef->getLoc(), "cannot capture overloaded value '")
          << declRef->spelling << "'";
      continue;
    }

    // Emit an immutable copy of the captured declaration.
    FnOp parentOp = funcOp->getParentOfType<FnOp>();
    if (!parentOp) {
      emitError(declRef->getLoc(), "'@")
          << decoratorSpelling
          << "' decorator only applies to nested functions";
      return;
    }

    IREmitter emitter(*decl.getParentDecl(), OpBuilder(funcOp));
    RValue captureRVal;
    if (!isMove) {
      // For a copy capture, just emit the value reference as an RValue, which
      // will make sure to copy it.
      captureRVal = emitter.emitExprRValue(declRef, EC_Capture);
      if (!captureRVal)
        return;
      // HACK: This only has the intended effect of "immortalizing" a
      // register-passable value by creating an SRValue.
      if (!captureRVal.getType().isTrivial(node.getLoc(), shared)) {
        emitError(node.getLoc(), "TODO: @__copy_capture only works as intended "
                                 "with trivial register-passable types");
      }
    } else {
      // For a move capture, we emit this with an implicit transfer.
      // HACK(#16110): This transfers ownership without an explicit `^` from
      // the user, because we don't have capture list syntax.
      UnaryOpNode transfer(ExprNode::kTransfer, declRef->getLoc(), declRef);
      captureRVal = emitter.emitExprRValue(&transfer, EC_Capture);
    }

    // We can only capture dynamic values so materialize param expressions.
    if (auto pval = captureRVal.getIfPValue()) {
      if (pval.getType().isRegisterPassable(decl.getLoc(), shared))
        captureRVal = emitter.emitSRValue({captureRVal, declRef}, EC_Capture);
      else
        captureRVal = emitter.emitMRValue({captureRVal, declRef}, EC_Capture);
    }
    if (!captureRVal)
      return;

    // How is this transferring the RValue into the closure?
    DeclIRValue resultVal;
    if (auto srVal = captureRVal.getIfSRValue())
      resultVal = srVal;
    else {
      assert(captureRVal.getIfMRValue() && "Unknown RValue kind");
      resultVal = captureRVal.getIfMRValue();
    }

    // Bind the name in the scope so further references don't look like
    // reference captures.
    // FIXME: It would be cleaner to have an explicit representation of this,
    // e.g. an op that produces a ref to the value in the capture list.  Instead
    // we are still forming references outside the closure for things that are
    // copied and moved into the closure.
    emitter.getDeclResolver().addFullyResolvedDecl(resultVal, declRef->spelling,
                                                   sigDecl.getLoc(), &sigDecl);

    // Both move and copy captures are the same here - a move capture just does
    // the transfer above to generate its RValue.
    shared.addCaptureToScope(decl, decls.front(),
                             Capture(captureRVal,
                                     CaptureConvention::kConventionCopy,
                                     declRef->spelling));
  }
}

void FnSigDecorators::applyExtern(SMLoc decoratorLoc,
                                  const CallNode *callNode) {
  size_t numOperands = callNode ? callNode->operands.size() : 0;
  if (numOperands != 1) {
    emitError(decoratorLoc, "'@extern' requires 1 argument");
    return;
  }

  Operand operand = callNode->operands[0];
  auto strNode = dyn_cast<StringLiteralNode>(operand.expr);
  if (!strNode || !operand.isPositional()) {
    emitError(operand.getLoc(), "'@extern' requires a string literal argument");
    return;
  }

  if (!funcOp.getInputParams().empty()) {
    // TODO: Can this even happen?
    emitError(callNode->getLoc(),
              "'@extern' cannot be applied to a function with parameters");
    return;
  }

  std::string libName = strNode->getValue();
  auto ctx = funcOp->getContext();
  funcOp.setLinkageNameAttr(LinkageNameAttr::get(ctx, libName));

  if (decl.getParentDecl() && llvm::isa_and_nonnull<TraitDeclOp, StructDeclOp>(
                                  decl.getParentDecl()->getIfOperation())) {
    emitError(callNode->getLoc(), "'@extern' cannot be applied to a method");
    return;
  }

  if (!tcSignature.argList.hasExplicitABI) {
    emitError(decoratorLoc,
              "'@extern' requires an explicit 'abi()' effect on the function");
  }

  funcOp.setExternal(true);
}

/// Apply `@export("linkageName")` or `@__name("linkageName")` to a declaration,
/// optionally mark it external, and register it with the shared state to ensure
/// no duplicate linkage names.
void FnSigDecorators::applyExportLike(SMLoc loc, bool isExport,
                                      const CallNode *node,
                                      IREmitter &emitter) {
  ArrayRef<Operand> operands;
  if (node)
    operands = node->operands;
  StringRef spelling = isExport ? "@export" : "@__name";
  if (!isExport && operands.empty()) {
    emitError(loc, spelling) << " must have at least 1 argument";
    return;
  }
  if (operands.size() > 2) {
    emitError(loc, spelling) << " requires at most 2 arguments";
    return;
  }
  SMLoc nodeLoc = node ? node->getLoc() : loc;

  // TODO: Consider extracting the operand-parsing loop below into a helper
  // (e.g. parseDecoratorArgs) that returns a struct {TypedAttr rawName,
  // std::optional<std::string> exportABI, std::optional<bool> mangle}, to
  // separate argument parsing from the semantic actions that follow.
  TypedAttr linkageName;
  std::optional<std::string> exportABI;
  for (const Operand &operand : operands) {
    auto strNode = dyn_cast<StringLiteralNode>(operand.expr);
    if (strNode && operand.isKeyword() && operand.name == "ABI") {
      exportABI = strNode->getValue();
      if (*exportABI != "C") {
        emitError(operand.getLoc(),
                  "only \"C\" ABI is supported at the moment");
        return;
      }
      shared.emitWarning(operand.getLoc())
          << "ABI=\"C\" is deprecated; use abi(\"C\") instead";
    } else if (strNode && operand.isPositional()) {
      if (linkageName) {
        emitError(nodeLoc, spelling) << " must have at most 1 name argument";
        return;
      }
      linkageName = StringAttr::get(strNode->getValue(),
                                    KGEN::StringType::get(decl.getContext()));
    } else if (operand.isPositional() && !isExport) { // Not @export for now
      auto paramEmitter = emitter.getParamEmitter(EC_Decorator);
      CValue linkageNameVal =
          paramEmitter.emitExprCValue(operand.expr, EC_Decorator);
      if (!linkageNameVal) {
        emitError(nodeLoc, spelling)
            << " requires a string specifying the linkage name of the symbol";
        return;
      }

      linkageName = paramEmitter.emitStringExprAsDataToStr(
          linkageNameVal, operand.expr, nodeLoc, EC_Decorator);
      if (!linkageName) {
        emitError(nodeLoc) << "failure to create linkage name";
        return;
      }
    } else {
      emitError(nodeLoc, spelling)
          << " requires a string specifying the name of the exported symbol";
      return;
    }
  }

  LinkageNameAttr wrappedName;
  if (linkageName)
    wrappedName = KGEN::LinkageNameAttr::get(linkageName, /*mangle=*/!isExport);

  // Handle the unique case of main. We implicitly export main, so this is
  // simply checking that the user didn't try to export it as something else.
  std::optional<StringRef> simpleLinkageName = baseName;
  if (wrappedName)
    if (auto strAttr = dyn_cast<StringAttr>(wrappedName.getName()))
      simpleLinkageName = strAttr.getValue();

  if (simpleLinkageName == kMainSymbolName) {
    if (baseName != kMainSymbolName)
      emitError(loc, "only 'main' can be exported as 'main'");
    if (!isa<FnOp>(decl.getIfOperation()))
      emitError(loc, "exported 'main' must be a function");
    return;
  }
  if (baseName == kMainSymbolName) {
    emitError(loc, "'main' can only be exported as 'main'");
    return;
  }

  if (wrappedName) {
    if (trySetLinkageName(loc, decl, wrappedName))
      return;
  }
  if (!isExport)
    return;

  funcOp.setExported();

  if (exportABI.has_value()) {
    tcSignature.argList.effects.setCABI(true);

    // Validate the linkage name is a valid C identifier. We don't permit
    // non-literal identifiers for these functions.
    if (wrappedName && !simpleLinkageName) {
      emitError(loc) << " \"C\" ABI functions must have literal identifiers";
      return;
    }

    if (!isCIdentifier(*simpleLinkageName)) {
      emitError(loc, *simpleLinkageName) << " is not a valid C identifier";
      return;
    }

    if (tcSignature.argList.effects.isThrows()) {
      emitError(loc) << "'abi(\"C\")' function may not be marked 'raises'; "
                        "remove 'raises' or use 'abi(\"Mojo\")'";
      return;
    }
  } else if (!tcSignature.argList.hasExplicitABI) {
    emitWarning(loc, spelling)
        << " requires an explicit 'abi()' effect on the function";
  }

  // FIXME: This is an incomplete check as doesn't handle complex
  // non-literal expressions. Should we just defer it all until later?
  if (simpleLinkageName)
    getDeclResolver().registerAndCheckExport(*simpleLinkageName, loc);
}

void FnSigDecorators::applyAlwaysInline(const CallNode *callNode) {
  size_t numOperands = callNode ? callNode->operands.size() : 0;
  if (numOperands == 0) {
    // `@always_inline` and `@always_inline()` are both allowed.
    funcOp.setInlineLevel(InlineLevel::Always);
    return;
  }

  if (numOperands > 1) {
    emitError(callNode->getLoc())
        << "'@always_inline' decorator takes 0 or 1 arguments, found "
        << numOperands;
    return;
  }

  const Operand &operand = callNode->operands[0];
  if (operand.isPositionalStringLiteral("nodebug")) {
    funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);
  } else if (operand.isPositionalStringLiteral("builtin")) {
    funcOp.setInlineLevel(InlineLevel::AlwaysBuiltin);
  } else {
    emitError(callNode->getLoc())
        << "'@always_inline' operand must be \"nodebug\" or \"builtin\"";
  }
}

void FnSigDecorators::applyArgumentless(StringRef spelling,
                                        const CallNode *callNode,
                                        function_ref<void()> applyImpl) {
  if (!callNode)
    return applyImpl();
  emitError(callNode->getLoc()) << "'@" << spelling << "' cannot have arguments"
                                << FixIt::remove(callNode->getRange());
}

/// Return AliasDeclOp that corresponds to the value's name by looking at all
/// aliases within parent scopes up to FileModule. Return nullopt if not found.
/// Emit error if cannot resolve import op or declaration with the value.name is
/// not an alias.
static std::optional<AliasDeclOp> getLLVMMetadataNameAlias(SharedState &shared,
                                                           ASTDecl &funcDecl,
                                                           StringAttr name) {
  ASTDecl *parent = &funcDecl;
  // Analyze all parent scopes of the function in order to find closed
  // declaration with the value.name. Fully resolve that declaration if needed.
  do {
    parent = parent->getParentDecl();
    if (!parent)
      return {};

    ArrayRef<ASTDecl *> nameDecls = parent->lookupInCurrentScope(name);
    // Not interesting scope. Keep looking up for the declaration with
    // value.name.
    if (nameDecls.empty())
      continue;

    if (isa_and_nonnull<UnresolvedImportOp>(
            nameDecls.back()->getIfOperation())) {
      if (failed(shared.getDeclResolver().resolveBody(*nameDecls.back(),
                                                      funcDecl.getLoc()))) {
        shared.emitError(funcDecl.getLoc(), "cannot resolve comptime value '")
            << name << "' used in '@__llvm_metadata'";
        return {};
      }
    }
    if (auto aliasOp =
            dyn_cast_or_null<AliasDeclOp>(nameDecls.back()->getIfOperation()))
      return aliasOp;

    shared.emitError(funcDecl.getLoc(), "name '")
        << name << "' cannot be used in '@__llvm_metadata'";
    return {};
  } while (!isa_and_nonnull<FileModuleOp>(parent->getIfOperation()));
  return {};
}

ArrayAttr FnSigDecorators::getLLVMMetadataArray(ArrayRef<Operand> operands) {
  IREmitter emitter(sigDecl, EC_Decorator);
  SmallVector<Attribute> metadata;
  for (Operand value : operands) {
    StringAttr metadataName;
    ExprNode *metadataValue;
    // Handle the case of only a metadata name, with no value associated.
    if (value.unpackStyle == ArgUnpackStyle::kPositional) {
      auto declRef = dyn_cast<DeclRefNode>(value.expr);
      if (!declRef) {
        emitError(value.getLoc(), "Expected LLVM metadata name");
        continue;
      }
      metadataName = StringAttr::get(getContext(), declRef->spelling);
      metadataValue = nullptr;
    } else {
      if (!value.name) {
        emitError(value.getLoc(), "LLVM metadata requires a name");
        continue;
      }
      metadataName = value.name;
      metadataValue = value.expr;
    }

    // It might be possible that name comes from alias, therefore need to
    // analyze all module's aliases to see if alias's value needs to be used.
    if (std::optional<AliasDeclOp> aliasOp =
            getLLVMMetadataNameAlias(shared, sigDecl, metadataName))
      metadata.push_back(*aliasOp->getValue());
    else
      metadata.push_back(metadataName);

    if (metadataValue) {
      if (PValue attr = emitter.emitExprPValue(value.expr, EC_Decorator))
        metadata.push_back(attr);
    } else {
      // Store unit attr as value.
      metadata.push_back(UnitAttr::get(getContext()));
    }
  }
  return ArrayAttr::get(getContext(), metadata);
}

void FnSigDecorators::applyLLVMMetadata(SMLoc decoratorLoc,
                                        const CallNode *node) {
  size_t numOperands = node ? node->operands.size() : 0;
  if (numOperands == 0) {
    emitError(decoratorLoc, "'@__llvm_metadata' requires operands");
    return;
  }

  ArrayAttr metadata = getLLVMMetadataArray(node->operands);
  llvmMetadata.append(metadata.begin(), metadata.end());
}

void FnSigDecorators::applyLLVMArgMetadata(SMLoc decoratorLoc,
                                           const CallNode *node) {
  size_t numOperands = node ? node->operands.size() : 0;
  if (numOperands == 0) {
    emitError(decoratorLoc, "'@__llvm_arg_metadata' requires operands");
    return;
  }

  Operand targetArg = node->operands[0];
  auto declRef = dyn_cast<DeclRefNode>(targetArg.expr);
  // We expect the first operand to be "positional", i.e. it should just be a
  // standalone name.
  if (targetArg.unpackStyle != ArgUnpackStyle::kPositional || !declRef) {
    emitError(
        targetArg.getLoc(),
        "First argument of '@__llvm_arg_metadata' must be an argument name");
    return;
  }

  // Ignore empty metadata list.
  if (numOperands == 1)
    return;

  // Find argument number corresponding to this arg name.
  int64_t argIdx = -1;
  for (auto [index, arg] : llvm::enumerate(tcSignature.argList.parsedArgs)) {
    if (arg.name.getValue() == declRef->spelling) {
      argIdx = index;
      break;
    }
  }

  if (argIdx < 0) {
    emitError(
        targetArg.getLoc(),
        "Function decorated by '@__llvm_arg_metadata' has no argument named '")
        << declRef->spelling << "'";
    return;
  }

  // First time setting arg metadata, initialize with array of empty attributes.
  if (llvmArgMetadata.empty())
    llvmArgMetadata.insert(llvmArgMetadata.begin(),
                           tcSignature.argList.parsedArgs.size(),
                           ArrayAttr::get(getContext(), {}));

  llvmArgMetadata[argIdx] = getLLVMMetadataArray(node->operands.drop_front());
}

void FnSigDecorators::finalize() {
  if (funcOp.isExternal()) {
    if (funcOp.getInlineLevel() != InlineLevel::Never &&
        funcOp.getInlineLevel() != InlineLevel::Automatic) {
      emitError(funcOp.getLoc(), "extern functions cannot be inlined");
      return;
    }
    funcOp.setInlineLevel(InlineLevel::Never);
  }

  // If we've an exported function with no explicit linkage name, set it now.
  if (funcOp.isExported() && !funcOp.getLinkageNameAttr()) {
    trySetLinkageName(decl.getLoc(), decl,
                      LinkageNameAttr::get(decl.getContext(), baseName));
  }

  if (!llvmArgMetadata.empty())
    funcOp.setLLVMArgMetadataArrayAttr(
        ArrayAttr::get(getContext(), llvmArgMetadata));

  if (!llvmMetadata.empty()) {
    // NOTE: @llvm_metadata are processed and added in reverse order
    funcOp.setLLVMMetadataArrayAttr(ArrayAttr::get(getContext(), llvmMetadata));
  }
}

/// Given the lexical context of a function, return true if the default bit
/// for the function is capturing.
static bool
isCapturingByDefault(SharedState &shared, FnOp funcOp, TraitType canonicalTrait,
                     std::optional<ArrayRef<ParamDeclAttr>> parentDecls,
                     ArrayRef<ParamDeclAttr> paramDecls) {
  // Any function that contains a capturing closure as a parameter is itself
  // capturing, include parent struct parameters.
  mlir::AttrTypeWalker walker;
  walker.addWalk([](FuncType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // Temporary solution to supporting capturing parametric closures inside
  // nested closures: propagate the capturing effect through the closure type.
  walker.addWalk([&](SymbolRefAttr symbol) {
    auto traitDecl = shared.declResolver->getDeclForTypeSymbolIfExists(symbol);
    if (!traitDecl)
      return WalkResult::advance();
    TraitDeclOp traitDeclOp =
        dyn_cast_if_present<TraitDeclOp>(traitDecl->getIfOperation());
    if (traitDeclOp && traitDeclOp.getDefinesClosure())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  bool isInterrupted = false;
  if (canonicalTrait)
    isInterrupted = walker.walk(canonicalTrait).wasInterrupted();
  return isInterrupted ||
         llvm::any_of(llvm::concat<const ParamDeclAttr>(
                          paramDecls, parentDecls
                                          ? parentDecls.value()
                                          : SmallVector<ParamDeclAttr>()),
                      [&](ParamDeclAttr decl) {
                        return walker.walk(decl).wasInterrupted();
                      });
}

std::pair<SmallVector<ParamDeclRefAttr>, FnTypeGeneratorType>
DeclResolver::createSelfContainedSignature(FnTypeGeneratorType original) {
  // Collect the subset of referenced parameters. Use a set vector to keep the
  // order deterministic.
  llvm::SmallSetVector<ParamDeclRefAttr, 4> capturedRefs;
  getCanonicalType(original).walk(
      [&](ParamDeclRefAttr ref) { capturedRefs.insert(ref); });

  SmallVector<ParamDeclRefAttr> captured = capturedRefs.takeVector();
  // Unbind the N capture parameters, creating a FuncType with N new input
  // parameters prepended.
  // TODO: what if we capture a variadic?
  auto unbound = FnTypeGeneratorType::prependParams(
      original, llvm::map_to_vector(captured, [](ParamDeclRefAttr ref) {
        return ParamDeclAttr::get(ref);
      }));
  return {std::move(captured), unbound};
}

static bool allCopyable(ArrayRef<Capture> captures, SharedState &shared,
                        SMLoc loc, ASTDecl &scope) {
  for (const Capture &capture : captures) {
    switch (capture.getCaptureConvention()) {
    case CaptureConvention::kConventionCopy:
    case CaptureConvention::kConventionTrivialCopy:
    case CaptureConvention::kConventionRef:
    case CaptureConvention::kConventionRead:
    case CaptureConvention::kConventionMut:
      continue;
    default:
      if (!capture.getValue().getRValueType().isCopyable(loc, shared, true,
                                                         scope))
        return false;
    }
  }
  return true;
}

static MLValue emitClosureInstance(ArrayRef<Capture> captures,
                                   ASTDecl &nestedFnDecl, SharedState &shared,
                                   ArrayRef<ParamDeclRefAttr> bodyParamCaptures,
                                   bool useParametricTrait) {
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  SMLoc loc = nestedFnDecl.getLoc();
  Location mlirLoc = shared.translateLocation(loc);
  if (shared.diBuilder)
    mlirLoc = shared.diBuilder->createScopedLoc(mlirLoc);
  FnTypeGeneratorType closureSig = nestedFn.getFuncTypeGenerator();
  assert(nestedFnDecl.getParentDecl() &&
         "closure instance must have a parent function");

  ASTDecl *moduleDecl = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();
  auto [capturedRefs, _] =
      DeclResolver::createSelfContainedSignature(closureSig);

  // Register captured external parameter references so that call sites can
  // pre-seed auxiliary parameters during overload fitness evaluation.
  if (!capturedRefs.empty()) {
    ASTDecl *enclosingDecl =
        nestedFnDecl.getParentDecl()->getNearestDeclOfType<FnOp>();
    if (enclosingDecl) {
      SmallVector<ClosureParamCapture> paramCaptures;
      for (ParamDeclRefAttr ref : capturedRefs)
        paramCaptures.push_back({ref.getName(), ref.getType()});
      shared.addClosureParamCaptures(*enclosingDecl,
                                     nestedFn.getSourceNameAttr(),
                                     std::move(paramCaptures));
    }
  }

  ASTDecl *closureTrait =
      useParametricTrait
          ? shared.getUniversalParametricClosureTrait()
          : shared.getOrCreateClosureTrait(loc, *moduleDecl, closureSig);
  bool isCopyable = allCopyable(captures, shared, loc, nestedFnDecl);

  ClosureEmitter &emitter = shared.getClosureEmitter();
  // Merge in parameter captures found in the body (from runtime capture types,
  // body ops, and locations) that are not already in capturedRefs from the
  // signature walk.
  {
    SmallPtrSet<StringAttr, 8> seen;
    for (ParamDeclRefAttr ref : capturedRefs)
      seen.insert(ref.getName());
    for (ParamDeclRefAttr ref : bodyParamCaptures) {
      if (seen.insert(ref.getName()).second)
        capturedRefs.push_back(ref);
    }
  }
  Value closureInstance =
      emitter.emitClosure(*moduleDecl, nestedFnDecl, captures, *closureTrait,
                          mlirLoc, isCopyable, closureSig, capturedRefs);
  if (!closureInstance)
    return {};
  return MLValue(closureInstance);
}

namespace {
/// The values and parameter references a (nested) function captures from its
/// enclosing scopes, as computed from its resolved body.
struct BodyCaptures {
  SmallVector<Capture> values;
  SmallVector<ParamDeclRefAttr> paramRefs;
};
} // namespace

/// Collect what the resolved body of a (nested) function captures from
/// enclosing scopes: `values` are the captured runtime values in scope and
/// `paramRefs` are the parameter references used from above (deduplicated
/// across the sugared/canonical versions of the same type).
static BodyCaptures collectBodyCaptures(SharedState &shared, ASTDecl &decl,
                                        FnOp funcOp) {
  BodyCaptures result;

  // Find all parameter captures in the function body.
  // TODO: Use the SharedState cache?
  ParameterCollector::Analysis collectorCache;
  ParameterUseDefGraph graph(funcOp.getBodyRegion());
  graph.calculate(collectorCache);

  // Get captured parameters that cross with captured values.
  ParameterCollector collector(collectorCache);
  SmallVector<ParamDeclRefAttr> capturedUses;
  for (auto &[_, capture] : shared.getCaptureRangeInScope(decl)) {
    result.values.push_back(capture);
    bool unusedHasConstExpr = false;
    size_t unusedRequiredSignatureDepth = 0;
    collector.collectUsesFromType(capture.getValue().getType(), capturedUses,
                                  unusedHasConstExpr,
                                  unusedRequiredSignatureDepth);
  }
  for (ParamDeclRefAttr use : capturedUses)
    graph.usesFromAbove.insert(use);

  result.paramRefs = graph.usesFromAbove.takeVector();

  // Because our IR has type sugar, we can capture the same parameter from
  // the sugared and canonical version of the same type.  Remove one of the
  // versions from the captured uses.
  SmallPtrSet<StringAttr, 8> capturedParamNames;
  llvm::erase_if(result.paramRefs, [&](ParamDeclRefAttr use) {
    return !capturedParamNames.insert(use.getName()).second;
  });

  return result;
}

/// Construct the runtime value for a fully-resolved non-legacy nested-`def`
/// closure: promote a stateless closure to a top-level function, or otherwise
/// materialize a storage-struct instance. Sets the decl's IR value on success.
static LogicalResult
constructClosure(SharedState &shared, ASTDecl &decl, FnOp funcOp,
                 ArrayRef<Capture> captures,
                 ArrayRef<ParamDeclRefAttr> paramCaptures,
                 const ParsedCaptureList &captureSignature,
                 ArrayRef<ConstraintAttr> closureExternalRefConstraints,
                 FnTypeGeneratorType signature, bool useParametricTrait) {
  // abi("C") functions must be bare function pointers with no captured
  // state, even in closure form.
  if (signature.getFnEffects().isCABI() && !captures.empty()) {
    shared.emitError(funcOp.getLoc())
        << "a abi(\"C\") function cannot capture variables";
    return failure();
  }

  // Stateless nested defs (no runtime captures) are still promoted to thin
  // functions so they remain usable as parameter values (e.g.
  // `_reflection_write_to[f=call_write_to]`).
  if (closureExternalRefConstraints.empty() &&
      captureSignature.parsedCaptures.empty() &&
      !captureSignature.captureAllByConvention) {
    shared.closureEmitter->promoteClosure(decl, paramCaptures);
    return success();
  }

  MLValue instance = emitClosureInstance(captures, decl, shared, paramCaptures,
                                         useParametricTrait);
  if (!instance)
    return failure();
  decl.setIRValue(instance);
  return success();
}

/// Make a copy or cast mutability if needed in the parent scope so that the
/// semantics of the closure body are upheld. For example, consider the
/// following:
///
/// def toy(read byCopy:String, prefix:String):
///   def myclosure(prefix: String) {var byCopy} -> String:
///      byCopy += "v2" // LINE A
///      return prefix
///   takeIt(myclosure, prefix)
///
/// Note the mutation on line A. If we do not make a copy in the parser, the
/// parser will complain that you cannot bind a read only value to the mutable
/// reference argument of __iadd__ because it will map the byCopy value on line
/// A to the byCopy value passed into the function with argument convention read
/// . The parser is wrong about this because the user expressed that he wants to
/// make a mutable copy of the immutable reference and mutate the copy. We
/// fulfill the user's request but not until after checklifetimes, where we emit
/// the copy and replace usages of the original value with that copy. I will
/// introduce an op to avoid the extra copy (MOCO 2291). Until then, we just
/// emit an extra copy/move.
static LogicalResult createCaptureValues(ParserBase &p, ASTDecl &sigDecl,
                                         ParsedCaptureList &captureSignature,
                                         ASTDecl &decl) {
  FnOp funcOp = cast<FnOp>(decl.getIfOperation());
  IREmitter emitter(*decl.getParentDecl(), OpBuilder(funcOp));
  bool didFail = false;
  for (auto [name, capture, location] : captureSignature.parsedCaptures) {
    if (!ClosureEmitter::addCaptureValue(decl, location, name, capture, emitter,
                                         &sigDecl))
      didFail = true;
  }
  return didFail ? failure() : success();
}

/// Registers the closure-typed parameters in `params` whose signatures capture
/// other parameters in the same list (e.g. `F: def[w: Int]() -> SIMD[dtype, w]`
/// captures `dtype`) and returns the type-equality `where` clauses that bind
/// each closure alias to the captured parameter (e.g. `eq(F.dtype, dtype)`).
static SmallVector<ConstraintAttr>
registerClosureParamCaptures(ArrayRef<ParamDeclAttr> params, ASTDecl &decl,
                             SharedState &shared, OpBuilder &builder) {
  SmallVector<ClosureExternalRef> closureExternalRefs;
  for (ParamDeclAttr param : params)
    shared.getClosureEmitter().collectClosureExternalRefs(param,
                                                          closureExternalRefs);

  SmallVector<ConstraintAttr> constraints;
  if (closureExternalRefs.empty())
    return constraints;

  ClosureParamCaptures closureParamCaptures;
  for (const ClosureExternalRef &ref : closureExternalRefs)
    closureParamCaptures[ref.closureParam.getName()].push_back(
        {ref.externalName, ref.externalType});
  shared.setClosureParamCaptures(decl, std::move(closureParamCaptures));

  // Emit a type-equality constraint for each external reference so the closure
  // alias binds to the captured parameter.
  for (const ClosureExternalRef &ref : closureExternalRefs) {
    TypedAttr rhs = ParamDeclRefAttr::get(ref.externalName, ref.externalType);

    ParamDeclAttr closureParam = ref.closureParam;

    std::optional<TraitDeclOp> closureTraitOr = ClosureEmitter::getClosureDecl(
        shared, getCanonicalType(closureParam.getType()));
    assert(closureTraitOr && "expected closure type");

    // Get the trait symbol for the GetWitnessAttr.
    TraitDeclOp closureTrait = *closureTraitOr;
    auto traitSymbol =
        TraitSymbolAttr::get(getFullyResolvedSymbolRef(closureTrait));

    // LHS: C.T - GetWitnessAttr accessing the alias on the closure param.
    TypedAttr witnessAttr =
        GetWitnessAttr::get(ParamDeclRefAttr::get(closureParam), traitSymbol,
                            ref.externalName, ref.externalType);

    TypedAttr idConstraint = ParamIdenticalAttr::get(witnessAttr, rhs);
    Location loc = shared.diags.translateLocation(decl.getLoc());
    constraints.push_back(
        ConstraintAttr::get(idConstraint, loc, /*message=*/StringAttr()));
  }
  return constraints;
}

/// Finalizes a fully type-checked function/closure signature onto `funcOp`:
/// registers closure-parameter captures (with their `eq(C.T, T)` where
/// clauses), builds the generator signature, and writes the params /
/// function-type / generator / mangled-symbol attributes. Returns the
/// (implicit-origin-indexed) generator signature, or null on failure.
static FnTypeGeneratorType finalizeResolvedFnOp(
    SharedState &shared, FnOp funcOp, ASTDecl &decl,
    TypeCheckedFnSignature &tcSignature, TypeCheckedParamList &paramList,
    StringAttr baseName,
    SmallVectorImpl<ConstraintAttr> &closureExternalRefConstraints) {
  OpBuilder builder = decl.getDeclEndBuilder();
  NamedAttrList attrs = funcOp->getAttrDictionary();

  // Register closure-parameter captures and their `eq(C.T, T)` where clauses.
  closureExternalRefConstraints = registerClosureParamCaptures(
      paramList.paramDeclAttrs, decl, shared, builder);
  llvm::append_range(tcSignature.paramList.emittedBodyConstraints,
                     closureExternalRefConstraints);

  FnTypeGeneratorType signature = tcSignature.getFnTypeGeneratorType();
  if (!signature)
    return {};

  decl.insertKnownAssumptions(closureExternalRefConstraints);

  /// configure FnOp

  // The implicitOriginDecls don't affect the signature, but they do get
  // prepended onto the paramDecls list.
  ParamDeclArrayAttr paramsArrayAttr;
  if (tcSignature.implicitOriginDecls.empty()) {
    paramsArrayAttr =
        builder.getAttr<ParamDeclArrayAttr>(paramList.paramDeclAttrs);
  } else {
    SmallVector<ParamDeclAttr> mergedParams;
    llvm::append_range(mergedParams, paramList.paramDeclAttrs);
    llvm::append_range(mergedParams, tcSignature.implicitOriginDecls);
    paramsArrayAttr = builder.getAttr<ParamDeclArrayAttr>(mergedParams);
  }

  attrs.set(funcOp.getParamsAttrName(), paramsArrayAttr);
  attrs.set(funcOp.getFunctionTypeAttrName(),
            TypeAttr::get(tcSignature.getFunctionType()));

  // Now that the FunctionType is set to the pretty type that includes implicit
  // origins, we strip off the named origin decl references and replace them
  // with indices.
  signature = signature.replaceImplicitOriginsWithIndexes(
      tcSignature.implicitOriginDecls);
  attrs.set(funcOp.getFuncTypeGeneratorAttrName(), TypeAttr::get(signature));

  // Set the symbol to the mangled name and check for redefinition.
  attrs.set(funcOp.getSymNameAttrName(),
            shared.declResolver->getMangledName(baseName, *decl.getParentDecl(),
                                                signature));
  attrs.set(funcOp.getSourceNameAttrName(), baseName);

  // Set the result name binding if specified.
  if (StringAttr resultName = tcSignature.argList.resultArg.name)
    attrs.set(funcOp.getNamedResultAttrName(), resultName);

  // Remove the temporary "sym_namex" attribute set up in
  // StmtParser::parseDefFnStmt, see that method for an explanation.
  attrs.erase("sym_namex");

  // Bulk update the attributes.
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));
  return signature;
}

/// Resolve a parsed `lambda` expr into an anonymous closure. The capture list
/// and return type may both be elided. An omitted capture list captures any
/// free variables the body uses by `imm`, so it is thin (no captured state)
/// when the body captures nothing. An omitted return type defaults to `None`
/// (like for `def`), so the body must be `None`-typed.
///
/// lambda    ::=  "lambda" [param_signature] ["(" [argument_list] ")"]
///                [effects] [capture_list] ["->" expression] ":" expression
AnyValue DeclResolver::resolveAnonymousClosure(const LambdaNode *node,
                                               IREmitter &emitter,
                                               ExprDest &dest) {
  // Build the synthetic anonymous `def` the lambda desugars to.
  SMLoc loc = node->getLoc();
  ASTDecl &parentScope = emitter.declScope;

  // A null `emitter.builder` is exactly a parameter context.
  bool insideParamContext = !emitter.builder;
  OpBuilder paramBuilder(shared.getContext());
  if (insideParamContext) {
    // A compile-time lambda value cannot capture runtime vars.
    // (Implicit captures are dealt with after body analysis.)
    if (!node->captures.empty() || node->captureAllByConvention) {
      emitter.emitErrorForDynamicValueInParameter(
          node, "cannot use a capturing lambda");
      return {};
    }
    // Anchor the synthetic def at the nearest enclosing decl op (a `def`,
    // struct, alias, ...) -- the param-list scope isn't an op, so walk up.
    ASTDecl *anchor = parentScope.getNearestDeclOfType<ASTDeclInterface>();
    if (!anchor) {
      emitter.emitErrorForDynamicValueInParameter(
          node, "cannot use a lambda expression");
      return {};
    }
    paramBuilder.setInsertionPoint(anchor->getIfOperation());
  }
  OpBuilder &builder = insideParamContext ? paramBuilder : *emitter.builder;
  MLIRContext *ctx = builder.getContext();
  Location mlirLoc = shared.translateLocation(loc);

  // 1. Provisional FnOp with a placeholder `() -> Error` signature, so
  //    signature type-checking has an IR anchor.
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto functionType = FunctionType::get(ctx, ArrayRef<Type>(), {errorType});
  SmallVector<PogMetadataAttr> argPogs(
      functionType.getNumInputs(),
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  auto provisionalSig = GeneratorType::get(
      {},
      FuncType::get(
          functionType,
          FnMetaOriginDataAttr::get(ctx, 0, OriginSetAttr::get(ctx, {}), false,
                                    /*definesInteriorOrigins=*/false),
          PogListAttr::get(ctx, argPogs)),
      PogListAttr::get(ctx));
  StringAttr emptyStr = StringAttr::get(ctx, "");
  StringAttr baseName = StringAttr::get(
      ctx, ("`lambda_" + Twine(anonymousClosureCounter++)).str());
  FnOp funcOp =
      FnOp::create(builder, mlirLoc, emptyStr, emptyStr, provisionalSig);
  funcOp->removeAttr("sym_name");
  funcOp->setAttr("sym_namex", emptyStr);
  funcOp.setSourceNameAttr(baseName);
  {
    auto endBuilder = OpBuilder::atBlockEnd(funcOp.getBody());
    EndFnOp::create(endBuilder, funcOp.getLoc(), /*unresolved=*/true);
  }

  // 2. Decls: one for the closure, plus a signature scope. Neither is listed
  //    in the enclosing scope's name table, and the synthetic base name begins
  //    with a backtick. A lambda is anonymous: the unlisted decl means no name
  //    in user source resolves to it, and the leading backtick (which no user
  //    identifier can contain -- backtick-quoted identifiers strip their
  //    backticks) means user source cannot even spell the symbol it emits, so
  //    user code can neither reference the closure nor collide with it.
  ASTDecl &decl = addFullyResolvedDecl(funcOp.getOperation(), StringAttr(), loc,
                                       &parentScope);
  ASTDecl &sigDecl = addFullyResolvedDecl(funcOp.getOperation(), StringAttr(),
                                          loc, &parentScope);

  // 3. Set up captures before type-checking: emit copies/casts for named
  //    captures, and record the default-all convention so free variables in
  //    the body are captured rather than seen as unknown declarations.
  for (auto [name, capture, location] : node->captures) {
    IREmitter capEmitter(*decl.getParentDecl(), OpBuilder(funcOp));
    if (!ClosureEmitter::addCaptureValue(decl, location, name, capture,
                                         capEmitter, &sigDecl))
      return {};
  }
  if (node->captureAllByConvention)
    shared.setDefaultCaptureForScope(decl, *node->captureAllByConvention);
  else if (!node->hasExplicitCaptureList)
    // An omitted capture list defaults to capturing free variables by `imm`;
    // a body that captures nothing then yields a thin closure (empty storage).
    shared.setDefaultCaptureForScope(decl, CaptureConvention::kConventionRead);

  // 4. Reconstitute and type-check the signature. This also adds the param/arg
  //    decls and the FnOp's block arguments.
  ParsedParamList parsedParamList;
  parsedParamList.params = llvm::to_vector(node->parsedParams);
  std::optional<TypeCheckedParamList> paramListOrError =
      TypeCheckedParamList::create(parsedParamList, sigDecl);
  if (!paramListOrError)
    return {};
  TypeCheckedParamList &paramList = *paramListOrError;

  ParsedArgumentList argList;
  argList.parsedArgs = llvm::to_vector(node->parsedArgs);
  argList.resultArg = node->resultArg;
  argList.effects = node->effects;
  argList.thrownTypeExpr = const_cast<ExprNode *>(node->thrownTypeExpr);

  TypeCheckedFnSignature tcSignature(paramList, argList, /*originExpr=*/nullptr,
                                     &decl, baseName);

  // Move the param/arg decls from the signature scope into the closure decl so
  // the body resolves them as locals.
  decl.takeDecls(sigDecl);

  // 5. Build the generator signature and write the signature attributes onto
  //    the FnOp (shared with resolveSignature via finalizeResolvedFnOp), then
  //    register the symbol. The closure-param-capture constraints output is
  //    unused here: the lambda materializes its instance directly via
  //    emitClosureInstance.
  SmallVector<ConstraintAttr> closureExternalRefConstraints;
  FnTypeGeneratorType signature =
      finalizeResolvedFnOp(shared, funcOp, decl, tcSignature, paramList,
                           baseName, closureExternalRefConstraints);
  if (!signature)
    return {};
  for (ParsedArgument &arg : tcSignature.argList.parsedArgs)
    if (arg.isErroneous)
      return {};
  (void)finalizeFuncSignature(funcOp, decl);

  shared.setLocationDebugScope(funcOp);
  for (auto [parsedArg, bbArg] : llvm::zip(tcSignature.argList.parsedArgs,
                                           funcOp.getBody()->getArguments()))
    bbArg.setLoc(shared.diags.translateLocation(parsedArg.loc));

  // 6. Synthesize the `return EXPR` body. Push the closure's debug scope so the
  //    body's ops carry the closure subprogram (not the enclosing function's),
  //    then pop before the closure instance is materialized in the enclosing
  //    scope.
  Block &body = *funcOp.getBody();
  auto endFn = cast<EndFnOp>(body.front());
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder) {
      diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());
      endFn->setLoc(shared.translateLocation(loc));
    }
    endFn.setUnresolved(false);

    IREmitter bodyEmitter(decl, OpBuilder(&body.front()));
    ASTType userResultType = funcOp.getUserResultType();
    ExprDest resultDest(userResultType, EC_ReturnValue);
    if (signature.hasMemoryOnlyResult())
      resultDest =
          ExprDest(MLValue(funcOp.getArguments().back()), EC_ReturnValue);
    AnyValue resultValue = bodyEmitter.emitExpr(node->body, resultDest);
    if (!resultValue) {
      resultDest.resetForError(bodyEmitter);
      return {};
    }
    Value resultVal;
    if (!signature.hasMemoryOnlyResult()) {
      resultVal =
          bodyEmitter.emitSRValue({resultValue, node->body}, EC_ReturnValue,
                                  funcOp.getMLIRResultType());
      if (!resultVal)
        return {};
    }
    // Scope the return location to the closure subprogram (the scope guard is
    // active here), matching the body ops emitted above.
    Location returnLoc = shared.translateLocation(loc);
    if (shared.diBuilder)
      returnLoc = shared.diBuilder->createScopedLoc(returnLoc);
    bodyEmitter.emitNormalReturn(returnLoc, resultVal, /*emitEndFunc=*/false);
  }

  // 7. Collect captures over the resolved body and materialize the closure.
  BodyCaptures bodyCaptures = collectBodyCaptures(shared, decl, funcOp);

  // A captured runtime variable makes a parameter-context lambda dynamic, so
  // reject it.
  if (insideParamContext && !bodyCaptures.values.empty()) {
    // Same diagnostic as the explicit-capture reject above.
    emitter.emitErrorForDynamicValueInParameter(
        node, "cannot use a capturing lambda");
    return {};
  }

  // Fold a thin lambda to the promoted function's literal -- the value a named
  // `def` yields -- so it decays and rebinds like one. The singleton test is
  // the one a reference to a named `def` applies to itself, so the two agree.
  bool isThin = bodyCaptures.values.empty() && node->captures.empty() &&
                !node->captureAllByConvention;
  if (insideParamContext ||
      (isThin && llvm::all_of(signature.getInputParamTypes(), [&](Type type) {
         return ASTType(type).isSingleton(shared);
       }))) {
    resolveAllWithin(decl);
    shared.closureEmitter->promoteClosure(decl, bodyCaptures.paramRefs);
    // Two folds, keyed on whether enclosing parameters were baked in. With
    // parameters (comptime _values_, not runtime captures -- e.g.
    // `comptime add[N: Int] = lambda (x: Int) {} -> Int: x + N`),
    // `promoteClosure` leaves a bound func-literal generator as the decl's IR
    // value; fold to it.
    if (CValue boundLiteral = decl.getIfIRValue())
      return emitter.emitResult(AnyValue(boundLiteral), node, dest);
    // With no parameters baked in, no such value is left, so build the promoted
    // function's own func-literal here -- the value a `def` reference yields.
    PValue literal(
        funcOp.getFuncLiteralGenerator(shared.getEvaluationContext()));
    return emitter.emitResult(AnyValue(literal), node, dest);
  }

  MLValue instance =
      emitClosureInstance(bodyCaptures.values, decl, shared,
                          bodyCaptures.paramRefs, /*useParametricTrait=*/false);
  if (!instance)
    return {};

  // 8. Emit the closure value into the expression destination (handles
  //    conversion to the expected closure-trait type at the use site).
  return emitter.emitResult(AnyValue(instance), node, dest);
}

/// funcdef   ::=  [decorators] "def" identifier [param_signature]
///                "(" [argument_list] ")" ["->" expression] ":" suite
LogicalResult DeclResolver::resolveSignature(FnOp funcOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  assert(p.getToken().isAny(Token::kw_async, Token::kw_def, Token::kw_fn) &&
         "not a function definition?");
  bool isAsync = p.consumeIf(Token::kw_async);
  // FIXME(26.5): Remove support for 'fn'.
  if (p.getToken().is(Token::kw_fn)) {
    shared.emitError(p.getToken().getLoc(),
                     "'fn' has been removed; use 'def' instead")
        << FixIt::replaceToken(p.getToken().getLoc(), "def");
  }

  p.consumeToken();

  StringAttr baseName;
  SMLoc identifierLoc;
  if (p.parseIdentifier(baseName, "expected function name", &identifierLoc,
                        /*forbidStartOfLine=*/false,
                        /*allowKeyword=*/true))
    return failure();

  ASTDecl *parentDecl = decl.getParentDecl();
  // The function signature is a self-contained scope where the input and result
  // parameters of the function are visible by all types.  We must use a
  // temporary declaration here (with an empty name) because we don't want
  // references to the function itself to resolve to a fully-resolved decl, but
  // we need a fully-resolved decl for incremental lookups within the scope to
  // work out.
  ASTDecl &sigDecl = addFullyResolvedDecl(funcOp.getOperation(), StringAttr(),
                                          decl.getLoc(), parentDecl);

  // If this is a struct method, inherit parameter defined in the struct so that
  // we reject
  //
  // struct S[param: Int]:
  //     def method[param: Int](self): pass
  if (auto structOp =
          dyn_cast_or_null<StructDeclOp>(parentDecl->getIfOperation())) {
    for (auto pog : structOp.getSignature().getParamListAttrs().getPogs()) {
      StringRef paramName = pog.getName().getValue();
      ArrayRef<ASTDecl *> paramDecls =
          parentDecl->lookupInCurrentScope(paramName);
      // Must be autoparams, they aren't explicitly declared by the user so
      // can't be looked up by their names, nor should they lead to name
      // conflict.
      if (paramDecls.empty())
        continue;

      // If we found it, it must be a parameter declaration.
      assert(paramDecls.size() == 1 &&
             "expected exactly one parameter declaration");
      ASTDecl *paramDecl = paramDecls.front();
      addFullyResolvedDecl(paramDecl->irValue, paramName, paramDecl->getLoc(),
                           &sigDecl);
    }
  }

  // Parse declared parameters and add them to the current scope.
  ParsedParamList parsedParamList;

  // Add the parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (parsedParamList.parseParametersIfPresent(p, ArgListKind::kParamList))
    return failure();

  if (!parsedParamList.params.empty() && baseName == "__call__" &&
      dyn_cast_or_null<StructDeclOp>(parentDecl->getIfOperation())) {

    bool hasOnlyInferredParams =
        llvm::all_of(parsedParamList.params, [](const ParsedArgument &param) {
          // TODO(MOCO-2928): This should ignore whether the parameter is
          //  variadic or not, and check only whether it's inferred or not.
          //  Checking for variadics as well is a proxy for signatures like:
          //    def foo[*Ts: AnyType](...)
          //  which _can't_ make `Ts` infer-only due to MOCO-2928, an overbroad
          //  check that `//` cannot come after any `*` token.
          return param.kwArgHandling == KWArgHandling::kInferred ||
                 param.variadicKind != VariadicKind::None;
        });

    auto getItems = parentDecl->lookupInCurrentScope("__getitem__");
    auto setItems = parentDecl->lookupInCurrentScope("__setitem__");
    auto getAttrs = parentDecl->lookupInCurrentScope("__getattr__");
    if (!hasOnlyInferredParams &&
        (!getItems.empty() || !setItems.empty() || !getAttrs.empty())) {
      auto diag = p.emitWarning(funcOp->getLoc())
                  << "parametric '__call__' method cannot be called directly "
                     "because '"
                  << *parentDecl->getUserNameIfOperation()
                  << "' defines '__getitem__', '__setitem__', or "
                     "'__getattr__'; consider using a different name for this "
                     "method";

      for (const auto &decl : getItems)
        diag.attachNote(decl->getLoc()) << "__getitem__ defined here";

      for (const auto &decl : setItems)
        diag.attachNote(decl->getLoc()) << "__setitem__ defined here";

      for (const auto &decl : getAttrs)
        diag.attachNote(decl->getLoc()) << "__getattr__ defined here";
    }
  }

  ParsedArgumentList fnSignature;
  // Set up the known effects.
  if (isAsync) {
    fnSignature.effects.setAsync(true);

    if (funcOp.isDefaultedTraitFn()) {
      // TODO(MOCO-2287): Support async defaulted trait methods
      shared.emitError(funcOp.getLoc())
          << "async defaulted trait methods are not supported; remove the "
             "method or remove 'async'";
      return failure();
    }
  }

  // Parse the argument list next if present.
  if (fnSignature.parseArgumentListAndEffects(p, ArgListKind::kArgList))
    return failure();

  auto hasParameterDecorator = [&]() {
    return llvm::any_of(
        decoratorExprs,
        [](const std::pair<ExprNode *, LexerCursor> &decorator) {
          auto isParameterSpelling = [](StringRef spelling) {
            return spelling == "__parameter" || spelling == "parameter";
          };
          auto *expr = decorator.first->getWithoutParens();
          if (auto *declRef = dyn_cast<DeclRefNode>(expr))
            return isParameterSpelling(declRef->spelling);
          auto *call = dyn_cast<CallNode>(expr);
          if (!call)
            return false;
          auto *callee =
              dyn_cast<DeclRefNode>(call->callee->getWithoutParens());
          return callee && isParameterSpelling(callee->spelling);
        });
  };
  if (hasParameterDecorator())
    fnSignature.effects.setCapturing();
  bool isNonlegacyClosure =
      funcOp->getParentOfType<FnOp>() && !fnSignature.effects.isCapturing();

  // Keep the provisional FnOp signature in sync with early effect decisions so
  // signature typechecking for nested defs classifies the current function
  // correctly before the final typed signature is available.
  {
    FnTypeGeneratorType provisionalSig = funcOp.getFuncTypeGenerator();
    funcOp.setFuncTypeGenerator(FnTypeGeneratorType::get(
        provisionalSig.getInputParamTypes(), provisionalSig.getValues(),
        provisionalSig.getArgConventions(), fnSignature.effects,
        provisionalSig.getFnMetaOriginData(),
        provisionalSig.getParamListAttrs(), provisionalSig.getArgListAttrs()));
  }

  // TODO: effects parsing must be moved after captures parsing.
  // Non-legacy nested closures may have an optional `{...}` capture list.
  ParsedCaptureList captureSignature;
  if (isNonlegacyClosure && captureSignature.parseCaptureList(p))
    return failure();

  // Emit copies/casts for captures. Otherwise the incorrect lifetime rules will
  // be applied to the values in the closure.
  if (isNonlegacyClosure) {
    if (captureSignature.hasExplicitCaptureList &&
        failed(createCaptureValues(p, sigDecl, captureSignature, decl)))
      return failure();
    if (captureSignature.captureAllByConvention.has_value()) {
      shared.setDefaultCaptureForScope(
          decl, *captureSignature.captureAllByConvention);
    }
  }

  // Parse the result type if present.
  fnSignature.parseResultIfPresent(p);

  // Parse trailing body constraints if present.
  if (failed(parsedParamList.parseTrailingConstraintsIfPresent(p)))
    return failure();

  // Reject where clauses on trait methods. Users almost certainly expect
  // availability semantics (method absent when constraint fails), but `where`
  // gives callability (method exists, constraint checked at call site). To be
  // useful, the struct would also need a conditional conformance implying the
  // method's constraint, but nothing useful can be expressed in such a
  // constraint today.
  // See: https://www.notion.so/modularai/Conditional-Method-Availability for a
  // discussion on conditional availability vs. callability.
  if (!parsedParamList.bodyConstraints.empty() &&
      isa_and_nonnull<TraitDeclOp>(decl.getParentDecl()->getIfOperation())) {
    shared.emitError(funcOp.getLoc())
        << "'where' clauses on trait methods are not supported";
    return failure();
  }

  std::optional<TypeCheckedParamList> paramListOrError =
      TypeCheckedParamList::create(parsedParamList, sigDecl);
  if (!paramListOrError.has_value())
    return failure();
  TypeCheckedParamList &paramList = *paramListOrError;

  // Emit the argument and result types.
  TypeCheckedFnSignature tcSignature(paramList, fnSignature,
                                     /*captureOrigins=*/nullptr, &decl,
                                     baseName);

  // If any of the arguments had an error or if the result type is a type check
  // error, then we won't allow forming a reference to this function.
  if (sugarIsa<TypeCheckErrorType>(tcSignature.resultType.mlirType) ||
      llvm::any_of(fnSignature.parsedArgs,
                   [](ParsedArgument &arg) { return arg.isErroneous; }))
    decl.setErroneous();

  TraitType traitType;
  std::optional<ArrayRef<ParamDeclAttr>> parentParams;
  ASTDecl *structDecl = getStructOrTargetStruct(*decl.getParentDecl(), *this);
  StructDeclOp structOp = nullptr;
  if (structDecl)
    structOp = dyn_cast_or_null<StructDeclOp>(structDecl->getIfOperation());
  if (structOp) {
    traitType = structOp.getCanonicalTrait();
    parentParams = structOp.getParams();
  } else if (auto traitDecl = dyn_cast_or_null<TraitDeclOp>(
                 decl.getParentDecl()->getIfOperation())) {
    traitType = traitDecl.getCanonicalTrait();
  }
  if (isCapturingByDefault(shared, funcOp, traitType, parentParams,
                           paramList.paramDeclAttrs))
    fnSignature.effects.setCapturing();

  // Now that we have figured out the lexical structure, allow decorators to
  // take a crack at the signature.
  FnSigDecorators fnDecorators(decl, sigDecl, shared, baseName, tcSignature);
  Decorators(decl).applySignatureDecorators(
      decoratorExprs,
      [&](ExprNode *decorator) { return fnDecorators.applyOne(decorator); });
  fnDecorators.finalize();

  // Propagate errors and the parsed decls in the signature.
  decl.takeDecls(sigDecl);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  tcSignature.verifyFunctionNameBinding(decl, baseName);

  // Now that we've processed the signature, bail if we had a missing colon.
  if (p.parseToken(Token::colon, "expected ':' in function definition"))
    return failure();

  // Finally now that the full signature has been resolved, build our IR.
  // Handle argument effects, build the ASTDecls for the arguments, and write
  // the finalized signature attributes onto the FnOp (shared with the lambda
  // desugaring via finalizeResolvedFnOp).
  SmallVector<ConstraintAttr> closureExternalRefConstraints;
  FnTypeGeneratorType signature =
      finalizeResolvedFnOp(shared, funcOp, decl, tcSignature, paramList,
                           baseName, closureExternalRefConstraints);
  if (!signature)
    return failure();

  // Check for API author error: stable function should return stable types.
  checkStableFunctionReturnType(decl, ASTType(signature.getUserResultType()),
                                shared);

  // Set the symbol and notice if we are redeclaring something.
  if (Operation *existing = finalizeFuncSignature(funcOp, decl)) {
    const char *errorMessage = nullptr;
    auto existingFunc = cast<FnOp>(existing);

    // We need to compare the (name erased) user result types, since memory-only
    // types may result in `!kgen.none` in the mlir signature result.
    auto resTy = ASTType(signature.getUserResultType());

    // Loop through the args and check if any are keyword-only while overloading
    // the name of the argument.
    bool overloadedKeywordArgName = false;
    auto existingArgs =
        existingFunc.getFuncTypeGenerator().getArgListAttrs().getPogs();
    for (auto [arg, existingArg] :
         llvm::zip(tcSignature.argList.parsedArgs, existingArgs)) {
      if ((arg.kwArgHandling == KWArgHandling::kKeywordOnly ||
           existingArg.getPassingKind() == PassingKind::KwOnly) &&
          arg.name != existingArg.getName()) {
        overloadedKeywordArgName = true;
        break;
      }
    }

    if (!overloadedKeywordArgName) {
      auto existingResTy =
          ASTType(existingFunc.getFuncTypeGenerator().getUserResultType());
      if (!resTy.isEqualCanon(existingResTy))
        errorMessage = ", cannot overload on return type";
      else if (!llvm::equal(
                   existingFunc.getFuncTypeGenerator().getArgConventions(),
                   signature.getArgConventions()))
        errorMessage = ", cannot overload on argument conventions";
      else
        errorMessage = " with identical signature";
    }

    // On redefinition this is an overload of the same name.
    if (errorMessage) {
      auto diag = p.emitError(funcOp.getLoc(), "redefinition of function ")
                  << baseName << errorMessage;
      diag.attachNote(existing->getLoc()) << "previous definition here";
      decl.setErroneous();
    }
  }

  // If have a main function, def main(), export it automatically.
  if (!structDecl && baseName == kMainSymbolName)
    getDeclResolver().exportMain(decl);

  // Generate a debug subprogram for this function.
  shared.setLocationDebugScope(funcOp);
  for (auto [parsedArg, bbArg] :
       llvm::zip(fnSignature.parsedArgs, funcOp.getBody()->getArguments()))
    bbArg.setLoc(shared.diags.translateLocation(parsedArg.loc));

  auto notify = llvm::scope_exit(
      [&] { shared.notifyListenerOnFunctionDecl(decl, identifierLoc); });

  // Upon fully resolving a nonparametric closure, immediately materialize it as
  // a runtime value. It cannot be used as a parameter.
  if (!funcOp->getParentOfType<FnOp>())
    return success();

  // Fully resolve the body so we can swap the IR value of the decl. Later on,
  // we will need this to determine the capture signature.
  if (failed(resolveBody(funcOp, lexer, decl)))
    return failure();

  // Collect the captured values and parameter references from the resolved
  // body.
  BodyCaptures bodyCaptures = collectBodyCaptures(shared, decl, funcOp);
  SmallVector<Capture> &captures = bodyCaptures.values;
  SmallVector<ParamDeclRefAttr> &paramCaptures = bodyCaptures.paramRefs;

  // If this is a `@__parameter` closure, attach the capture origins.
  if (signature.isCapturing()) {
    SmallVector<Type> captureTypes;
    for (const Capture &cap : captures)
      captureTypes.push_back(cap.getValue().getType());
    for (ParamDeclRefAttr param : paramCaptures)
      captureTypes.push_back(param.getType());

    SmallVector<TypedAttr> origins =
        shared.cachedOriginFinder.findOriginsIn(captureTypes);
    signature = signature.getWithBody(signature.getBody().getWithMetadata(
        signature.getFnMetaOriginData().addCaptureOrigins(
            OriginSetAttr::get(getContext(), origins)),
        signature.getBody().getArgListAttrs()));
    funcOp.setFuncTypeGenerator(signature);

    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));
    funcOp.removeSymNameAttr();
    return success();
  }

  if (isNonlegacyClosure)
    return constructClosure(shared, decl, funcOp, captures, paramCaptures,
                            captureSignature, closureExternalRefConstraints,
                            signature, fnSignature.isExperimentalParamTrait);

  if (captures.empty() && captureSignature.parsedCaptures.empty() &&
      !captureSignature.captureAllByConvention) {
    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));
    funcOp.removeSymNameAttr();
    return success();
  }

  // abi("C") functions must be bare function pointers with no captured
  // state.  C calling convention has no closure mechanism, so a capturing
  // abi("C") function would silently pass the wrong values in argument
  // registers.
  if (signature.getFnEffects().isCABI() && !captures.empty())
    return emitError(funcOp.getLoc(),
                     "a abi(\"C\") function cannot capture variables");

  return emitError(funcOp.getLoc(),
                   "capturing nested functions cannot capture variables; omit "
                   "'capturing' to form a closure");
}

LogicalResult DeclResolver::resolveSyntheticBody(FnOp fn, ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

  StructEmitter gen(*decl.getParentDecl());

  if (fn.getInheritedFrom())
    return gen.populateDefaultedTraitFunction(decl);

  switch (fn.getSpecialFunctionKind()) {
  default:
    llvm_unreachable("unknown synthetic function to synthesize");
  case SpecialFunctionKind::kMoveCtor:
    (void)gen.populateMoveCopy(decl, /*isMove*/ true);
    return success();
  case SpecialFunctionKind::kCopyCtor:
    (void)gen.populateMoveCopy(decl, /*isMove*/ false);
    return success();
  }
}

ParseResult DeclResolver::resolveBody(FnOp funcOp, Lexer &lexer,
                                      ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

  Block &body = *funcOp.getBody();
  auto endFn = cast<EndFnOp>(body.front());

  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder) {
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

    // Reset the location on the endfn to correct debug scope.
    endFn->setLoc(shared.translateLocation(decl.getLoc()));
  }
  // About to parse the body.
  endFn.setUnresolved(false);

  // If this is an extern function or marked @unavailable, we only allow a
  // "..." as the body. If it's a trait method this must mean it's not
  // defaulted so we can early exit the function here as well.
  bool isUnavailableFn = funcOp.isUnavailable();
  if (isa_and_nonnull<TraitDeclOp>(decl.getParentDecl()->getIfOperation()) ||
      funcOp.isExternal() || isUnavailableFn) {
    // Skip any docstring's that might be present.
    ParserBase p(shared, lexer);
    p.parseDocString(decl);

    // If we see an ellipsis, the function member is well formed: don't emit
    // arguments or any other setup logic.
    if (p.consumeIf(Token::dot_dot_dot)) {
      body.front().erase(); // Remove the lit.endfn op to replace it.
      auto builder = OpBuilder::atBlockEnd(&body);
      UnreachableOp::create(builder, funcOp.getLoc());
      // Body decorators (e.g. @doc_hidden) are deferred from signature
      // resolution and are normally applied after the body is parsed (see the
      // call near the end of this function). Functions with a "..." body
      // (@unavailable, and required trait methods) return early here
      // and would otherwise skip that step, silently dropping their body
      // decorators -- which makes a @doc_hidden @unavailable function appear
      // to require a doc string. Apply them here as well.
      Decorators(decl).applyBodyDecorators(
          [&](ExprNode *decorator) { return failure(); });
      return success();
    }

    // @unavailable functions must have exactly "..." as their body.
    if (isUnavailableFn) {
      InflightDiag diag = shared.emitError(
          p.getToken().getLoc(),
          "unexpected function body in @unavailable function declaration, "
          "use '...'");
      diag.attachNote(funcOp.getLoc())
          << "in '" << funcOp.getDeclName().getValue() << "', declared here";
      return failure();
    }

    // Otherwise, must be a trait method with default implementation.

    // If a defaulted trait method should return a value but 'pass' is used,
    // emit an error. The user likely meant to use '...', so suggest that.
    if (auto tok = p.getToken();
        funcOp.isDefaultedTraitFn() && tok.is(Token::kw_pass)) {
      if (!ASTType(funcOp.getUserResultType()).isNoneType()) {
        InflightDiag diag = shared.emitError(
            tok.getLoc(),
            "trait method with a return type must not use 'pass'; use '...' "
            "to declare the method as required");
        diag.attachNote(funcOp.getLoc())
            << "in '" << funcOp.getDeclName().getValue() << "', declared here";
        diag.addFixIt(FixIt::replaceToken(tok.getLoc(), "..."));
        return failure();
      }
    }
  }

  // Set up information about value arguments, emitting before the lit.endfn.
  IREmitter emitter(decl, OpBuilder(&body.front()));

  // Set up the body of the fn/def, creating declarations for the value
  // parameters and adding them to the symbol table.
  FnTypeGeneratorType funcSignature = funcOp.getFuncTypeGenerator();
  for (auto [argIdxX, bbArg, convention] :
       llvm::enumerate(funcOp.getBody()->getArguments(),
                       funcSignature.getArgConventions())) {
    size_t argIdx = argIdxX;

    StringAttr argName = funcSignature.getArgName(argIdx);

    // Figure out which decl corresponds to this argument so we can finish it.
    ArrayRef<ASTDecl *> argDeclList = decl.lookupInCurrentScope(argName);

    // Don't bind anonymous result slots, they don't have a decl.
    if (argDeclList.empty() && isResultSlot(convention))
      continue;

    assert(argDeclList.size() == 1 &&
           "Argument should be added by signature resolution");
    ASTDecl &argDecl = *argDeclList[0];

    // The argDecl is already set up with a basic representation when the
    // function signature was type checked.  We have to hack it a bit for
    // variadics and other cases that aren't modeled right.
    // TODO: Move variadics to be formed on the caller side not the callee side.

    // This function sets the argument decl to be fully resolved with the
    // specified IR representation.
    auto setDecl = [&](DeclIRValue value) {
      argDecl.setIRValue(std::move(value));
      shared.notifyListenerOnArgumentDecl(argDecl, argName, argDecl.getLoc());
    };

    // Emit type refinement rebind if the argument has a parametric type
    // with known trait constraints from where clauses.
    Value refinedArg = bbArg;
    if (emitter.builder) {
      Location loc = decl.getShared().translateLocation(decl.getLoc());
      refinedArg =
          maybeEmitRefinementRebind(bbArg, decl, *emitter.builder, loc);
    }

    CValue argValue;
    if (convention == ArgConvention::ImmMem) {
      setDecl(MBValue(refinedArg)); // borrowed (possibly refined)
      continue;
    }
    if (convention == ArgConvention::ImmReg) {
      // borrowed_in_reg is used for TrivialRegisterPassable types. Use
      // SBValue (not SRValue) to preserve borrowed semantics, while still
      // keeping any type refinement rebind visible in the function body.
      setDecl(SBValue(refinedArg));
      continue;
    }

    // Ref convention works with registers and def functions without any funny
    // business.
    setDecl(CValue::getMValueForRef(refinedArg));
  }

  // If we had a named result in a register, create a var decl to hold the
  // temporary and register it for name lookup.
  if (!funcSignature.hasMemoryOnlyResult() && funcOp.getNamedResultAttr()) {
    // Emit a VarDeclOp for the temporary within the function.  This makes it
    // assignable etc.
    // This also provides a user name for the argument.
    StringAttr resultName = funcOp.getNamedResultAttr();
    // If this is the 'out' argument of an initializer, we use a special
    // VarDeclKind so CheckLifetimes knows the whole object bit is live on
    // input.
    auto kind = funcOp.getSpecialFunctionInfo().isInitializer()
                    ? VarDeclKind::InitOutArg
                    : VarDeclKind::Arg;
    VarDeclOp varDecl = emitter.emitVarDecl(
        resultName, funcOp.getUserResultType(), funcOp.getLoc(), kind);
    ASTDecl &argDecl = addFullyResolvedDecl(MLValue(varDecl), resultName,
                                            decl.getLoc(), &decl);
    shared.notifyListenerOnArgumentDecl(argDecl, resultName, argDecl.getLoc());
  }

  // With all the argument declarations set up, we can resolve the body of the
  // function.
  if (ParserBase(shared, lexer).parseSuite(decl))
    return failure();

  // If this decl or a parent is erroneous, return before emitting.  There is no
  // point to emitting after errors, and we might trip assertions because
  // erroneous decls don't respect invariants.
  if (decl.isErroneous() || decl.getParentDecl()->isErroneous())
    return success();

  // Determine whether we need an implicit return at the end of the function.
  // An implicit return is generated for functions that return None.
  bool needDefaultReturn = false;
  if (ASTType(funcOp.getUserResultType()).isNoneType() ||
      funcOp.getNamedResultAttr())
    needDefaultReturn = true;

  // We can elide the boilerplate if we can trivially the user already has a
  // return. This won't catch cases where an 'if' has two returns in the bodies
  // etc but is enough to avoid generating IR noise.
  if (emitter.builder->getInsertionPoint() != body.begin()) {
    if (isa<LIT::ReturnOp, LIT::RaiseOp>(
            std::prev(emitter.builder->getInsertionPoint())))
      needDefaultReturn = false;
  }

  // Emit a default "return None" if the function returns nothing.
  if (needDefaultReturn)
    emitter.emitNormalReturn(funcOp.getLoc(), Value(), /*emitEndFunc=*/false);

  // Now that the body of the function is parsed, run any body decorators.
  Decorators(decl).applyBodyDecorators(
      [&](ExprNode *decorator) { return failure(); });

  // If this function is @always_inline("builtin"), check that its body obeys
  // the right invariants.
  if (funcOp.getInlineLevel() == InlineLevel::AlwaysBuiltin) {
    if (failed(FnSigDecorators::checkAlwaysInlineBuiltin(funcOp, shared)))
      funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);
  }

  if (funcOp.isExternal()) {
    shared.emitError(decl.getLoc(),
                     "unexpected function body in extern function "
                     "declaration, use `...`");
    return success();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Module Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::FileModuleOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

  // Push a scope for the file of this module.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (shared.diBuilder) {
    FileLineColLoc loc =
        shared.diags.translateLocation(lexer.getToken().getLoc());
    if (loc)
      fileGuard = shared.diBuilder->pushFile(loc.getFilename().getValue());
  }

  return ParserBase(shared, lexer).parseSuite(decl);
}

//===----------------------------------------------------------------------===//
// Package Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::PackageOp op, ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

  // A source package corresponds to a directory. Its children (modules
  // and sub-packages) become decls, but the package's own importable scope
  // stays empty. This is because the package's public surface is its __init__,
  // which external lookups are redirected to. Children are unlisted decls, so a
  // file inside the package can't see a sibling via upward lexical lookup, and
  // a hidden submodule isn't reachable through component access. The children
  // are, however, navigable via ModuleState::nestedModules and are present in
  // the package IR; just not in declsInScope.

  // Grab the directory that this package is defined in.
  std::optional<std::string> directoryStr = shared.getModuleSourcePath(decl);
  if (!directoryStr)
    return emitError(op.getLoc(), "unable to locate package directory");

  std::error_code ec;
  std::filesystem::path directory(*directoryStr);
  if (!std::filesystem::is_directory(directory, ec) || ec)
    return emitError(op.getLoc(), "unable to locate package directory");

  // Resolve __init__ (the public surface) if present, inheriting its docstring.
  if (std::filesystem::exists(directory / "__init__.mojo", ec) && !ec) {
    ASTDecl &initModule = shared.importModule(
        SharedState::ImportPath({"__init__"}, /*relativeLevel=*/1), op,
        decl.loc);
    if (!initModule.isErroneous()) {
      if (failed(resolveBody(initModule, decl.loc)))
        return failure();
      if (auto initDeclOp =
              dyn_cast_or_null<ASTDeclInterface>(initModule.getIfOperation()))
        if (auto docstring = initDeclOp.getDocStringAttr())
          op.setDocStringAttr(docstring);
    }
  }

  // Register a (deferred, unlisted) child decl for every sibling module and
  // sub-package.
  shared.getModuleLoader().registerSourcePackageChildren(decl);

  return success();
}

//===----------------------------------------------------------------------===//
// Alias Decl implementation
//===----------------------------------------------------------------------===//

/// alias_decl_stmt ::=
///   | "alias" identifier [param_signature] ":" expression
///                                          ("where" expression)*
///                                          ["=" expression]
///   | "alias" identifier [param_signature] ("where" expression)* "="
///   expression
LogicalResult DeclResolver::resolveSignature(AliasDeclOp aliasDeclOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);

  // Parse the decorators for alias declarations but only process them if they
  // are outside function bodies. This is because decorators inside function
  // bodies are not allowed, and this prevents redundant (and potentially
  // confusing) error messages.
  auto decoratorExprs = p.parseDecorators(decl);
  if (!isa_and_nonnull<FnOp>(decl.getParentDecl()->getIfOperation())) {
    Decorators(decl, /*signatureOnly=*/true)
        .applySignatureDecorators(decoratorExprs);
  }

  // Parse the type if present. Accept 'comptime' keyword.
  if (!p.consumeIf(Token::kw_comptime)) {
    p.emitError(p.getToken().getLoc(),
                "internal error: checked by stmt parser");
    return failure();
  }

  SMLoc identifierLoc;
  if (p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc))
    return failure();

  // Parse the param signature if present.
  ParsedParamList parsedParams;
  if (parsedParams.parseParametersIfPresent(p, ArgListKind::kParamList))
    return failure();

  // Parse the alias body type as a raw expression (no IR emission yet).
  // Deferring emission lets us parse and emit any trailing 'where' clauses
  // first, so the type's emission sees the body constraints in scope.
  ExprNode *bodyTypeExpr = nullptr;
  if (p.consumeIf(Token::colon)) {
    if (p.parseExpression(bodyTypeExpr, decl.getIndentation()))
      return failure();
  }

  // Parse trailing 'where' clauses if present.
  if (parsedParams.parseTrailingConstraintsIfPresent(p))
    return failure();

  // The alias signature is a self-contained scope where the input parameters
  // of the alias are visible by all types.  We must use a temporary
  // declaration here (with an empty name) because we don't want references to
  // the alias itself to resolve to a fully-resolved decl, but we need a
  // fully-resolved decl for incremental lookups within the scope to work out.
  ASTDecl &sigDecl =
      addFullyResolvedDecl(aliasDeclOp.getOperation(), StringAttr(),
                           decl.getLoc(), decl.getParentDecl());

  std::optional<TypeCheckedParamList> paramSignatureOrError =
      TypeCheckedParamList::create(parsedParams, sigDecl);
  if (!paramSignatureOrError.has_value())
    return failure();
  TypeCheckedParamList &paramSignature = *paramSignatureOrError;

  // Emit the trailing body constraints. This seeds `sigDecl`'s known
  // assumptions with them so any subsequent emission (including the alias
  // type below) can rely on them.
  paramSignature.emitBodyConstraints();

  // Now emit the alias body type (if any).
  ASTType type;
  if (bodyTypeExpr) {
    IREmitter emitter(sigDecl, EC_Type);
    type = emitter.emitExprType(bodyTypeExpr, /*allowUnbound=*/true);
    if (!type)
      return failure();
  }

  // Parse & emit the initializer if one exists. If one does not exist, check
  // for errors because the initializer can only be emitted in certain contexts.
  TypedAttr initValue;
  if (p.consumeIf(Token::equal)) {
    ExprNode *initExpr = nullptr;
    if (p.parseVarInitExpression(initExpr, decl.getIndentation()))
      return failure();

    IREmitter emitter(sigDecl, EC_AliasValue);

    // Parametric aliases become generator types whose *body* type may still
    // mention unbound / infer-only parameters (e.g. `Origin[mut=mut]` leaves
    // `_mlir_origin` open). That type annotation maybe dependent on the
    // parameters of the comptime.
    ASTType emissionType = type;
    if (type && type.hasUnboundParameters()) {
      if (paramSignature.paramDeclAttrs.empty()) {
        p.emitError(initExpr->getLoc(),
                    "cannot construct a value with parametric type: ")
            << type << initExpr->getRange();
        return failure();
      }
      emissionType = {};
    }

    // Emit the value and convert to the expected type if we know it and it is
    // concrete enough to convert into.
    PValue rhsValue =
        emitter.emitExprPValue(initExpr, EC_AliasValue, emissionType);
    if (!rhsValue)
      return failure();

    // If we had no declared body type (`alias x = 42`), or if it was parametric
    // in a way we couldn't use, infer the body type from the initializer.
    type = rhsValue.getType();
    initValue = rhsValue.get();
  } else {
    ASTDecl &parentDecl = *decl.getParentDecl();
    if (!isa_and_nonnull<LIT::TraitDeclOp>(parentDecl.getIfOperation())) {
      // Disallow this, because it would create diamond inheritance problems.
      p.emitError(identifierLoc)
          << "only traits may contain a comptime member without an initializer";
      return failure();
    }

    if (!type) {
      p.emitError(identifierLoc)
          << "comptime value without an initializer must have a type";
      return failure();
    }
  }

  // If there are input parameters or constraints, the actual type of the alias
  // is a generator type. Parameterize the body type & initializer value with
  // the input parameters.
  bool needsGeneratorType = !paramSignature.paramDeclAttrs.empty() ||
                            !paramSignature.emittedBodyConstraints.empty();
  if (needsGeneratorType) {
    // The body type of the alias is a standalone type, so it needs to reference
    // its input parameters by index refs (IRAIDAI), not name refs. This
    // remapper handles converting the name refs to index refs.
    IndexRefRemapper remapper(paramSignature.paramDeclAttrs, {});

    SmallVector<Type> inputParamTypes;
    for (ParamDeclAttr param : paramSignature.paramDeclAttrs)
      inputParamTypes.push_back(remapper.replace(param.getType()));

    PogListAttr metadata = remapper.replace(paramSignature.getParamListAttr());
    type = GeneratorType::get(inputParamTypes, remapper.replace(type.mlirType),
                              metadata);

    // Wrap the initializer in a generator attr too.
    if (initValue) {
      TypedAttr remappedBody = remapper.replace(initValue);
      initValue = GeneratorAttr::get(inputParamTypes, remappedBody, metadata);
    }
  }

  NamedAttrList attrs = aliasDeclOp->getAttrDictionary();
  // Remember the value
  if (initValue)
    attrs.set(aliasDeclOp.getValueAttrName(), initValue);

  // Propagate signature errors and decls.
  decl.takeDecls(sigDecl);

  // Update the type from UnresolvedType
  attrs.set(aliasDeclOp.getParamDeclAttrName(),
            ParamDeclAttr::get(aliasDeclOp.getName(), type));
  aliasDeclOp->setAttrs(attrs.getDictionary(decl.getContext()));

  // Process the doc string of the alias.
  p.parseDocString(decl);

  shared.notifyListenerOnAliasDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(AliasDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

struct ParsedTraitConstraint {
  TraitSymbolAttr traitSymbol;
  ConstraintAttr constraint;
  /// Whether this constraint was for an explicitly listed trait in the
  /// conformance list (vs propagated from an ancestor).
  bool isExplicit;
};

/// A single entry in a parsed conformance list: a type expression naming a
/// parent trait, plus an optional `where` constraint expression for
/// conditional conformance. Used as the intermediate representation between
/// `parseOptionalConformanceListSyntax` (parsing) and `resolveConformanceList`
/// (type resolution + IR emission).
struct ParsedConformanceEntry {
  ExprNode *typeExpr = nullptr;
  SMLoc loc;
  std::optional<ParsedConstraint> constraint;
};

/// Verify that each explicitly listed derived trait's constraint implies its
/// explicitly listed ancestor's constraint. This enforces that whenever a
/// derived trait is available, its ancestors are also available.
///
/// Example -- rejected because `condA` does not imply `condB`:
///
///   trait Derived(Base): ...
///   struct S(Derived where condA, Base where condB): ...
///
/// Example -- rejected because unconditional `Derived` always requires `Base`,
/// so `Base` cannot be conditional:
///
///   struct S(Derived, Base where cond): ...
///
/// Example -- accepted because `condA and condB` implies `condB`:
///
///   struct S(Derived where condA and condB, Base where condB): ...
///
/// Returns failure if any implication errors were found.
static LogicalResult verifyDerivedAncestorImplication(
    const DenseMap<TraitSymbolAttr, ConstraintAttr> &explicitConstraints,
    SharedState &shared) {
  bool hasErrors = false;
  for (const auto &[symbol, constraint] : explicitConstraints) {
    TypedAttr prop = constraint.getProposition();

    ASTDecl &traitDecl =
        shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
    auto traitDeclOp =
        dyn_cast_or_null<TraitDeclOp>(traitDecl.getIfOperation());
    if (!traitDeclOp)
      continue;

    for (TraitSymbolAttr ancestor :
         traitDeclOp.getCanonicalTrait().getSymbols()) {
      if (ancestor == symbol)
        continue;

      auto it = explicitConstraints.find(ancestor);
      if (it == explicitConstraints.end())
        continue; // Not explicitly listed -- handled by propagation.

      TypedAttr ancestorProp = it->second.getProposition();

      if (!isImplicationProven(ancestorProp, prop)) {
        shared.emitError(constraint.getLoc())
            << "constraint for " << symbol.getSymbol().getLeafReference()
            << " does not imply constraint for ancestor trait "
            << ancestor.getSymbol().getLeafReference()
            << "; strengthen the derived constraint by adding the ancestor's "
               "constraint with 'and'";
        hasErrors = true;
      }
    }
  }
  return failure(hasErrors);
}

/// Resolve propagated (non-explicit) ancestor constraints after all explicit
/// constraints have been collected. Applies three rules:
///   - Any unconditional path makes the ancestor unconditional.
///   - Single path or all paths agree: auto-propagate the constraint.
///   - Multiple paths disagree (diamond): require explicit listing.
///
/// Returns failure if any diamond errors were found.
static LogicalResult resolvePropagatedConstraints(
    const DenseMap<TraitSymbolAttr, SmallVector<ConstraintAttr, 2>> &propagated,
    DenseMap<TraitSymbolAttr, ConstraintAttr> &traitConstraints,
    SharedState &shared) {
  ConstraintAttr unconditional =
      getUnconditionalConstraint(shared.getContext());
  bool hasErrors = false;

  for (const auto &[symbol, paths] : propagated) {
    assert(!paths.empty() && "propagated constraint entry cannot be empty");

    // Any unconditional path makes the ancestor unconditional.
    bool hasUnconditionalPath = llvm::any_of(paths, [](ConstraintAttr c) {
      return isTriviallyTrueProposition(c.getProposition());
    });
    if (hasUnconditionalPath) {
      traitConstraints[symbol] = unconditional;
      continue;
    }

    // Single path or all paths carry the same constraint: auto-propagate.
    // Canonicalization already normalizes operand order for commutative ops
    // like AND/OR, so structural equality suffices here.
    TypedAttr firstProp = getCanonicalAttr(paths.front().getProposition());
    bool allSame = llvm::all_of(paths, [&](ConstraintAttr c) {
      return getCanonicalAttr(c.getProposition()) == firstProp;
    });

    if (allSame) {
      // Carry the derived clause's message down to the propagated ancestor
      // constraint, so requiring the bare ancestor still surfaces it. The note
      // points at the originating `where` clause, so naming the derived trait
      // on an ancestor failure still gives the user the full picture. If paths
      // agree on the proposition but carry different messages, none is correct
      // for the merged result, so drop it.
      StringAttr message = paths.front().getMessage();
      bool allSameMessage = llvm::all_of(
          paths, [&](ConstraintAttr c) { return c.getMessage() == message; });
      traitConstraints[symbol] =
          ConstraintAttr::get(firstProp, paths.front().getLoc(),
                              allSameMessage ? message : StringAttr());
      continue;
    }

    // Diamond: multiple paths disagree -- require explicit listing.
    shared.emitError(paths.front().getLoc())
        << "ancestor trait " << symbol.getSymbol().getLeafReference()
        << " is reached via multiple inheritance paths with different "
           "constraints; it must be explicitly listed in the conformance "
           "list with the desired constraint";
    hasErrors = true;
  }
  return failure(hasErrors);
}

/// Build the trait constraint map from parsed constraints and explicit trait
/// information. Partitions constraints into explicit vs. propagated, verifies
/// derived->ancestor implication for explicit traits, and resolves propagated
/// constraints for ancestor traits.
///
/// Given a conformance list like:
///
///   trait Base: ...
///   trait Mid(Base): ...
///   trait Other(Base): ...
///   struct S(Mid where cond, Other where cond): ...
///
/// The parsed constraints contain both explicit entries (`Mid where cond`,
/// `Other where cond`) and propagated ancestors (`Base` via `Mid` with `cond`,
/// `Base` via `Other` with `cond`). This function:
///   1. Records explicit constraints and checks for duplicates.
///   2. Verifies derived->ancestor implication (see
///      `verifyDerivedAncestorImplication`).
///   3. Resolves propagated ancestors -- here `Base` is reached via two paths
///      that agree on `cond`, so it auto-propagates. If they disagreed (e.g.,
///      `Mid where condA, Other where condB`), an error requires the user to
///      explicitly list `Base` with the desired constraint.
///
/// `compilerInjectedTraits` contains canonical symbols for traits that the
/// compiler injects into every struct's parent list when the stdlib is
/// available (e.g., `AnyType`, `Deinitable`). These are treated
/// as unconditionally available and never require explicit listing. The set
/// may be empty when builtins are disabled (`--mojo-disable-builtins`).
///
/// Returns failure if any constraint errors were found.
static LogicalResult buildTraitConstraintsMap(
    ArrayRef<ParsedTraitConstraint> parsedConstraints,
    const DenseSet<TraitSymbolAttr> &explicitTraits,
    const DenseSet<TraitSymbolAttr> &compilerInjectedTraits,
    DenseMap<TraitSymbolAttr, ConstraintAttr> &traitConstraints,
    SharedState &shared) {
  ConstraintAttr unconditional =
      getUnconditionalConstraint(shared.getContext());
  DenseMap<TraitSymbolAttr, ConstraintAttr> explicitConstraints;
  DenseMap<TraitSymbolAttr, SmallVector<ConstraintAttr, 2>> propagated;
  bool hasErrors = false;

  // Partition parsed constraints into explicit and propagated.
  for (const auto &pc : parsedConstraints) {
    TypedAttr prop = pc.constraint.getProposition();

    if (pc.isExplicit) {
      auto newConstraint = ConstraintAttr::get(prop, pc.constraint.getLoc(),
                                               pc.constraint.getMessage());
      auto [it, inserted] =
          explicitConstraints.try_emplace(pc.traitSymbol, newConstraint);
      // Catches cases where a trait is listed twice with different constraints.
      // Canonicalization normalizes operand order for commutative ops, so
      // structural equality after canonicalization suffices here.
      if (!inserted && getCanonicalAttr(it->second.getProposition()) !=
                           getCanonicalAttr(prop)) {
        shared.emitError(pc.constraint.getLoc())
            << "trait '" << pc.traitSymbol.getSymbol().getLeafReference()
            << "' appears multiple times in the conformance list with "
               "different constraints";
        hasErrors = true;
      }
      continue;
    }

    // The explicit constraint takes precedence over any propagated one.
    if (explicitTraits.contains(pc.traitSymbol))
      continue;

    // Compiler-injected traits are always unconditionally available --
    // don't require explicit listing.
    if (compilerInjectedTraits.contains(pc.traitSymbol)) {
      traitConstraints.try_emplace(pc.traitSymbol, unconditional);
      continue;
    }

    // Store the whole constraint (proposition, location, and message) so the
    // derived clause's message can be carried down to the propagated ancestor.
    propagated[pc.traitSymbol].push_back(pc.constraint);
  }

  // Verify derived->ancestor implication for explicitly listed traits.
  // This operates on explicitConstraints only, before propagated entries are
  // resolved, so it cannot accidentally inspect propagated ancestors.
  if (failed(verifyDerivedAncestorImplication(explicitConstraints, shared)))
    hasErrors = true;

  // Merge explicit constraints into the output map (after verification).
  traitConstraints.insert(explicitConstraints.begin(),
                          explicitConstraints.end());

  // Resolve propagated constraints for non-explicit ancestor traits.
  if (failed(
          resolvePropagatedConstraints(propagated, traitConstraints, shared)))
    hasErrors = true;

  return failure(hasErrors);
}

//===----------------------------------------------------------------------===//
// Conformance list processing
//
// Conformance list processing happens in two phases:
//   1. `parseOptionalConformanceListSyntax` consumes the tokens of the
//      conformance list and stores each entry as a `ParsedConformanceEntry`
//      (type expression + optional `where` constraint expression). No types
//      are resolved and no IR is emitted in this phase.
//   2. `resolveConformanceList` walks the parsed entries, resolves each type
//      expression, emits any `where`-clause constraints, and populates the
//      caller's parent / constraint / lineage tables.
// Callers are expected to invoke them in order on the same `ASTDecl`.
//===----------------------------------------------------------------------===//

/// For a struct or trait declaration, parse an optional conformance list
/// without resolving the trait types or emitting constraints.
///
/// Set `allowConformanceConstraints` to `false` for declarations that don't
/// support conditional conformance (traits and extensions).
static ParseResult parseOptionalConformanceListSyntax(
    ParserBase &p, SmallVectorImpl<ParsedConformanceEntry> &parsedConformances,
    std::optional<size_t> stmtIndent, bool allowConformanceConstraints) {
  if (!p.consumeIf(Token::l_paren) || p.consumeIf(Token::r_paren))
    return success();

  auto parseConformance = [&]() -> ParseResult {
    ParsedConformanceEntry conformance;
    if (p.getLocation(conformance.loc) ||
        p.parseExpression(conformance.typeExpr, stmtIndent))
      return failure();

    SMLoc whereLoc = p.getToken().getLoc();
    if (p.consumeIfSoftIdentifier("where")) {
      if (!allowConformanceConstraints) {
        return p.emitError(
            whereLoc,
            "'where' clauses in conformance lists are only supported on "
            "structs");
      }
      ParsedConstraint constraint;
      constraint.loc = whereLoc;
      ExprNode *parsed;
      if (p.parseExpression(parsed, stmtIndent))
        return failure();
      // A message is written `where (condition, "message")`. Because the
      // message lives inside the parentheses, the trailing comma that
      // separates the next conformance entry is unambiguous -- no lookahead
      // is needed here.
      if (constraint.extractParenthesizedMessage(p, parsed))
        return failure();
      conformance.constraint = constraint;
    }

    parsedConformances.push_back(conformance);
    return success();
  };
  if (p.parseCommaSeparatedList(parseConformance, Token::r_paren) ||
      p.parseToken(Token::r_paren, "expected ')' for conformance list"))
    return failure();
  return success();
}

/// For a parsed conformance list, resolve parent trait types and emit
/// conformance-local constraints. `immediateParents` will be populated with
/// the smallest set of equivalent parent trait decls.
///
/// For struct declarations with parameters already in scope,
/// `traitConstraints` will be populated with a constraint entry for every
/// trait encountered (explicit and propagated ancestors). Entries from `where`
/// clauses carry the emitted constraint; entries without a `where` clause
/// carry the trivially-true (unconditional) constraint. Parameters must be in
/// scope in declScope for constraint emission to work correctly.
///
/// `explicitTraits` (if provided) will be populated with all traits that are
/// explicitly listed in the conformance list (regardless of whether they have
/// a where clause). This is used to give explicit constraints precedence over
/// propagated ones during constraint map building.
static ParseResult resolveConformanceList(
    ArrayRef<ParsedConformanceEntry> parsedConformances, ASTDecl &declScope,
    ASTDecl &decl, SharedState &shared,
    DenseSet<TraitSymbolAttr> &immediateParents,
    SmallVectorImpl<ParsedTraitConstraint> *traitConstraints = nullptr,
    DenseSet<TraitSymbolAttr> *explicitTraits = nullptr) {

  // Helper to emit an error for a non-trait type in a conformance list.
  auto emitNonTraitTypeInConformanceListError = [&](Type type, SMLoc loc) {
    bool isTrait = isa_and_nonnull<TraitDeclOp>(decl.getIfOperation());
    if (sugarIsa<LIT::StructType>(type)) {
      if (isTrait) {
        shared.emitError(loc)
            << "traits only refine other traits; remove the struct type "
               "from the refinement list";
      } else {
        shared.emitError(loc)
            << "structs only conform to traits or trait compositions; "
               "remove the struct type from the conformance list";
      }
    } else if (sugarIsa<ParamType>(type)) {
      if (isTrait)
        shared.emitError(loc)
            << "traits only refine other traits; remove the type parameter "
               "from the refinement list";
      else
        shared.emitError(loc)
            << "structs only conform to traits or trait compositions; "
               "remove the type parameter from the conformance list";
    } else {
      shared.emitError(loc)
          << "refinement and conformance lists may only contain traits";
    }
  };

  if (parsedConformances.empty())
    return success();

  DenseMap<TraitSymbolAttr, std::pair<TraitSymbolAttr, SMLoc>> *inheritedFrom =
      decl.getTraitConformanceLineage(/*createIfMissing=*/true);

  for (const ParsedConformanceEntry &conformance : parsedConformances) {
    IREmitter typeEmitter(declScope, EC_Type);
    ASTType type = typeEmitter.emitExprType(conformance.typeExpr,
                                            /*allowUnbound=*/false);
    if (!type)
      return failure();

    // Reject non-trait types in refinement and conformance lists.
    auto traitType = sugarDynCast<TraitType>(type);
    if (!traitType) {
      emitNonTraitTypeInConformanceListError(type, conformance.loc);
      declScope.setErroneous();
      continue;
    }

    // Emit optional where clause for conditional conformance.
    // Parameters must already be in scope in declScope for this to work.
    // Defaults to the trivially-true (unconditional) constraint with the
    // trait's source location (for diagnostic accuracy).
    ConstraintAttr constraint =
        traitConstraints
            ? ConstraintAttr::get(
                  SIMDAttr::getScalarBool(shared.getContext(), true),
                  shared.diags.translateLocation(conformance.loc),
                  /*message=*/StringAttr())
            : ConstraintAttr();
    if (traitConstraints && conformance.constraint) {
      IREmitter constraintEmitter(declScope, EC_Requires);
      RValue prop = constraintEmitter.emitExprScalarBool(
          conformance.constraint->propExpr, EC_Requires);
      if (!prop) {
        constraintEmitter.emitError(conformance.loc,
                                    "failed to emit constraint expression");
        return failure();
      }
      PValue propVal = prop.getIfPValue();
      if (!propVal) {
        constraintEmitter.emitErrorForDynamicValueInParameter(conformance.loc);
        return failure();
      }
      TypedAttr simplifiedProp = LIT::deShortCircuitCond(propVal);
      constraint = ConstraintAttr::get(
          simplifiedProp, shared.diags.translateLocation(conformance.loc),
          conformance.constraint->message);
    }

    // If we want to extra information to detect conflict conditional
    // conformance constraints, set them now.
    if (explicitTraits || traitConstraints) {
      // We used the reduced set to detect conflict constraints.
      SmallVector<TraitSymbolAttr> reduced =
          reduceTraitCompositionSymbols(shared, traitType.getSymbols());
      if (explicitTraits)
        explicitTraits->insert_range(reduced);

      if (traitConstraints) {
        for (TraitSymbolAttr symbol : reduced) {
          traitConstraints->push_back(
              {symbol, constraint, /*isExplicit=*/true});
        }
      }
    }

    // Successively flatten the parent list so we always have all the parents
    // available to check.
    // TODO: Encode an "inherited from" here, to make diagnostics nice.
    for (TraitSymbolAttr symbol : traitType.getSymbols()) {
      // If this symbol is already a parent, skip further processing.
      // Note: The explicit constraint was already recorded above.
      if (inheritedFrom->contains(symbol))
        continue;
      ASTDecl &traitDecl =
          shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());

      // Check for API author error: stable trait cannot inherit from unstable.
      // Only applies when `decl` is a trait (not a struct conforming to a
      // trait).
      if (isa<TraitDeclOp>(decl.getIfOperation())) {
        // Need to sig resolve to get stable marker.
        if (failed(shared.declResolver->resolveSignature(traitDecl, SMLoc())))
          return failure();
        checkStableTraitInheritance(decl, traitDecl, declScope, shared);
      }

      TraitType canonicalParent =
          cast_or_null<TraitDeclOp>(traitDecl.getIfOperation())
              .getCanonicalTrait();

      auto paramEvaluator = populateTraitBindingEvaluator(symbol, shared);
      for (TraitSymbolAttr ancestor : canonicalParent.getSymbols()) {
        if (paramEvaluator)
          ancestor = paramEvaluator->replace(ancestor);
        // It is probably fine to relax `ancestor.isFullyResolved()` here, just
        // to catch any unexpected behavior before exposing parametric traits to
        // users. It is now impossible to inherit from a parametric trait.
        assert(ancestor.isFullyResolved() || ancestor == symbol);

        inheritedFrom->try_emplace(ancestor,
                                   std::make_pair(symbol, conformance.loc));

        // Any immediate parent that is actually a parent of this `symbol` is no
        // longer an immediate parent.
        immediateParents.erase(ancestor);

        // Propagate the constraint to ancestor traits (the immediate symbol
        // was already added above as an explicit entry).
        // Each path's constraint is collected; the final constraint map
        // builder checks that all paths to the same ancestor agree, or
        // requires explicit listing if they disagree (diamond case).
        if (traitConstraints && ancestor != symbol)
          traitConstraints->push_back(
              {ancestor, constraint, /*isExplicit=*/false});
      }
      // Insert this `symbol` as an immediate parent. This must happen after the
      // loop, because this symbol itself is part of `canonicalParent` too.
      immediateParents.insert(symbol);
    }
  }
  return success();
}

bool isTrivialRegisterPassable(CallNode *callNode) {
  if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
    if (declRef->spelling == "register_passable" &&
        callNode->operands.size() == 1 &&
        callNode->operands[0].isPositionalStringLiteral("trivial")) {
      return true;
    }
  }
  return false;
}

/// Process the @align(N) decorator on structs.
/// Returns true if the decorator was handled (with or without errors),
/// false if not an align decorator.
static bool processAlignDecoratorHelper(ExprNode *alignExpr,
                                        StructDeclOp structOp,
                                        SharedState &shared, ASTDecl &decl) {
  // Emit the alignment expression as an index-typed value. This handles
  // both integer literals and parametric expressions.
  IREmitter emitter(decl, EC_Decorator);
  CValue alignCValue = emitter.emitIndex(alignExpr, EC_Decorator);
  if (!alignCValue) {
    // Error already emitted by IREmitter.
    return true;
  }

  PValue alignPValue = alignCValue.getIfPValue();
  if (!alignPValue) {
    shared.emitError(alignExpr->getLoc(),
                     "@align requires a compile-time value");
    return true;
  }

  TypedAttr alignAttr = alignPValue.get();

  // If this resolved to an immediate integer value (which is expected to be the
  // common case), validate it early for faster feedback.
  if (auto intAttr = dyn_cast<IntegerAttr>(alignAttr)) {
    int64_t alignVal = intAttr.getInt();

    // Validate: must be positive power of 2
    if (alignVal <= 0 || !llvm::isPowerOf2_64(alignVal)) {
      shared.emitError(alignExpr->getLoc(),
                       "@align value must be a positive power of 2");
      return true;
    }

    // Validate: must not exceed reasonable upper bound (2^29 bytes = 512MB).
    // This matches common compiler limits and avoids overflow issues.
    constexpr int64_t kMaxAlignment = 1LL << 29;
    if (alignVal > kMaxAlignment) {
      shared.emitError(alignExpr->getLoc(),
                       "@align value exceeds maximum alignment (2^29)");
      return true;
    }
  }

  structOp.setMinAlignmentAttr(alignAttr);
  return true;
}

/// Process a decorator that is resolved at the signature phase of resolution
/// and return success, otherwise failure if it is handled later.
/// `sigDecl` is the fully-resolved signature scope with struct parameters.
static LogicalResult processStructSignatureDecorator(ExprNode *decorator,
                                                     StructDeclOp structOp,
                                                     SharedState &shared,
                                                     ASTDecl &sigDecl) {
  ASTDecl *parentDecl = sigDecl.getParentDecl();

  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "register_passable") {
      shared.emitError(decorator->getLoc(),
                       "decorator @register_passable is removed, conform to "
                       "RegisterPassable trait instead");

      return failure();
    }
    // @align without parentheses is an error
    if (declRef->spelling == "align") {
      shared.emitError(decorator->getLoc(),
                       "@align requires exactly one argument");
      return success();
    }
    // We don't process @explicit_destroy here, we do it in resolveSignature.
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      if (isTrivialRegisterPassable(callNode)) {
        shared.emitError(decorator->getLoc(),
                         "decorator @register_passable(\"trivial\") is removed,"
                         " conform to TrivialRegisterPassable trait instead");
        return failure();
      }

      // @__nonmaterializable(TargetType)
      if (declRef->spelling == "__nonmaterializable" &&
          callNode->operands.size() == 1) {
        if (auto drn = dyn_cast<DeclRefNode>(callNode->operands[0].expr)) {
          IREmitter emitter(*parentDecl, EC_Type);
          if (ASTType t = emitter.emitExprType(drn)) {
            structOp.setNonmaterializableTargetAttr(TypeAttr::get(t.mlirType));
            return success();
          }
        }
      }

      // @align(N) - specify minimum alignment for the struct
      if (declRef->spelling == "align") {
        if (callNode->operands.size() != 1) {
          shared.emitError(decorator->getLoc(),
                           "@align requires exactly one argument");
          return success();
        }
        processAlignDecoratorHelper(callNode->operands[0].expr, structOp,
                                    shared, sigDecl);
        return success();
      }
    }
  }
  // Not handled in signature phase.
  return failure();
}

/// Silence internal verifier errors when constructing types from the parser. We
/// don't want to show these to the user.
static auto silenceErrors(MLIRContext *ctx) {
  return [ctx] {
    InFlightDiagnostic diag = mlir::emitError(UnknownLoc::get(ctx));
    diag.abandon();
    return diag;
  };
}

// True when the struct's own where-clause makes `condition` unsatisfiable.
static bool conformancePrecluded(ArrayRef<TypedAttr> bodyPropositions,
                                 TypedAttr condition) {
  TypedAttr negated = ParamOperatorAttr::getNot(condition);
  return isPropositionImplied(negated, bodyPropositions).isTrue();
}

// Returns the condition under which the struct conforms to the trait (a
// trivially-true constraint for an unconditional conformance), or std::nullopt
// when the struct does not conform to the trait at all (including if the
// conformance condition is precluded by the struct's own body constraints).
static std::optional<ConstraintAttr>
getConformanceCondition(ASTDecl &structDecl, StringRef traitName) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl.getIfOperation());
  TraitType canonicalTrait = structOp.getCanonicalTrait();
  ArrayRef<TraitSymbolAttr> symbols = canonicalTrait.getSymbols();
  ArrayRef<ConstraintAttr> constraints = canonicalTrait.getConstraints();

  SmallVector<TypedAttr> bodyPropositions = llvm::map_to_vector(
      structOp.getSignature().getParamListAttrs().getBodyConstraints(),
      [&](ConstraintAttr constraint) { return constraint.getProposition(); });
  IndexRefRemapper remapper(structOp.getParamsAttr());

  for (auto [i, symbol] : llvm::enumerate(symbols)) {
    if (symbol.getSymbol().getLeafReference() != traitName)
      continue;

    MLIRContext *ctx = structDecl.getShared().getContext();

    // An empty constraint array is the canonical form for a fully
    // unconditional trait, so every slot is trivially true.
    if (constraints.empty())
      return getUnconditionalConstraint(ctx);

    ConstraintAttr condition = constraints[i];
    assert(!isTriviallyFalseConstraint(condition) &&
           "trivially false conformance should have been erased during "
           "canonical trait construction");

    // A condition the struct's own trailing where-clause precludes is an
    // opt-out.
    if (!bodyPropositions.empty() &&
        conformancePrecluded(bodyPropositions,
                             remapper.replace(condition.getProposition())))
      return std::nullopt;

    return condition;
  }

  // This intentionally does not handle extension conformances as that feature
  // is not stable yet.
  return std::nullopt;
}

/// Emits the diagnostic for a bare `@explicit_destroy` decorator, which now
/// requires a string message argument, with a note nudging users toward the
/// `Deinitable where ...` replacement. The caller marks the decl
/// erroneous and returns failure. Shared by `struct` and `trait` resolution.
static void emitExplicitDestroyRequiresArgError(SharedState &shared,
                                                SMLoc decoratorLoc,
                                                const ASTDecl &decl) {
  auto diag = shared.emitError(decoratorLoc)
              << "@explicit_destroy requires an argument: "
                 "`@explicit_destroy(\"...\")`";
  diag.attachNote(decl)
      << "Use `Deinitable where False` conformance to opt out of "
         "implicit deletion. `@explicit_destroy` is no longer required.";
}

/// Validates that an `@explicit_destroy(...)` call has exactly one string
/// literal argument, storing its value in `message`. Emits a diagnostic and
/// returns failure for the wrong-arity and non-string-literal forms. Shared by
/// `struct` and `trait` resolution.
static LogicalResult parseExplicitDestroyMessage(SharedState &shared,
                                                 CallNode *callNode,
                                                 std::string &message) {
  if (callNode->operands.size() != 1) {
    shared.emitError(callNode->getLoc())
        << "expected exactly one argument: `@explicit_destroy(\"...\")`";
    return failure();
  }
  auto *strExpr = dyn_cast<StringLiteralNode>(callNode->operands.front().expr);
  if (!strExpr) {
    shared.emitError(callNode->operands.front().expr->getLoc())
        << "expected a string literal argument: `@explicit_destroy(\"...\")`";
    return failure();
  }
  message = strExpr->getValue();
  return success();
}

/// structdef ::=
///   [decorators] "struct" identifier [param_signature]
///                ["(" conformance_list ")"] ("where" expression)* ":" suite
///
LogicalResult DeclResolver::resolveSignature(StructDeclOp structOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  // The struct signature is a self-contained scope where the input and result
  // parameters of the function are visible by all types.  We must use a
  // temporary declaration here (with an empty name) because we don't want
  // references to the function itself to resolve to a fully-resolved decl, but
  // we need a fully-resolved decl for incremental lookups within the scope to
  // work out.
  ASTDecl &sigDecl = addFullyResolvedDecl(structOp.getOperation(), StringAttr(),
                                          decl.getLoc(), decl.getParentDecl());

  ParsedParamList parsedParams;
  SMLoc identifierLoc;
  SmallVector<ParsedConformanceEntry> parsedConformances;
  SmallVector<ParsedTraitConstraint> parsedConstraints;

  if (p.parseToken(Token::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      parsedParams.parseParametersIfPresent(p, ArgListKind::kParamList) ||
      parseOptionalConformanceListSyntax(
          p, parsedConformances, sigDecl.getIndentation(),
          /*allowConformanceConstraints=*/true) ||
      parsedParams.parseTrailingConstraintsIfPresent(p) ||
      p.parseToken(Token::colon, "expected ':' in struct definition") ||
      decl.isErroneous())
    return failure();

  // Type-check parameters before resolving the conformance list so parameter
  // declarations are in scope for conformance-list constraint emission.
  std::optional<TypeCheckedParamList> paramSignatureOrError =
      TypeCheckedParamList::create(parsedParams, sigDecl);
  if (!paramSignatureOrError.has_value())
    return failure();
  TypeCheckedParamList &paramSignature = *paramSignatureOrError;

  DenseSet<TraitSymbolAttr> explicitTraits;
  {
    DenseSet<TraitSymbolAttr> immediateParents; // unused.
    if (resolveConformanceList(parsedConformances, sigDecl, decl, shared,
                               immediateParents, &parsedConstraints,
                               &explicitTraits))
      return failure();
  }

  paramSignature.emitBodyConstraints();

  // Register closure-parameter captures and their `eq(F.dtype, dtype)` where
  // clauses.
  OpBuilder structBuilder = decl.getDeclEndBuilder();
  SmallVector<ConstraintAttr> closureExternalRefConstraints =
      registerClosureParamCaptures(paramSignature.paramDeclAttrs, decl, shared,
                                   structBuilder);
  llvm::append_range(paramSignature.emittedBodyConstraints,
                     closureExternalRefConstraints);
  sigDecl.insertKnownAssumptions(closureExternalRefConstraints);

  // Look up traits the compiler unconditionally injects into every struct.
  // These lookups are reused below both for constraint building (to skip
  // propagated constraints) and for the actual injection into parentTraits,
  // so the trait names are stated only once here.
  ASTDecl *anyTypeDecl = shared.lookupBuiltinTrait("AnyType", decl.getLoc());
  ASTDecl *implicitDelDecl =
      shared.lookupBuiltinTrait("Deinitable", decl.getLoc());
  ASTDecl *movableDecl = shared.lookupBuiltinTrait("Movable", decl.getLoc());

  DenseSet<TraitSymbolAttr> compilerInjectedTraits;
  if (anyTypeDecl)
    compilerInjectedTraits.insert(
        TraitSymbolAttr::get(anyTypeDecl->getSymbolRef()));
  if (implicitDelDecl)
    compilerInjectedTraits.insert(
        TraitSymbolAttr::get(implicitDelDecl->getSymbolRef()));
  if (movableDecl)
    compilerInjectedTraits.insert(
        TraitSymbolAttr::get(movableDecl->getSymbolRef()));
  // Build the final constraint map from the parsed constraints.
  DenseMap<TraitSymbolAttr, ConstraintAttr> traitConstraints;
  if (failed(buildTraitConstraintsMap(parsedConstraints, explicitTraits,
                                      compilerInjectedTraits, traitConstraints,
                                      shared)))
    decl.setErroneous();

  auto paramsArrayAttr =
      ParamDeclArrayAttr::get(getContext(), paramSignature.paramDeclAttrs);
  auto sig = TypeSignatureType::remapToSignature(
      silenceErrors(getContext()), paramsArrayAttr,
      paramSignature.getParamListAttr());
  assert(sig && "could not remap signature");
  structOp.setParamsAttr(paramsArrayAttr);
  structOp.setSignature(sig);

  SmallVector<TraitSymbolAttr> traitConformed;
  if (auto *inheritedFrom = decl.getTraitConformanceLineage())
    for (auto [symbol, _] : *inheritedFrom)
      traitConformed.push_back(symbol);

  // Make every nominal struct type inherit from `AnyType`.
  if (anyTypeDecl)
    traitConformed.push_back(TraitSymbolAttr::get(anyTypeDecl->getSymbolRef()));

  // Make every nominal struct type inherit from `Deinitable` and
  // `Movable`. May be overridden / narrowed by explicit conditional `where`
  // conformance.
  if (implicitDelDecl)
    traitConformed.push_back(
        TraitSymbolAttr::get(implicitDelDecl->getSymbolRef()));
  if (movableDecl)
    traitConformed.push_back(TraitSymbolAttr::get(movableDecl->getSymbolRef()));

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(structOp));

  // Structs are memory-only unless they opt-in to being passed in registers.
  structOp.setConvention(TypeConvention::MemoryOnly);

  // Now that we have the basic struct set up, process signature decorators.
  Decorators(decl).applySignatureDecorators(
      decoratorExprs, [&](ExprNode *decorator) {
        return processStructSignatureDecorator(decorator, structOp, shared,
                                               sigDecl);
      });

  // Propagate signature errors and decls.
  decl.takeDecls(sigDecl);

  // Capture the location of the `@explicit_destroy` decorator for more
  // accurate diagnostics later.
  std::optional<std::tuple<std::string, SMLoc>> linearTypeErrorMsg;
  for (auto decoratorExpr : decoratorExprs) {
    if (auto *callNode = dyn_cast<CallNode>(decoratorExpr.first)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        if (declRef->spelling == "explicit_destroy") {
          std::string message;
          if (failed(parseExplicitDestroyMessage(shared, callNode, message)))
            return failure();
          linearTypeErrorMsg =
              std::make_tuple(std::move(message), callNode->getLoc());
        }
      }
    }
  }
  if (linearTypeErrorMsg && !std::get<0>(*linearTypeErrorMsg).empty()) {
    structOp.setLinearTypeErrorMsg(
        std::make_optional(llvm::StringRef(std::get<0>(*linearTypeErrorMsg))));
  }

  // Build canonical trait with constraints for conditional conformance.
  SmallVector<ConstraintAttr> constraintsArray =
      canonicalizeTraitSymbolsAndConstraints(shared, traitConformed,
                                             traitConstraints);
  structOp.setCanonicalTrait(
      TraitType::get(getContext(), traitConformed, constraintsArray));

  // Always generate SourceName for structs (even on non-debug builds).
  structOp.setSourceNameAttr(shared.getSourceName(structOp));

  if (std::optional<ConstraintAttr> deinitableConstraint =
          getConformanceCondition(decl, "Deinitable")) {
    if (isTriviallyTrueConstraint(*deinitableConstraint)) {
      // Unconditional Deinitable cannot be combined with
      // @explicit_destroy.
      if (linearTypeErrorMsg) {
        auto diag = shared.emitError(std::get<1>(*linearTypeErrorMsg))
                    << "@explicit_destroy is not valid on `struct` with "
                       "unconditional conformance to `Deinitable`";
        diag.attachNote(decl.getLoc())
            << "Add `Deinitable where False` conformance or "
               "remove `@explicit_destroy`";
        decl.setErroneous();
        return failure();
      }
    } else if (!structOp.getLinearTypeErrorMsg().has_value()) {
      structOp.setLinearTypeErrorMsg(
          "type '" + structOp.getDeclName().str() +
          "' does not conditionally conform to 'Deinitable' "
          "for these parameters");
    }
  } else if (!structOp.getLinearTypeErrorMsg().has_value()) {
    // Never Deinitable: the struct is linear.
    // Synthesize a default message since the user didn't provide one.
    structOp.setLinearTypeErrorMsg(
        "type '" + structOp.getDeclName().str() +
        "' does not conform to 'Deinitable' and must be explicitly "
        "destroyed");
  }

  // Make the decl as signature resolved before we check rp conformance.
  decl.resolvedness = DeclResolvedness::signature;

  // Only set the convention for unconditional RP conformance. Conditional RP
  // leaves the struct as MemoryOnly at declaration time; the parametric
  // isMemoryOnly bit resolves per-instantiation during lowering.
  if (std::optional<ConstraintAttr> rpConstraint =
          getConformanceCondition(decl, "RegisterPassable")) {
    if (isTriviallyTrueConstraint(*rpConstraint)) {
      structOp.setConvention(TypeConvention::RegisterPassable);
    } else {
      structOp.setRegisterPassableConstraintAttr(*rpConstraint);
    }
  }

  // TrivialRegisterPassable conforms to RegisterPassable, so should set this
  // after setting RegisterPassable.
  if (std::optional<ConstraintAttr> trpConstraint =
          getConformanceCondition(decl, "TrivialRegisterPassable")) {
    // Check for unsupported conditional conformance to TrivialRegisterPassable.
    if (!isTriviallyTrueConstraint(*trpConstraint)) {
      shared.emitError(trpConstraint->getLoc())
          << "conditional conformance to 'TrivialRegisterPassable' is "
             "not supported";
      decl.setErroneous();
      return failure();
    }
    structOp.setConvention(TypeConvention::RegisterPassableTrivial);
  }

  shared.notifyListenerOnStructDecl(decl, identifierLoc);
  return success();
}

/// Look up a special method impl for the specified `type` when there is exactly
/// one implementation (not overloaded).  This returns the method if successful,
/// and returns null if there is none.
static FnOp lookupSpecialInit(ASTDecl &structDecl,
                              SpecialFunctionKind specialKind) {
  auto &shared = structDecl.getShared();
  LookupResult inits =
      shared.lookupAndResolveDecl("__init__", structDecl.getLoc(), structDecl,
                                  /*searchParentScopes=*/false);

  for (ASTDecl *candidate : inits.getIfSuccess()) {
    FnOp func = dyn_cast_or_null<FnOp>(candidate->getIfOperation());
    if (!func || failed(shared.getDeclResolver().resolveSignature(
                     *candidate, candidate->getLoc())))
      continue;

    // Found it.
    if (func.getSpecialFunctionKind() == specialKind)
      return func;
  }
  return {};
}

namespace {
struct StructDecorators : public SharedStateUser {
  StructDecorators(StructDeclOp structOp, ASTDecl &structDecl,
                   DeclResolver &resolver)
      : SharedStateUser(resolver.shared), structOp(structOp),
        structDecl(structDecl) {}

  LogicalResult processBodyDecorator(ExprNode *decorator);

private:
  /// Process the @fieldwise_init body decorator on structs.
  void processFieldwiseInitDecorator(SMLoc decoratorLoc, bool isImplicit);

  StructDeclOp structOp;
  ASTDecl &structDecl;
};
} // namespace

/// Look at the initializers of the specified struct to see if there is already
/// a fieldwise init.  If so, return it, otherwise return null.
static FnOp findFieldwiseInit(ASTDecl &structDecl) {
  auto &shared = structDecl.getShared();

  LookupResult inits =
      shared.lookupAndResolveDecl("__init__", structDecl.getLoc(), structDecl,
                                  /*searchParentScopes=*/false);
  if (inits.isErroneous())
    return {};

  auto structOp = cast_or_null<StructDeclOp>(structDecl.getIfOperation());
  unsigned numFields = std::distance(structOp.getFieldDecls().begin(),
                                     structOp.getFieldDecls().end());
  for (ASTDecl *declaration : inits.getIfSuccess()) {
    auto func = dyn_cast_or_null<FnOp>(declaration->getIfOperation());
    if (!func)
      continue;
    auto signature = func.getFuncTypeGenerator();
    ArrayRef<Type> inputTypes = signature.getArguments();
    ArrayRef<ArgConvention> convs = signature.getArgConventions();
    // Ignore the result slot and error result.
    while (!convs.empty() && isResultSlot(convs.back())) {
      inputTypes = inputTypes.drop_back();
      convs = convs.drop_back();
    }
    // TODO: Handle default arguments.
    if (inputTypes.size() != numFields)
      continue;
    // Skip any kind of var-args.
    if (signature.getBody().getArgListAttrs().hasAnyVarArg())
      continue;

    bool isMatch = true;
    for (auto [type, conv, field] :
         llvm::zip(inputTypes, convs, structOp.getFieldDecls())) {
      // Strip the pointer type if present.
      ASTType argType = type;
      // Fieldwise initializers must have read/owned conventions. ref etc
      // are lit.ref's mechanically but these are invisible the to the caller.
      if (hasImplicitOrigin(conv)) {
        if (conv != ArgConvention::ImmMem && conv != ArgConvention::OwnedMem &&
            conv != ArgConvention::DeinitMem) {
          isMatch = false;
          break;
        }
        argType = ASTType(argType).getReferenceElementType();
      }

      if (!argType.isEqualCanon(field.getType())) {
        isMatch = false;
        break;
      }
    }
    if (isMatch)
      return func;
  }
  return {};
}

/// Process the @fieldwise_init body decorator on structs. 'isRequired'
/// indicates whether it is an error to already have a fieldwise init.
void StructDecorators::processFieldwiseInitDecorator(SMLoc decoratorLoc,
                                                     bool isImplicit) {
  // Don't add one if we already have one.
  if (FnOp init = findFieldwiseInit(structDecl)) {
    auto diag =
        emitError(decoratorLoc, "'")
        << cast_or_null<StructDeclOp>(structDecl.getIfOperation()).getSymName()
        << "' has an explicitly declared fieldwise initializer";
    diag.attachNote(init.getLoc()) << "initializer declared here";
    return;
  }

  // Generate the fieldwise init.
  StructEmitter structEmitter(structDecl);
  auto fn = structEmitter.synthesizeFieldwiseInit();
  if (!fn)
    return;

  // If "implicit", check for validity and set the bit.
  if (isImplicit) {
    auto fieldsRange =
        cast_or_null<StructDeclOp>(structDecl.getIfOperation()).getFieldDecls();
    if (std::distance(fieldsRange.begin(), fieldsRange.end()) != 1) {
      emitError(decoratorLoc,
                "@fieldwise_init(\"implicit\") is only valid on structs "
                "with a single field");
      return;
    }
    fn.setImplicitConversion(ImplicitConversionKind::Implicit);
  }
}

LogicalResult StructDecorators::processBodyDecorator(ExprNode *decorator) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "fieldwise_init") {
      processFieldwiseInitDecorator(decorator->getRangeStart(),
                                    /*implicit*/ false);
      return success();
    }
    if (declRef->spelling == "explicit_destroy") {
      emitExplicitDestroyRequiresArgError(shared, declRef->getLoc(),
                                          structDecl);
      structDecl.setErroneous();
      return failure();
    }
  }
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      if (declRef->spelling == "explicit_destroy")
        return success();

      if (declRef->spelling == "fieldwise_init") {
        if (callNode->operands.size() != 1 ||
            !isa<StringLiteralNode>(callNode->operands.front().expr) ||
            cast<StringLiteralNode>(callNode->operands.front().expr)
                    ->getValue() != "implicit")
          emitError(decorator->getRangeStart(),
                    "@fieldwise_init only allows an \"implicit\" argument");
        processFieldwiseInitDecorator(decorator->getRangeStart(),
                                      /*implicit*/ true);
        return success();
      }
    }
  }
  return failure();
}

/// Process the RegisterPassable decorator on structs.  This finalizes
/// semantic checks.
static void processRegisterPassableDecorator(
    StructDeclOp structOp, ASTDecl &structDecl,
    ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields,
    DeclResolver &resolver, TypeConvention structPassability) {

  bool isTrivial = structPassability == TypeConvention::RegisterPassableTrivial;
  for (auto [fieldOp, fieldDecl] : structFields) {
    ASTType fieldType = fieldOp.getType();

    // Register-passable structs may only contain register-passable stored
    // values.
    // TODO(traits): We need to type constrain mlirtype parameters to being
    // register-only types to support things like this correctly:
    //  struct P[T: mlirtype]:
    //    var storage : T

    // If the field is at least as register-passable as the container then
    // we're happy.
    if (fieldType.getRegisterPassability(fieldDecl->getLoc(), resolver.shared) <
        structPassability) {
      StringRef trivialPrefix;
      if (isTrivial) {
        trivialPrefix = "Trivial";
      }

      auto diag = resolver.emitError(structOp.getLoc())
                  << "all members of '" << trivialPrefix
                  << "RegisterPassable' struct must themselves be '"
                  << trivialPrefix << "RegisterPassable'";
      diag.attachNote(fieldDecl->getLoc())
          << fieldOp.getNameAttr() << " declared with type " << fieldType;

      // We cannot support IRGen'ing references to this type, since it will
      // break invariant about being register passable without being composed
      // of such types.
      fieldDecl->getParentDecl()->setErroneous();
      return;
    }
  }
}

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, Lexer &lexer,
                                      ASTDecl &structDecl) {

  // If the type lacks a __sp_fn__is_trivial member, synthesize it to
  // unresolved.
  auto synthesizeTrivialFlagIfNeeded = [&](StringRef spFnName) {
    std::string trivialDelTag = (spFnName + "is_trivial").str();
    if (!shared.typeHasMember(structDecl, trivialDelTag, structDecl.getLoc()))
      StructEmitter(structDecl).synthesizeUnresolvedAlias(trivialDelTag);
  };

  // Push the debug scope for this struct if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(structOp.getLocScope());

  // Parse the body of the struct, which will give us all the methods and
  // fields, but without resolving their signatures or bodies.
  if (ParserBase(shared, lexer).parseSuite(structDecl))
    return failure();

  // At this point, we have to mark the struct as body resolved... for very
  // unfortunate reasons. The issue is that we need nested declarations (e.g.
  // struct fields) to be able to do unqualified name lookups from within the
  // struct body:
  //
  //    struct S:
  //       var x : Int  # Must look up 'Int', it isn't a param of 'S'.
  //
  // To support this, we mark the struct as body resolved at this point, even
  // though we don't even know what all the decls are within it - we're about to
  // synthesize new members etc, which means it definitely isn't body resolved
  // here.  This is a phase ordering and a modeling problem with ASTDecl - we
  // could add a new resolvedness level for this (between signature and body
  // resolved indicating that we can name lookup through it?).
  structDecl.resolvedness = DeclResolvedness::body;

  // This collects all the resolved struct fields. Now that the body is
  // parsed we can check the declared fields for extra invariants.
  bool hasBadField = false;
  SmallVector<std::pair<StructFieldOp, ASTDecl *>> structFields;

  // Iterate over all the parsed decls.  in general these won't be signature
  // resolved, and we don't want to resolve functions.  We do need to resolve
  // struct fields signatures to understand their type.
  for (std::pair<StringAttr, TinyPtrVector<ASTDecl *>> decls :
       structDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls.second) {
      auto fieldOp = dyn_cast_or_null<StructFieldOp>(decl->getIfOperation());
      if (!fieldOp)
        continue;

      if (failed(resolveSignature(*decl, decl->getLoc()))) {
        hasBadField = true;
        continue;
      }
      structFields.push_back({fieldOp, decl});
    }
  }

  // Determine if there is an explicit conformance to Deinitable.
  if (std::optional<ConstraintAttr> deinitableConstraint =
          getConformanceCondition(structDecl, "Deinitable")) {
    // Synthesize an empty __deinit__ when the type conforms to
    // Deinitable
    // ([mojo-lang] Accept '__deinit__' as the canonical destructor spelling)
    // Deinitable but has no explicit destructor.
    if (!shared.typeHasMember(structDecl, "__deinit__", structDecl.getLoc()))
      (void)StructEmitter(structDecl)
          .synthesizeEmptyDtor(*deinitableConstraint);

    // If the structure conforms to "Deinitable", we populate the
    // trivial flag.
    synthesizeTrivialFlagIfNeeded("__del__");
  }

  // If the struct conforms to well-known traits but doesn't have explicit
  // implementations of the corresponding methods, add signatures for them.
  // These can all be synthesized without resolving the members.
  if (std::optional<ConstraintAttr> movableConstraint =
          getConformanceCondition(structDecl, "Movable")) {
    FnOp moveFn = lookupSpecialInit(structDecl, SpecialFunctionKind::kMoveCtor);
    if (!moveFn)
      moveFn = StructEmitter(structDecl)
                   .synthesizeEmptyMoveOrCopyInit(
                       /*isMove=*/true, *movableConstraint);
    if (moveFn) {
      synthesizeTrivialFlagIfNeeded("__move_ctor_");
      structOp.setMoveInitAttr(moveFn.getBoundSymbolRef(
          structDecl.getShared().getEvaluationContext()));
    }
  }
  if (std::optional<ConstraintAttr> copyableConstraint =
          getConformanceCondition(structDecl, "Copyable")) {
    FnOp copyFn = lookupSpecialInit(structDecl, SpecialFunctionKind::kCopyCtor);
    if (!copyFn)
      copyFn = StructEmitter(structDecl)
                   .synthesizeEmptyMoveOrCopyInit(
                       /*isMove=*/false, *copyableConstraint);
    if (copyFn) {
      synthesizeTrivialFlagIfNeeded("__copy_ctor_");
      structOp.setCopyInitAttr(copyFn.getBoundSymbolRef(
          structDecl.getShared().getEvaluationContext()));
    }
  }

  // If the struct is RegisterPassable, check invariants imposed by it before
  // checking other decorators.  This ensures that we reject invalid
  // RegisterPassable types before processing them.
  if (structOp.isRegisterPassable())
    processRegisterPassableDecorator(structOp, structDecl, structFields, *this,
                                     structOp.getConvention());

  // If any of the fields are bad, we do not process decorators since they
  // assume that the struct body if valid.
  if (hasBadField && !structDecl.getBodyDecorators().empty()) {
    structDecl.setErroneous();
    return failure();
  }

  // If there are any body decorators, resolve them now.
  StructDecorators structDecorators(structOp, structDecl, *this);
  Decorators(structDecl).applyBodyDecorators([&](ExprNode *decorator) {
    return structDecorators.processBodyDecorator(decorator);
  });

  if (structDecl.isErroneous())
    return success();

  // Finally, emit empty conformance tables.
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  TraitType canonicalTrait = structOp.getCanonicalTrait();
  ArrayRef<TraitSymbolAttr> symbols = canonicalTrait.getSymbols();
  // Constraints array is either empty (all unconditional) or parallel with
  // symbols.
  ArrayRef<ConstraintAttr> constraints = canonicalTrait.getConstraints();
  bool hasConstraints = !constraints.empty();
  SmallVector<TypedAttr> bodyPropositions = llvm::map_to_vector(
      structOp.getSignature().getParamListAttrs().getBodyConstraints(),
      [&](ConstraintAttr constraint) { return constraint.getProposition(); });
  IndexRefRemapper remapper(structOp.getParamsAttr());
  for (auto [i, parent] : llvm::enumerate(symbols)) {
    // A conformance whose condition the struct's own where-clause precludes is
    // an opt-out.
    if (hasConstraints && !bodyPropositions.empty() &&
        conformancePrecluded(bodyPropositions,
                             remapper.replace(constraints[i].getProposition())))
      continue;

    TraitSymbolAttr traitSymbol = parent;
    StringAttr name = traitSymbol.getFlattenedName();
    ASTDecl &parentDecl = getDeclForTypeSymbol(parent.getSymbol());
    TraitSymbolArrayAttr immediateParents =
        cast_or_null<TraitDeclOp>(parentDecl.getIfOperation())
            .getImmediateParentsAttr();
    // Use 3-arg builder for unconditional conformance, 4-arg for conditional.
    ConformanceOp witnessTable =
        hasConstraints
            ? ConformanceOp::create(b, traitSymbol, immediateParents,
                                    constraints[i])
            : ConformanceOp::create(b, traitSymbol, immediateParents);
    witnessTable.getBody().push_back(new Block());
    ASTDecl &decl = addDecl(witnessTable, structDecl.getLoc(), name,
                            &structDecl, {}, {}, -1);
    decl.resolvedness = DeclResolvedness::signature;
    // Conformances are always created as signature-resolved because there's no
    // less-resolved state for it (see CALROC for more).

    // Make sure the trait decl has been body resolved so we can check if
    // any methods provide implementations.
    if (failed(resolveBody(parentDecl, parentDecl.getLoc()))) {
      structDecl.setErroneous();
      return failure();
    }

    auto isDefaulted = [](auto nestedOp) {
      if constexpr (std::is_same_v<FnOp, decltype(nestedOp)>)
        return nestedOp.isDefaultedTraitFn();
      else if constexpr (std::is_same_v<AliasDeclOp, decltype(nestedOp)>)
        return nestedOp.isDefaultedAssociatedAlias();
      return false;
    };

    auto insertDefaultDecl = [&](auto newOp, StringAttr childName,
                                 ASTDecl *childDecl) -> LogicalResult {
      if (!isDefaulted(newOp))
        return success();

      if constexpr (std::is_same_v<AliasDeclOp, decltype(newOp)>) {
        // Since we do not allow alias to be overloaded, we can at
        // most insert one defaulted alias per name into the
        // struct (if there is no user-provided one already) or it
        // will lead to redefinition error. If there are multiple
        // defaulted alias with the same name, raise an error.
        for (ASTDecl *existingDecl :
             structDecl.lookupInCurrentScope(childName)) {
          Operation *op = existingDecl->getIfOperation();
          if (auto existingAlias = dyn_cast_or_null<AliasDeclOp>(op);
              existingAlias && existingAlias.isDefaultedAssociatedAlias()) {

            SymbolRefAttr currentTraitRef = getFullyResolvedSymbolRef(
                existingAlias->getParentOfType<TraitDeclOp>());

            StringRef currentTraitName = currentTraitRef.getLeafReference();
            StringRef otherTraitName =
                childDecl->getParentDecl()->getSymbolRef().getLeafReference();

            // There are multiple default associated aliases with
            // the same name. Raise an error.
            auto diag = shared.emitError(structDecl.getLoc())
                        << "trait member '"
                        << existingAlias.getDeclName().getValue()
                        << "' has conflicting default implementations in "
                        << otherTraitName << " and " << currentTraitName
                        << "; you must implement it manually";

            diag.attachNote(existingDecl->getLoc())
                << "original default implementation from trait "
                << currentTraitName << " here";

            diag.attachNote(newOp.getLoc())
                << "conflicting implementation from trait " << otherTraitName
                << " here";

            structDecl.setErroneous();
            return failure();
          }
          // This is a user provided alias, which shadows the
          // default value.
          return success();
        }
      }

      // Create a decl corresponding to the trait method we're inheriting.
      //
      // NOTE: this decl points to the lit.fn op in the actual trait so we now
      // have two decls pointing to the same lit.fn op.
      //
      // Ideally we'd create a stub lit.fn op in the struct with it's
      // inheritedFrom attribute pointing to the symbol ref attr of the trait
      // method, but since symbols are only created at signature resolution
      // time for lit.fn ops that's not an option (and attempting to signature
      // resolve trait methods at this point tends to cause cycles so is not
      // an option).
      //
      // Stashing the trait's lit.fn op here gives us an easy way to refer
      // back to it, signature resolving this struct's decl will actually
      // create the lit.fn op in the struct.decl op's body.
      auto &decl = shared.getDeclResolver().addDecl(
          newOp, childDecl->getLoc(), childName, &structDecl, LexerCursor(),
          LexerCursor(), -1);
      decl.resolvedness = DeclResolvedness::unparsed;
      return success();
    };

    StructEmitter emitter(structDecl);
    SmallVector<std::pair<FnOp, ASTDecl *>> nonEmptyTraitFns;
    for (auto &[childName, childDecls] : parentDecl.getDeclsInScope()) {
      for (ASTDecl *childDecl : childDecls) {
        if (auto nestedOp = childDecl->getIfOperation()) {
          LogicalResult result =
              TypeSwitch<Operation &, LogicalResult>(*nestedOp)
                  .Case<FnOp, AliasDeclOp>([&](auto nestedOp) {
                    return insertDefaultDecl(nestedOp, childName, childDecl);
                  })
                  .Default(LogicalResult::success());

          if (failed(result))
            return failure();
        }
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StructFieldDecl implementation
//===----------------------------------------------------------------------===//

/// struct_field_decl_stmt ::= "var" identifier ":" expression
/// TODO: Support default values?
///
LogicalResult DeclResolver::resolveSignature(StructFieldOp fieldOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  ASTType type;
  SMLoc identifierLoc;
  // Parse the type if present.
  p.consumeToken(); // let or var.
  if (p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      p.parseToken(Token::colon, "struct field declaration must have a type") ||
      parseType(p, type, *decl.getParentDecl(), decl.getIndentation(),
                /*allowUnbound=*/false))
    return failure();

  if (sugarIsa<TraitType>(type)) {
    emitError(decl.getLoc())
        << "struct fields do not support trait types; " << type
        << " is a trait, use a concrete type or "
           "compile-time generic";
    return failure();
  }

  fieldOp.setType(type);

  // Process field decorators syntactically to avoid recursive scope lookups
  // that can arise when using the IR emitter from within a struct's scope.
  // Currently only @doc_hidden and @__allow_legacy_any_origin_fields are
  // supported on struct fields.
  for (auto &[decorator, _] : decoratorExprs) {
    if (auto *declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "doc_hidden") {
        fieldOp.setIsDocHiddenAttr(UnitAttr::get(fieldOp.getContext()));
        continue;
      }
      if (declRef->spelling == "__allow_legacy_any_origin_fields") {
        fieldOp.setAllowLegacyAnyOriginAttr(
            UnitAttr::get(fieldOp.getContext()));
        continue;
      }
    }
    shared.emitError(decorator->getLoc(),
                     "decorators not supported on this statement")
        << SourceRange(decorator->getRangeStart(), decorator->getRangeEnd());
  }

  shared.notifyListenerOnStructFieldDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(StructFieldOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  // Perform additional semantic analysis of the fields. Erasing an AnyOrigin is
  // not allowed because Mojo won't know that the enclosing struct contains it,
  // and therefore won't do lifetime extension.
  //
  // Fields explicitly annotated with @__allow_legacy_any_origin_fields opt out
  // of this check: the author takes responsibility for keeping the referenced
  // memory alive, since the enclosing struct will not extend its lifetime.
  if (op.getAllowLegacyAnyOrigin())
    return success();

  ASTType fieldType = op.getType();
  for (TypedAttr origin :
       shared.cachedOriginFinder.findOriginsIn(getCanonicalType(fieldType))) {
    if (sugarIsa<AnyOriginAttr>(OriginType::stripMutCastAndRebind(origin))) {
      auto diag = emitError(decl.getLoc())
                  << "struct fields cannot expose AnyOrigin in their type; "
                  << op.getNameAttr().getValue() << " has type " << fieldType;
      diag.attachNote(decl)
          << "consider parameterizing enclosing struct with an Origin";
      diag.attachNote(op.getLoc()) << "alternatively, use UntrackedOrigin if "
                                      "lifetime is managed explicitly";
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Trait Decl implementation
//===----------------------------------------------------------------------===//
LogicalResult
DeclResolver::addSelfTypeToTrait(TraitDeclOp traitOp, ASTDecl &decl,
                                 SmallVector<TraitSymbolAttr> &parentTraits,
                                 DenseSet<TraitSymbolAttr> &immediateParents,
                                 ArrayRef<ParamDeclAttr> parameters,
                                 ArrayRef<PassingKind> passingKinds) {
  assert(parameters.size() == passingKinds.size());

  MLIRContext *ctx = getContext();
  SmallVector<TypedAttr> declRefs =
      llvm::map_to_vector(parameters, [&](ParamDeclAttr decl) -> TypedAttr {
        return ParamDeclRefAttr::get(decl);
      });

  // Can not use bindReference since we are constructing the trait.
  TraitSymbolAttr selfTrait =
      TraitSymbolAttr::get(getFullyResolvedSymbolRef(traitOp), declRefs);
  // Add the trait itself to its canonical trait list.
  parentTraits.push_back(selfTrait);
  TraitType canonTrait = getCanonicalTrait(parentTraits);
  traitOp.setCanonicalTrait(canonTrait);

  // Set up the parameter decl array by appending `_Self` to the end.
  SmallVector<ParamDeclAttr> traitParams(parameters);
  auto selfDecl = ParamDeclAttr::get(decl.mangleParamName("_Self"), canonTrait);
  traitParams.push_back(selfDecl);
  auto paramArray = ParamDeclArrayAttr::get(ctx, traitParams);

  // Set up the pog list too.
  SmallVector<PassingKind> traitParamsPassingKinds(passingKinds);
  traitParamsPassingKinds.push_back(PassingKind::Implicit);
  SmallVector<StringAttr> names = llvm::map_to_vector(
      parameters, [&](ParamDeclAttr decl) { return decl.getName(); });
  names.push_back(StringAttr::get(ctx, "_Self"));
  auto paramListAttr = PogListAttr::get(ctx, names, traitParamsPassingKinds);

  auto sig = TypeSignatureType::remapToSignature(silenceErrors(ctx), paramArray,
                                                 paramListAttr);
  if (!sig)
    return failure();
  traitOp.setParams(paramArray);
  traitOp.setSignature(sig);

  // Add the immediate parents to the trait.
  SmallVector<TraitSymbolAttr> immediateParentsVec(immediateParents.begin(),
                                                   immediateParents.end());
  sortAndDeduplicateTraitSymbols(immediateParentsVec);
  traitOp.setImmediateParents(
      TraitSymbolArrayAttr::get(ctx, immediateParentsVec));

  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(traitOp));
  return success();
}

LogicalResult DeclResolver::resolveSignature(TraitDeclOp traitOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  Decorators(decl).applySignatureDecorators(decoratorExprs,
                                            [&](ExprNode *decorator) {
                                              // No trait decorators supported.
                                              return failure();
                                            });

  // TODO(MOCO-1468): Pull this out into a common helper.
  ArrayRef<ExprNode *> bodyDecorators = decl.getBodyDecorators();
  for (auto decorator : bodyDecorators) {
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "explicit_destroy") {
        continue;
      }
    }
    if (auto callNode = dyn_cast<CallNode>(decorator)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        if (declRef->spelling == "explicit_destroy") {
          continue;
        }
      }
    }
    emitError(bodyDecorators.front()->getLoc(), "unrecognized body decorators ")
        << SourceRange(bodyDecorators.front()->getRangeStart(),
                       bodyDecorators.back()->getRangeEnd());
  }

  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_trait, "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by trait parser",
                        &identifierLoc))
    return failure();

  if (p.consumeIf(Token::l_square)) {
    // If the current token is on a new line, report the error on the end of
    // the previous line, this is probably where the punctuation was omitted.
    auto diagLoc = p.getTokenLocOrEndOfPreviousLineIfOnNewLine();
    // Report the error.
    emitError(diagLoc, "trait declarations do not support parameters; remove "
                       "the parameter list");
    return failure();
  }

  // Map from each symbol to the first symbol that explicitly inherits from it.
  DenseSet<TraitSymbolAttr> immediateParents;
  SmallVector<ParsedConformanceEntry> parsedConformances;
  if (parseOptionalConformanceListSyntax(
          p, parsedConformances, decl.getParentDecl()->getIndentation(),
          /*allowConformanceConstraints=*/false) ||
      resolveConformanceList(parsedConformances, *decl.getParentDecl(), decl,
                             shared, immediateParents))
    return failure();
  SmallVector<TraitSymbolAttr> parentTraits;
  bool definesClosure = traitOp.getDefinesClosure();
  if (auto *inheritedFrom = decl.getTraitConformanceLineage()) {
    for (auto [symbol, _] : *inheritedFrom) {
      parentTraits.push_back(symbol);
      if (definesClosure)
        continue;
      ASTDecl &type = getDeclForTypeSymbol(symbol.getSymbol());
      if (auto traitDecl =
              dyn_cast_if_present<TraitDeclOp>(type.getIfOperation()))
        if (traitDecl.getDefinesClosure())
          definesClosure = true;
    }
  }
  traitOp.setDefinesClosure(definesClosure);

  if (p.parseToken(Token::colon, "expected ':' in trait definition"))
    return failure();

  // Make every trait inherit from `AnyType`, except itself.
  if (parentTraits.empty() && traitOp.getSymName() != "AnyType") {
    if (ASTDecl *anyTypeDecl =
            shared.lookupBuiltinTrait("AnyType", decl.getLoc())) {
      parentTraits.push_back(TraitSymbolAttr::get(anyTypeDecl->getSymbolRef()));
      // No need to add AnyType to immediateParents, since it
      // has an empty requirements table.
    }
  }

  auto conformsToTrait = [&](StringRef traitName) {
    return traitOp.getSymName() == traitName ||
           llvm::any_of(parentTraits, [&](TraitSymbolAttr symbol) {
             return symbol.getSymbol().getLeafReference().getValue() ==
                    traitName;
           });
  };

  if (conformsToTrait("RegisterPassable"))
    traitOp.setConvention(TypeConvention::RegisterPassable);

  // TrivialRegisterPassable conforms to RegisterPassable, so should set this
  // after setting RegisterPassable.
  if (conformsToTrait("TrivialRegisterPassable"))
    traitOp.setConvention(TypeConvention::RegisterPassableTrivial);

  // Check if the trait conforms to Deinitable by checking the
  // parent traits list. We can't use doesNominalTypeConformTo or
  // lookupBuiltinTrait here because they would trigger signature resolution
  // and cause a cycle when resolving base traits like AnyType.
  bool conformsToDeinitable = conformsToTrait("Deinitable");

  // Parse @explicit_destroy decorator if present. It requires a string message
  // argument; the bare and empty-argument forms are errors, mirroring the
  // handling on `struct`.
  std::optional<std::string> linearTypeErrorMsg;
  for (auto decoratorExpr : decoratorExprs) {
    if (auto *declRefNode = dyn_cast<DeclRefNode>(decoratorExpr.first)) {
      if (declRefNode->spelling == "explicit_destroy") {
        emitExplicitDestroyRequiresArgError(shared, declRefNode->getLoc(),
                                            decl);
        decl.setErroneous();
        return failure();
      }
    } else if (auto *callNode = dyn_cast<CallNode>(decoratorExpr.first)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        if (declRef->spelling == "explicit_destroy") {
          std::string message;
          if (failed(parseExplicitDestroyMessage(shared, callNode, message)))
            return failure();
          linearTypeErrorMsg = std::move(message);
        }
      }
    }
  }

  // Validate @explicit_destroy usage and set error message for linear traits.
  if (conformsToDeinitable) {
    if (linearTypeErrorMsg) {
      shared.emitError(decl.getLoc(),
                       "@explicit_destroy cannot be used on a trait that "
                       "conforms to Deinitable");
      return failure();
    }
  } else {
    // Trait does not conform to Deinitable, so it is a linear type.
    // Set a default error message if @explicit_destroy wasn't used.
    if (!linearTypeErrorMsg) {
      linearTypeErrorMsg = "unhandled explicitly destroyed type '" +
                           traitOp.getDeclName().str() + "'";
    }
    traitOp.setLinearTypeErrorMsg(
        std::make_optional(llvm::StringRef(*linearTypeErrorMsg)));
  }

  // Insert the implicit trait parameter:
  // - _Self: a value of this trait type - the struct conforming to this trait.
  if (failed(addSelfTypeToTrait(traitOp, decl, parentTraits, immediateParents)))
    return failure();

  shared.notifyListenerOnTraitDecl(decl, identifierLoc);

  return success();
}

ParseResult DeclResolver::resolveBody(TraitDeclOp traitOp, Lexer &lexer,
                                      ASTDecl &traitDecl) {
  // TODO: Sink this to when the body is actually resolved.
  traitDecl.resolvedness = DeclResolvedness::body;

  // Push the debug scope for this trait if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(traitOp.getLocScope());

  if (ParserBase(shared, lexer).parseSuite(traitDecl))
    return failure();

  // Delegate error detection to the union type.
  return shared.getDeclResolver().resolve(
      *ASTType(traitOp.getCanonicalTrait()).getDecl(shared),
      DeclResolvedness::signature, traitDecl.getLoc());
}

/// Handles signature resolving a defaulted trait method inherited by a
/// conforming struct. The passed in ASTDecl is a child of that struct, while
/// the function op it contains belongs to the trait supplying the default.
LogicalResult
DeclResolver::resolveSyntheticSignature(FnOp inheritedFnOp,
                                        ASTDecl &childTraitFnDecl) {
  assert(isa<TraitDeclOp>(inheritedFnOp->getParentOp()) &&
         "Expected synthetic function decl's parent to be a trait");

  ASTDecl *structDecl = childTraitFnDecl.getParentDecl();
  assert(inheritedFnOp.isDefaultedTraitFn() &&
         isa_and_nonnull<StructDeclOp>(structDecl->getIfOperation()) &&
         "Expected trait -> struct default method inheritance");

  // TODO(MOCO-4712): this can be further simplified too.
  return resolveDefaultedOpFromTrait(*this, inheritedFnOp, structDecl);
}

/// Handles signature resolving an alias declaration inherited by a conforming
/// struct. The passed in ASTDecl is a child of that struct, while the
/// alias.decl op it contains belongs to the trait it is inherited from.
LogicalResult
DeclResolver::resolveSyntheticSignature(AliasDeclOp inheritedAliasOp,
                                        ASTDecl &childTraitAliasDecl) {
  auto getFnIsTrivialKind = [](StringRef trivialTagName) {
    // Matching by name is a bit gross, but we don't have general synthesized
    // decls so it should be robust.
    if (trivialTagName == "__del__is_trivial")
      return SpecialFunctionKind::kDeinit;
    if (trivialTagName == "__move_ctor_is_trivial")
      return SpecialFunctionKind::kMoveCtor;
    if (trivialTagName == "__copy_ctor_is_trivial")
      return SpecialFunctionKind::kCopyCtor;

    return SpecialFunctionKind::kNormal;
  };

  ASTDecl *structDecl = childTraitAliasDecl.getParentDecl();
  assert(isa_and_nonnull<StructDeclOp>(structDecl->getIfOperation()) &&
         "Expected the inherited alias to be resolved into a struct");

  // '__*__is_trivial' aliases are synthesized when a struct inherits from a
  // trait that declares them, and take their value from the struct's fields
  // rather than from the trait.
  SpecialFunctionKind spFn = getFnIsTrivialKind(inheritedAliasOp.getDeclName());
  if (spFn != SpecialFunctionKind::kNormal) {
    TypedAttr isTrivial =
        StructEmitter(*structDecl).populateSpecialFnIsTrivial(spFn);

    if (isTrivial) {
      inheritedAliasOp.setParamDeclAttr(ParamDeclAttr::get(
          inheritedAliasOp.getParamDecl().getName(), isTrivial.getType()));
      inheritedAliasOp.setValueAttr(isTrivial);
    } else {
      // Something went wrong while resolving fields.
      childTraitAliasDecl.setErroneous();
    }
    childTraitAliasDecl.resolvedness = DeclResolvedness::body;
    return success();
  }

  assert(inheritedAliasOp.isDefaultedAssociatedAlias() &&
         "Expected trait -> struct default associated alias");

  // TODO(MOCO-4712): this can be further simplified too.
  return resolveDefaultedOpFromTrait(*this, inheritedAliasOp, structDecl);
}

//===----------------------------------------------------------------------===//
// Extension implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(ExtensionDeclOp extensionDeclOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);

  SMLoc identifierLoc;
  StringAttr structNameAttr;
  if (p.parseToken(Token::kw___extension,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier(structNameAttr,
                        "internal error: checked by extension parser",
                        &identifierLoc))
    return failure();

  ASTDecl *parentDecl = decl.getParentDecl();
  assert(parentDecl && "Extension has no parent decl");

  // Look up all declarations with the same name. We need to look up ALL because
  // if we just look for the first, we'll find the extension itself, which has
  // the same name.
  // Note we aren't resolving the struct here, just finding it.
  // TODO(MOCO-522): Arcana references about how the extension has a unique
  // generated name, but is known to the parent ASTDecl (and therefore to the
  // rest of the world) as the struct's name.
  // TODO(MOCO-522): Consider requiring the extension to import the exact struct
  // rather than being able to import an intermediate extension. If we have that
  // restriction, then we can:
  // - Make the struct (e.g. Spaceship) known by two names:
  //   - "Spaceship" (as before)
  //   - "struct:Spaceship"
  //   The latter would let us do a single lookupAndResolveDecl here instead of
  //   the more expensive lookupAllDeclsWithName.
  // TODO(MOCO-522): Consider modifying the import system to automatically
  // import a target struct when we import an extension.
  // TODO(MOCO-522): Update the conflict test and simplify this to
  // lookupAndResolveDecl call, now that we've upgraded extension names.
  StringRef structName = structNameAttr.getValue();
  LookupAllResult lookupResult = shared.lookupAllDeclsWithName(
      structName, identifierLoc, *parentDecl, /*resolve=*/false);
  ArrayRef<ASTDecl *> foundDecls = lookupResult.getIfSuccess();
  // Find the actual struct declaration among all the found declarations.
  StructDeclOp structDeclOp = nullptr;
  ASTDecl *structAstDecl = nullptr;
  for (ASTDecl *decl : foundDecls) {
    if (auto foundStructDeclOp =
            dyn_cast_or_null<StructDeclOp>(decl->getIfOperation())) {
      structDeclOp = foundStructDeclOp;
      structAstDecl = decl;
      break;
    }
  }
  if (!structDeclOp) {
    return emitError(identifierLoc, "can't find a struct named '")
           << structName << "'";
  }
  if (failed(resolve(*structAstDecl, DeclResolvedness::signature,
                     identifierLoc))) {
    return failure();
  }

  SymbolRefAttr targetStructAttr = structAstDecl->getSymbolRef();
  extensionDeclOp.setTargetStructAttr(targetStructAttr);

  // This is an extension, but all the methods should think they're inside a
  // struct, so let's use 'computeSelfTypeForStruct' on the structDeclOp to
  // figure out the self type they can use.
  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(structDeclOp));

  // Use the parent scope to resolve the traits in the conformance list.
  // TODO(MOCO-522): This might need to change once we have parametric traits,
  // we might want to resolve from the extension's scope at that point.
  DenseSet<TraitSymbolAttr> immediateParents;
  SmallVector<ParsedConformanceEntry> parsedConformances;
  if (failed(parseOptionalConformanceListSyntax(
          p, parsedConformances, parentDecl->getIndentation(),
          /*allowConformanceConstraints=*/false)) ||
      failed(resolveConformanceList(parsedConformances, *parentDecl, decl,
                                    shared, immediateParents)))
    return failure();

  // Store the immediate parent traits in the extension
  SmallVector<TraitSymbolAttr> immediateParentsVec(immediateParents.begin(),
                                                   immediateParents.end());
  sortAndDeduplicateTraitSymbols(immediateParentsVec);
  extensionDeclOp.setImmediateParents(
      TraitSymbolArrayAttr::get(getContext(), immediateParentsVec));

  // Compute canonicalTrait for the extension (flattened trait hierarchy)
  if (!immediateParentsVec.empty()) {
    SmallVector<TraitSymbolAttr> canonicalSymbols(immediateParentsVec);
    TraitType canonicalTrait = getCanonicalTrait(canonicalSymbols);
    extensionDeclOp.setCanonicalTrait(canonicalTrait);
  }

  shared.notifyListenerOnTraitDecl(decl, identifierLoc);

  if (p.consumeIf(Token::l_square)) {
    // If the current token is on a new line, report the error on the end of
    // the previous line, this is probably where the punctuation was omitted.
    auto diagLoc = p.getTokenLocOrEndOfPreviousLineIfOnNewLine();
    // Report the error.
    auto diag = emitError(
        diagLoc, "cannot specify parameter declarations on extensions");

    diag.attachNote(structAstDecl->getLoc())
        << "extension already assumes these parameter declarations";
    return failure();
  }

  if (p.parseToken(Token::colon, "expected ':' in extension definition"))
    return failure();

  return success();
}

ParseResult DeclResolver::resolveBody(ExtensionDeclOp extensionDeclOp,
                                      Lexer &lexer, ASTDecl &extensionDecl) {
  SymbolRefAttr structSymbolRef = extensionDeclOp.getTargetStruct().value();

  ASTDecl &structAstDecl = getDeclForTypeSymbol(structSymbolRef);
  StructDeclOp structDeclOp =
      cast<StructDeclOp>(structAstDecl.getIfOperation());

  // Copy struct param decls into the extension, so extension methods and
  // aliases can reference the struct's param decls.
  for (ParamDeclAttr param : structDeclOp.getParams()) {
    StringAttr demangledName =
        StringAttr::get(getContext(), demangleParameterName(param.getName()));
    addFullyResolvedDecl(PValue(ParamDeclRefAttr::get(param)), demangledName,
                         extensionDecl.getLoc(), &extensionDecl);
  }
  // Set extension's parameters to match target struct. This is to make it so
  // the verifier can see the param-decls and be aware of them when it's
  // verifying their param-refs.
  // TODO(MOCO-522): Definitely need arcana docs here.
  // TODO(MOCO-522): Possibly-related problem: there might be parts of the
  // compiler that assume a method is contained by a *struct* specifically, and
  // that could cause problems when the method introduces its own param decls,
  // see https://github.com/modularml/modular/pull/69012.
  if (!structDeclOp.getParams().empty()) {
    extensionDeclOp.setParamsAttr(structDeclOp.getParamsAttr());
    extensionDeclOp.setSignature(structDeclOp.getSignature());
  }

  // Push the struct's debug scope for this extension if necessary so that
  // nested operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(structDeclOp.getLocScope());

  if (ParserBase(shared, lexer).parseSuite(extensionDecl))
    return failure();

  // Now check for conflicts; things in the extension shouldn't already be in
  // the struct, unless they're both methods because overloading is fine.
  if (extensionDecl.declsInScope && structAstDecl.declsInScope) {
    for (auto &[declName, extensionMemberDecls] : *extensionDecl.declsInScope) {
      ASTDecl *firstExtensionMemberDecl = extensionMemberDecls.front();

      // Skip parameter declarations, they are intentionally inherited from the
      // struct, the extension has an ASTDecl for every param declaration from
      // the target struct. If this is a ParamDeclRefAttr, skip it.
      if (CValue cval = firstExtensionMemberDecl->getIfIRValue())
        if (PValue pval = cval.getIfPValue())
          if (isa<ParamDeclRefAttr>(pval.get()))
            continue;

      bool isExtensionMethod =
          isa_and_nonnull<FnOp>(firstExtensionMemberDecl->getIfOperation());
      auto it = structAstDecl.declsInScope->find(declName);
      if (it == structAstDecl.declsInScope->end()) {
        // If there's nothing in the struct with this name, no conflict, done.
        continue;
      }
      ASTDecl *firstStructMemberDecl = it->second.front();
      bool isStructMethod =
          isa_and_nonnull<FnOp>(firstStructMemberDecl->getIfOperation());

      if (isExtensionMethod && isStructMethod) {
        // Method overloading is okay, done.
        continue;
      }

      // Show an error for each conflicting member in the extension decl, and
      // mark it erroneous.
      for (ASTDecl *extensionMemberDecl : extensionMemberDecls) {
        auto diag =
            emitError(extensionMemberDecl->getLoc(), "invalid redefinition of ")
            << declName;
        diag.attachNote(firstStructMemberDecl->getLoc())
            << "extension " << (isExtensionMethod ? "method" : "declaration")
            << " conflicts with struct "
            << (isStructMethod ? "method" : "declaration");
        extensionMemberDecl->setErroneous();
      }
      return failure();
    }
  }

  // Generate conformance tables for the extension's traits that the struct
  // doesn't already have. Use set difference to avoid duplicate conformances
  // between struct and extension. Extensions might have no canonical trait.
  //
  // TODO: Propagate constraints to ConformanceOp for extension conditional
  // conformance. Currently only struct decls support conditional conformance.
  if (extensionDeclOp.getCanonicalTrait()) {
    Block *extensionBody = extensionDeclOp.getBody();
    ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
        extensionDeclOp.getLoc(), extensionBody);

    // Get the target struct's existing conformances
    SmallVector<TraitSymbolAttr> structConformances(
        structDeclOp.getCanonicalTrait().getSymbols().begin(),
        structDeclOp.getCanonicalTrait().getSymbols().end());

    // Compute set difference: extension traits - struct traits
    SmallVector<TraitSymbolAttr> extensionOnlyTraits;
    for (TraitSymbolAttr extensionTrait :
         extensionDeclOp.getCanonicalTrait()->getSymbols()) {
      if (!llvm::is_contained(structConformances, extensionTrait))
        extensionOnlyTraits.push_back(extensionTrait);
    }

    // Create conformances only for extension-specific traits
    for (TraitSymbolAttr traitSymbol : extensionOnlyTraits) {
      StringAttr name = traitSymbol.getFlattenedName();
      ASTDecl &parentDecl = getDeclForTypeSymbol(traitSymbol.getSymbol());
      TraitSymbolArrayAttr immediateParents =
          cast_or_null<TraitDeclOp>(parentDecl.getIfOperation())
              .getImmediateParentsAttr();
      ConformanceOp witnessTable =
          ConformanceOp::create(b, traitSymbol, immediateParents);
      witnessTable.getBody().push_back(new Block());
      ASTDecl &decl = addDecl(witnessTable, extensionDecl.getLoc(), name,
                              &extensionDecl, {}, {}, -1);
      decl.resolvedness = DeclResolvedness::signature;
      // Conformances are always created as signature-resolved because there's
      // no less-resolved state for it (see CALROC for more).

      // Extension conformance verification follows the same pattern as structs
      // and is handled in verifyAndBuildConformance() during ConformanceOp body
      // resolution. The trait body will be resolved there.
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnresolvedImport Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(LIT::UnresolvedImportOp op,
                                           ASTDecl &decl, bool resolveTarget) {
  PackageOp packageOp = op->getParentOfType<PackageOp>();

  // Grab the location of the import name if present.
  SMLoc importNameLoc =
      shared.diags.convertLocToSMLoc(op.getImportNameLocAttr());
  if (!importNameLoc.isValid())
    importNameLoc = decl.getLoc();

  // Check that we are importing a specific decl within the module.
  auto declName = op.getDeclNameAttr();
  assert(declName && "Whole-package imports should be resolved by now");

  SMLoc declNameLoc = shared.diags.convertLocToSMLoc(op.getDeclNameLocAttr());
  if (!declNameLoc.isValid())
    declNameLoc = decl.getLoc();

  return getDeclResolver().importDeclFromModule(
      *decl.getParentDecl(), packageOp, op.getModulePathAttr(), declName,
      op.getImportNameAttr(), decl.getLoc(), declNameLoc, importNameLoc,
      resolveTarget);
}

//===----------------------------------------------------------------------===//
// Trait Composition Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(WitnessDecl *witness,
                                           ASTDecl &decl) {
  if (auto alias = dyn_cast<AliasDeclOp>(
          witness->getDecls().front()->getIfOperation())) {
    Type mergedType;
    ASTDecl *mergedTypeDecl = nullptr;

    DenseSet<TraitType> traitTypesToMerge;
    for (ASTDecl *curDecl : witness->getDecls()) {
      if (failed(resolve(*curDecl, DeclResolvedness::signature,
                         curDecl->getLoc())))
        return failure();

      auto alias = cast<AliasDeclOp>(curDecl->getIfOperation());
      Type aliasType = alias.getType();
      std::optional<ParameterEvaluator> evaluatorOpt =
          populateTraitBindingEvaluator(witness->traitSymbol, shared);
      if (evaluatorOpt) {
        assert(shared.isUniversalParametricClosureTrait(witness->traitSymbol));
        aliasType = cast<FnTypeGeneratorType>(evaluatorOpt->replace(aliasType));
      }

      if (!mergedType) {
        mergedType = aliasType;
        mergedTypeDecl = curDecl;
        if (isa<TraitType>(mergedType))
          traitTypesToMerge.insert(cast<TraitType>(mergedType));
        continue;
      }

      if (ASTType(aliasType).isEqualCanon(mergedType))
        continue;

      if (isa<TraitType>(mergedType) && isa<TraitType>(aliasType)) {
        traitTypesToMerge.insert(cast<TraitType>(aliasType));
        continue;
      }

      // We can only merge two trait types.
      auto diag =
          emitError(mergedTypeDecl->getLoc(), "invalid redefinition of '")
          << alias.getDeclName().getValue() << "': cannot convert "
          << ASTType(mergedType) << " to the other trait's member's type "
          << ASTType(aliasType);
      diag.attachNote(curDecl->getLoc())
          << "the other trait's member defined here";
      return failure();
    }
    if (traitTypesToMerge.size() > 1) {
      SmallVector<TraitSymbolAttr> mergedTraitSymbols;
      for (TraitType traitType : traitTypesToMerge)
        mergedTraitSymbols.append(traitType.getSymbols().begin(),
                                  traitType.getSymbols().end());
      sortAndDeduplicateTraitSymbols(mergedTraitSymbols);
      mergedType = TraitType::get(getContext(), mergedTraitSymbols);
    }

    // The merged entry is keyed by whichever trait the first alias came from;
    // all of them agree on the name, and conformance checking re-verifies the
    // merged bound against each.
    WitnessDecl::ResolvedType resolved;
    resolved.witnessName = alias.getDeclName();
    resolved.witnessType = mergedType;
    witness->storage = resolved;
    return success();
  }

  ASTDecl &onlyWitness = *witness->getDecls().front();
  if (failed(resolve(onlyWitness, DeclResolvedness::signature,
                     onlyWitness.getLoc())))
    return failure();

  // Propagate the disabled state. Signature resolution is what disables an
  // overridden method, so this has to be checked before touching the
  // operation, which disabling clears.
  if (onlyWitness.isDisabled()) {
    decl.markDisabled();
    return success();
  }

  auto fnOp = cast<FnOp>(onlyWitness.getIfOperation());

  WitnessDecl::ResolvedType resolved;
  resolved.witnessName = fnOp.getSymNameAttr();
  resolved.witnessType = fnOp.getFullSignature();
  resolved.implicitConversion = fnOp.getImplicitConversion();
  resolved.isStaticMethod = fnOp.getIsStatic();
  witness->storage = resolved;
  return success();
}

ParseResult DeclResolver::resolveBody(WitnessDecl *witness, ASTDecl &decl) {
  // We don't need to resolve the body of a witness, only need the type.
  assert(witness->getWitnessEntry().witnessType &&
         "Witness type should be resolved");
  decl.resolvedness = DeclResolvedness::body;
  return success();
}

ParseResult DeclResolver::resolveSignature(TraitType traitType,
                                           ASTDecl &traitDecl) {
  // There is no signature to resolve for a trait composition.
  return success();
}

ParseResult DeclResolver::resolveBody(TraitType traitType, ASTDecl &traitDecl) {
  // TODO: Sink this to when the body is actually resolved.
  traitDecl.resolvedness = DeclResolvedness::body;

  // Synthetic Trait Composition ASTDecl (STCASTD):
  // A trait composition decl is modeled as an "anonymous child trait" that
  // inherits from each trait in the composition. The differences are that:
  // - There is no physical TraitDeclOp in the IR for the trait composition.
  //   The ASTDecl's irValue is a TraitType (instead of a TraitDeclOp).
  DenseMap<StringAttr, std::pair<TraitSymbolAttr, SmallVector<ASTDecl *>>>
      aliases;

  // Collect all fn decl for error detection.
  SmallVector<std::pair<StringAttr, ASTDecl *>> fnDecls;

  for (TraitSymbolAttr symbol : traitType.getSymbols()) {
    // FIXME: we need to handle trait type with constraints correctly here...
    // The constraints need to be preserved on the incorporated decls.
    ASTDecl &parentDecl = getDeclForTypeSymbol(symbol.getSymbol());
    if (failed(resolveBody(parentDecl, traitDecl.getLoc())))
      return failure();

    // Inherit members from the parent.
    for (auto &[name, decls] : parentDecl.getDeclsInScope()) {
      for (ASTDecl *decl : decls) {
        Operation *memberOp = decl->getIfOperation();
        // Skip disabled member and inherited decls.
        if (decl->isDisabled())
          continue;
        if (!isa<AliasDeclOp, FnOp>(memberOp)) {
          // If the decl is not a function or alias, it is an error.
          return emitError(parentDecl.getLoc(), "unexpected decl in trait")
                     .attachNote(decl->getLoc())
                 << " declared here";
        }
        if (isa<FnOp>(memberOp)) {
          attachDeclToTraitCompositionDecl(&traitDecl, symbol,
                                           SmallVector<ASTDecl *>{decl}, name);
          fnDecls.push_back(std::make_pair(name, decl));
        } else {
          auto &witnessForAndDecls = aliases[name];
          witnessForAndDecls.second.push_back(decl);
          // It doesn't not really matter which trait symbol we picked for a
          // mergeable trait alias decls.
          witnessForAndDecls.first = symbol;
        }
      }
    }
  }

  for (auto [name, fnDecl] : fnDecls) {
    if (auto it = aliases.find(name); it != aliases.end()) {
      auto diag = shared.emitError(fnDecl->getLoc())
                  << "invalid redefinition of " << name;
      diag.attachNote(it->second.second.front()->getLoc())
          << "conflicting comptime alias declared here";
      traitDecl.setErroneous();
      return failure();
    }
  }

  for (auto &[name, decls] : aliases)
    attachDeclToTraitCompositionDecl(&traitDecl, decls.first,
                                     std::move(decls.second), name);

  return success();
}

//===----------------------------------------------------------------------===//
// WitnessTable Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(ConformanceOp op, ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

  if (ConstraintAttr constraint = op.getConstraintAttr())
    if (!isTriviallyTrueConstraint(constraint))
      decl.insertKnownAssumptions({constraint});

  // Verify conformance explicitly.
  std::optional<MojoInflightDiag> diag;

  // For extension conformances, we need to pass the target struct, not the
  // extension
  ASTDecl *declToVerify = getStructOrTargetStruct(*decl.getParentDecl(), *this);
  assert(declToVerify &&
         "ConformanceOps are only created inside structs or extensions");

  TraitSymbolAttr traitSymbol = op.getTraitSymbolAttr();
  assert(traitSymbol && "conformance bodies are resolved before `lower-lit` "
                        "erases the trait reference");
  if (failed(verifyAndBuildConformance(*declToVerify, traitSymbol, diag, op,
                                       decl)))
    return failure();

  return success();
}
