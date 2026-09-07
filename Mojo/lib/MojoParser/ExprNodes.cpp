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
// This file implements logic related to the expression nodes for the Lightning
// language.
//
//===----------------------------------------------------------------------===//

#include "ClosureEmitter.h"
#include "DLValues.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/MojoParser/SharedState.h"
#include "OverloadSet.h"
#include "ParamInf.h"
#include "ParserEvaluationContext.h"
#include "StabilityMarkers.h"

#include "ExprNodes.h"
#include "MojoUtils.h"
#include "Signatures.h"
#include "Traits.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/POPDialect/POPUtils.h"
#include "Mojo/ToolCommon/MLIROpFString.h"

#include "Support/AssertStream.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include <Support/LLVMForwardDecls.h>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

static bool usesClosurePipeline(FnOp fn) {
  return fn->getParentOfType<FnOp>() && !fn.isOptionalSymbol() &&
         !fn.getFuncTypeGenerator().isCapturing();
}

LogicalResult ExprNode::emitDestructuringPValue(PValue value,
                                                IREmitter &emitter) const {
  emitter.emitError(
      getLoc(), "invalid comptime declaration: expected an identifier or '_'");
  return failure();
}

ExprNode::ELVIITResult LValueCapableExprNode::emitLValueIfImplicitlyTyped(
    IREmitter &emitter, PatternDeclKind kind, bool /*hasInferrableRHS*/) const {
  ExprDest dest(EC_Assignment);
  dest.setPatternDeclKind(kind);
  return emitLCVIR(dest, emitter, true);
}

AnyValue LValueCapableExprNode::emitIR(ExprDest &dest,
                                       IREmitter &emitter) const {
  auto result = emitLCVIR(dest, emitter, false);
  if (result.isFailure())
    return {};
  assert(!result.getIfExprNode() &&
         "Non-speculative emitIR should not return an ExprNode*");
  return result.getIfValue();
}

/// Given a StringRef for an MLIR attribute, invoke the MLIR parser to resolve
/// it into an Attribute (which may not be a TypedAttr) and return it.  On
/// error, emit a diagnostic and return null.
static ErrorOr<Attribute> parseMLIRAttrFromString(StringRef name, SMLoc loc,
                                                  SharedState &shared) {
  Attribute result;
  std::string errorMsg;
  {
    // Capture errors thrown by parseAttribute and ignore them.
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(
        shared.getContext(), [&](Diagnostic &diag) { errorMsg = diag.str(); });

    // FIXME(#9621): Need to track the number of bytes read because we pass in
    // more than just the attribute we actually want to parse. This avoids
    // returning an error but is actually just masking the real problem.
    size_t bytesRead;
    result =
        mlir::parseAttribute(name, shared.getContext(), Type(), &bytesRead);
  }

  if (!result)
    return Error(errorMsg);

  return result;
}

static Attribute parseMLIRAttrFromStringWithError(StringRef name, SMLoc loc,
                                                  SharedState &shared) {
  auto result = parseMLIRAttrFromString(name, loc, shared);
  if (result.isError()) {
    auto diag = shared.emitError(loc, "invalid MLIR attribute: ")
                << result.takeError().get();
    diag.attachNote(loc) << "attempting to parse: '" << name << "'";
    return {};
  }
  return result.takeValue();
}

/// This implements __mlir_attr.x lookup, synthesizing a PValue for the
/// attribute on demand.
static PValue synthesizeMLIRAttrFromString(StringRef name, SMLoc loc,
                                           SharedState &shared,
                                           bool reportError = true) {
  Attribute attr;
  if (reportError) {
    attr = parseMLIRAttrFromStringWithError(name, loc, shared);
  } else {
    // If we were not able to build a string right now, we will wrap this
    // attribute with `#kgen.deferred` and let elaborator try to build attribute
    // again.
    auto res = parseMLIRAttrFromString(name, loc, shared);
    attr = res.isError() ? nullptr : res.takeValue();
  }

  if (!attr)
    return {};

  auto typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr) {
    // wrap non-typed attribute with #kgen.deferred attribute
    typedAttr = DeferredAttr::get(attr);
  }
  return PValue(typedAttr);
}

/// Given an __mlir_type[a,b,c] or __mlir_attr[a,b,c] usage, stringize the
/// subscript operands and return the result.  On error, emit an error and
/// return an empty string.
static std::string substituteMLIRMagic(const SubscriptNode &node,
                                       IREmitter &emitter) {
  std::string result;
  llvm::raw_string_ostream os(result);

  SMLoc loc = node.getLoc();
  for (const Operand &operand : node.operands) {
    ExprNode *expr = operand.expr;
    if (!operand.isPositional()) {
      emitter.emitError(loc, "only positional operands allowed in mlir magic")
          << expr->getRange();
      return {};
    }

    // If the index is an identifier, and if it is a backtick identifier, we
    // treat it as an interpolated literal string.  Otherwise we look it up as
    // an expression.  Rationale: this allows using strings attributes, which
    // could be useful someday, and keeps __mlir_attr.`thing` more consistent
    // with __mlir_attr[`thing`].
    if (auto *dre = dyn_cast<DeclRefNode>(expr))
      if (dre->spelling.data()[dre->spelling.size()] == '`') {
        os << dre->spelling;
        continue;
      }

    // As a very special hack, we treat a unary plus as a marker that the type
    // should not be printed when the attribute is stringized.
    // A unary tilde is a marker that the type should be printed first instead
    // of last.
    bool elideType = false;
    bool invertType = false;
    if (expr->kind == ExprNode::kPos) {
      elideType = true;
      expr = cast<UnaryOpNode>(expr)->subExpr;
    } else if (expr->kind == ExprNode::kInvert) {
      invertType = true;
      expr = cast<UnaryOpNode>(expr)->subExpr;
    }

    auto indexVal = emitter.emitExprPValue(expr, EC_MLIRMagic);
    if (!indexVal)
      return "";

    // Strip all sugar off of the value - many types and attributes will not
    // tolerate this.
    indexVal = getCanonicalAttr(indexVal.get());

    assert(indexVal.getType().mlirType ==
               getCanonicalType(indexVal.getType()) &&
           "indexVal type and value type must match");

    // If this is a wrapper for a type, print it as such.
    if (invertType) {
      os << ":" << ASTType(indexVal.getType()).mlirType << " ";
      indexVal.get().print(os, /*elideType=*/true);
    } else if (sugarIsa<TraitType>(indexVal.getType())) {
      // values of trait type are printed in a kgen compatible way, e.g.
      // "":!lit.trait<@std::@builtin::@stubs::@AnyType> someParamValue"
      if (!elideType)
        os << ":" << ASTType(indexVal.getType()).mlirType << " ";
      os << ASTType(indexVal).mlirType;
    } else if (sugarIsa<NonStructTypeType, TypeType, StructMetaType,
                        AnyTraitType>(indexVal.getType()))
      os << ASTType(indexVal).mlirType;
    else // Otherwise print it as an attribute.
      indexVal.get().print(os, elideType);
  }

  if (result.empty())
    emitter.emitError(loc, "mlir magic expanded to an empty string");
  return result;
}

static AttrCtorDeferredAttr
buildAttrCtorDeferredAttrFromMLIRAttr(const SubscriptNode &node,
                                      IREmitter &emitter) {
  SmallVector<TypedAttr> strings;
  SMLoc loc = node.getLoc();
  for (const Operand &operand : node.operands) {
    ExprNode *expr = operand.expr;
    if (!operand.isPositional()) {
      emitter.emitError(loc, "only positional operands allowed in mlir magic")
          << expr->getRange();
      return {};
    }

    // If the index is an identifier, and if it is a backtick identifier, we
    // treat it as an interpolated literal string.  Otherwise we look it up as
    // an expression.  Rationale: this allows using strings attributes, which
    // could be useful someday, and keeps __mlir_attr.`thing` more consistent
    // with __mlir_attr[`thing`].
    if (auto *dre = dyn_cast<DeclRefNode>(expr))
      if (dre->spelling.data()[dre->spelling.size()] == '`') {
        strings.push_back(StringAttr::get(emitter.getContext(), dre->spelling));
        continue;
      }

    bool elideType = false;
    if (expr->kind == ExprNode::kPos) {
      elideType = true;
      expr = cast<UnaryOpNode>(expr)->subExpr;
    }

    auto indexVal = emitter.emitExprPValue(expr, EC_MLIRMagic);
    if (!indexVal)
      return {};
    // Strip all sugar off of the value - many types and attributes will not
    // tolerate it.
    indexVal = getCanonicalAttr(indexVal.get());

    strings.push_back(ToStringDeferredAttr::get(indexVal.get(), elideType));
  }

  if (strings.empty())
    emitter.emitError(loc, "mlir magic expanded to an empty string");

  return AttrCtorDeferredAttr::get(strings);
}

#ifndef NDEBUG
/// Return the string representation of attr by concatenating held strings.
static std::string getStringRepresentation(AttrCtorDeferredAttr attr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto str : attr.getStrings()) {
    if (auto strAttr = sugarDynCast<StringAttr>(str)) {
      os << strAttr.str();
    } else if (auto toStrAttr = sugarDynCast<ToStringDeferredAttr>(str)) {
      auto val = cast<TypedAttr>(toStrAttr.getAttr());
      bool elideType = toStrAttr.getNeedElideType() != nullptr;

      if (sugarIsa<TraitType>(val.getType())) {
        // values of trait type are printed in a kgen compatible way, e.g.
        // "":!lit.trait<@std::@builtin::@stubs::@AnyType> someParamValue"
        if (!elideType)
          os << ":" << ASTType(val.getType()).mlirType << " ";
        os << ASTType(val).mlirType;
      } else if (sugarIsa<NonStructTypeType, TypeType, StructMetaType,
                          AnyTraitType>(val.getType()))
        os << ASTType(val).mlirType;
      else // Otherwise print it as an attribute.
        val.print(os, elideType);
    } else {
      llvm_unreachable("unexpected attribute type");
    }
  }

  return result;
}
#endif // NDEBUG

/// Report a warning to let user know that they should use __mlir_attr instead.
static void emitMLIRDeferredAttrToMLIRAttrWarning(SMLoc loc,
                                                  IREmitter &emitter) {
  emitter.emitWarning(loc)
      << "trivially constructable attribute. Use `__mlir_attr` "
         "instead.";
}

/// When a lookup in __mlir_op fails for a named field, this method tries to
/// resolve it.  On success, it lazily creates a resolved declaration.  On
/// failure, it bails out.
static PValue synthesizeMLIROpFromString(StringRef name, IREmitter &emitter) {
  auto *context = emitter.getContext();
  auto nameStr = StringAttr::get(context, name);

  auto result =
      UnboundMLIROperationAttr::get(nameStr, DictionaryAttr::get(context));
  return PValue(result);
}

/// Given an expression, try to resolve it into an Attribute that we can install
/// on this operation.
static Attribute getAttrFromExpr(StringRef name, ExprNode *node,
                                 IREmitter &emitter) {
  // Special case handling of __mlir_attr.`xxx` directly in this parser,
  // because we want to be able to install arbitrary attributes into an
  // operation's attribute list, and emitPValue only supports TypedAttrs.
  if (auto attrRef = dyn_cast<AttributeRefNode>(node)) {
    auto mlirAttr = dyn_cast<DeclRefNode>(attrRef->base);
    if (mlirAttr && mlirAttr->spelling == "__mlir_attr") {
      if (attrRef->spelling.empty())
        return {};
      return parseMLIRAttrFromStringWithError(
          attrRef->spelling, attrRef->getLoc(), emitter.shared);
    }
  }

  // Likewise, special case the __mlir_attr[a,b,c] syntax to support
  // attributes without types.
  if (auto subscript = dyn_cast<SubscriptNode>(node)) {
    auto mlirAttr = dyn_cast<DeclRefNode>(subscript->base);
    if (mlirAttr && mlirAttr->spelling == "__mlir_attr") {
      std::string result = substituteMLIRMagic(*subscript, emitter);
      if (result.empty())
        return {};
      return parseMLIRAttrFromStringWithError(result, subscript->getLoc(),
                                              emitter.shared);
    }
  }

  // Otherwise emit the value as an PValue.
  return emitter.emitExprPValue(node, EC_MLIRMagic);
}

/// Calculate the result of an __mlir_op.`thing`[attributes], applying the
/// attributes list to the operation specification.
static PValue
bindAttributesToMLIROperatorCall(const SubscriptNode &subscript,
                                 UnboundMLIROperationAttr unboundOp,
                                 IREmitter &emitter) {
  SMLoc loc = subscript.getLoc();
  MLIRContext *context = emitter.getContext();

  // Only allow applying attributes to something without them.
  if (!unboundOp.getAttrs().empty()) {
    emitter.emitError(loc, "operation already has attributes")
        << subscript.getRange();
    return {};
  }

  // Each element of the subscript must have a name identifier and a value as an
  // PValue.
  SmallVector<NamedAttribute> attrValues;
  for (const Operand &operand : subscript.operands) {
    ExprNode *valueExpr = operand.expr;
    if (!operand.isKeyword()) {
      MojoInflightDiag diag =
          emitter.emitError(loc, "attribute spec requires a keyword parameter");

      // Jump through some hoops to emit a hint about using the old syntax.
      if (auto *slice = dyn_cast<SliceLiteralNode>(valueExpr);
          slice && slice->upper && !slice->colon2Loc.isValid()) {
        if (auto *kwRef = dyn_cast_or_null<DeclRefNode>(slice->lower))
          diag << "; did you mean '" << kwRef->spelling << "=...'?"
               << FixIt::replaceToken(slice->colon1Loc, "=");
      }

      diag << valueExpr->getRange();
      return {};
    }

    auto value = getAttrFromExpr(operand.name, valueExpr, emitter);
    if (!value)
      return {};
    // Remove any sugar from attribute passed to mlir operators, the sugar may
    // break invariants of the op.
    value = getCanonicalAttr(value);
    attrValues.push_back({operand.name, value});
  }

  // Return it.
  auto attrs = DictionaryAttr::get(context, attrValues);
  return UnboundMLIROperationAttr::get(unboundOp.getName(), attrs);
}

/// Given a reference to an alias (either a direct reference from a DeclRefNode
/// "x" or an AttributeRefNode "x.y"), return the PValue for the result.
///
/// 'decl' is the alias declaration to resolve.
static PValue resolveAliasReference(AliasDeclOp decl, StringRef declName,
                                    ArrayRef<TypedAttr> paramValues,
                                    SMLoc refLoc, IREmitter &emitter) {

  // If the param is declared in a function, then just directly use it.
  Operation *parent = decl->getParentOp();
  while (parent && !isa<FileModuleOp, FnOp>(parent)) {

    // If this reference is within a trait then keep it symbolic since the
    // conforming type will ultimately provide the value to use.
    // TODO: What does it mean to write "SomeTrait.alias"?  This handles a
    // reference from within the body of the trait, but wouldn't a reference
    // from outside of it be an error?
    if (isa<TraitDeclOp>(parent)) {
      SyntheticNode synthNode(refLoc);
      SimpleLiteralNode selfNode(ExprNode::kSelfLiteral, refLoc);
      AttributeRefNode refNode(&selfNode, refLoc, declName);
      return emitter.emitExprPValue(&refNode, EC_AttributeRefBase);
    }

    // If this is in a struct, then the value may refer to parameters declared
    // on the struct, whose values come through 'bindings'.  Remap.
    if (auto structDecl = dyn_cast<StructDeclOp>(parent)) {
      assert(decl.getValueAttr() && "Struct's alias should have value");
      DenseMap<StringAttr, size_t> paramDeclIndices;
      for (auto [idx, paramDecl] : llvm::enumerate(structDecl.getParams()))
        paramDeclIndices[paramDecl.getName()] = idx;
      assert(paramDeclIndices.size() == paramValues.size() &&
             "incorrect number of type parameters for struct");

      // If the reference is to a member of the struct that has bindings, remap
      // them.  This allows things like `SomeType[a,b].someAlias` to substitute
      // the a/b values into the body of `someAlias`.  Note that the type may
      // be only partially bound: e.g. you might use `SomeType.someAlias` or
      // `SomeType[b=42].someAlias`. We want to support this so long as the
      // alias doesn't refer to an unbound parameter:
      //    struct X[a: Int, b: Int]:
      //        comptime a1 = 42
      //        comptime a2 = a+1
      //    def test():
      //        use(X.a1) # Ok
      //        use(X[1].a2) # Ok
      //        use(X.a2) # Error: 'a' needs to be bound
      // To check whether the value (incl. its type) of this alias depends on
      // unbound parameters, walk the attribute and check for ParamDeclRefAttrs
      // to values that are UnboundAttrs.
      TypedAttr aliasValue = decl.getValueAttr();
      WalkResult findUnboundParams = aliasValue.walk([&](ParamDeclRefAttr attr)
                                                         -> WalkResult {
        if (auto it = paramDeclIndices.find(attr.getName());
            it != paramDeclIndices.end() &&
            isa<UnboundAttr>(paramValues[it->second])) {
          emitter.shared.emitError(refLoc, "cannot access comptime member '")
              << declName << "' with unbound parameter '"
              << structDecl.getName() << "." << attr.getName().str() << "'";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (findUnboundParams.wasInterrupted())
        return {};

      ParameterEvaluator evaluator = emitter.shared.getParameterEvaluator(
          structDecl.getParams(), paramValues);
      return evaluator.getReboundAttribute(aliasValue);
    }

    if (auto extensionDecl = dyn_cast<ExtensionDeclOp>(parent)) {
      auto targetStructRef = extensionDecl.getTargetStruct();
      assert(targetStructRef && "Extension target struct not resolved somehow");
      // TODO(MOCO-522): Bring in the walking from the struct case, with a nice
      // test too.
      ASTDecl &structAstDecl =
          emitter.shared.declResolver->getDeclForTypeSymbol(*targetStructRef);
      StructDeclOp structDeclOp =
          cast<StructDeclOp>(structAstDecl.getIfOperation());
      assert(decl.getValueAttr() && "Extension's alias should have value");
      TypedAttr aliasValue = decl.getValueAttr();
      ParameterEvaluator evaluator = emitter.shared.getParameterEvaluator(
          structDeclOp.getParams(), paramValues);
      return evaluator.getReboundAttribute(aliasValue);
    }

    // Ignore 'if' and other control flow things.
    parent = parent->getParentOp();
  }

  // If this is at file or function scope, inline the value of the alias.
  return decl.getValueAttr();
}

//===----------------------------------------------------------------------===//
// ExprNode Implementation
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() = default;

/// Return the start or end of the source range.
llvm::SMLoc ExprNode::getRangeStart() const { return getRange().getStart(); }
llvm::SMLoc ExprNode::getRangeEnd() const { return getRange().getEnd(); }

/// Return the 'loc' for this node translated to an MLIR location.
Location ExprNode::getLocation(IREmitter &emitter) const {
  return emitter.translateLocation(getLoc());
}
/// Recursively dig through noop paren nodes (if present) to find what is
/// inside of them.
ExprNode *ExprNode::getWithoutParens() {
  if (auto *paren = dyn_cast<ParenNode>(this))
    return paren->subExpr->getWithoutParens();
  return this;
}

/// Return true if this is a TupleNode with no subexpressions.
bool ExprNode::isEmptyTuple() const {
  if (auto *tuple = dyn_cast<TupleNode>(this))
    return tuple->exprs.empty();
  return false;
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

AnyValue BoolLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  auto boolAttr = SIMDAttr::getScalarBool(emitter.getContext(), value);

  // Convert this to an instance of Bool. Bool must be in scope since it is
  // auto-imported.
  return emitter.emitBool({boolAttr, this}, dest);
}

LogicalResult
SimpleLiteralNode::emitDestructuringPValue(PValue value,
                                           IREmitter &emitter) const {
  // Simply discard the value.
  if (kind == kDiscardLiteral)
    return success();

  return ExprNode::emitDestructuringPValue(value, emitter);
}

AnyValue SimpleLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  if (kind == kNoneLiteral)
    return emitter.emitResult(emitter.shared.getNoneAttr(), this, dest);

  if (kind == kEllipsisLiteral)
    return emitter.emitResult(EllipsisAttr::get(emitter.getContext()), this,
                              dest);

  if (kind == kDiscardLiteral) {
    ASTType initializerType = dest.getIfInitializerType();
    // The discard pattern can only be used in case where we have an inferred
    // type for the lvalue or pvalue.
    if (!initializerType) {
      emitter.emitError(getLoc(), "cannot read from discard pattern '_'");
      return {};
    }

    AnyValue result =
        DLValue(RCRef<DiscardDLValue>::create(initializerType, this));
    return emitter.emitResult(result, this, dest);
  }

  assert(kind == kSelfLiteral && "Unknown simple literal kind");
  // Self resolves to the type of the enclosing structure type.
  ASTDecl *astDecl =
      emitter.declScope
          .getNearestDeclOfType<StructDeclOp, TraitDeclOp, ExtensionDeclOp>();
  if (!astDecl) {
    emitter.emitError(
        getLoc(),
        "'Self' type may only be used inside a struct, trait, or extension");
    return {};
  }

  ASTType selfType = astDecl->getTypeDeclSelf();
  if (!selfType) {
    // The user is attempting to use "Self" when the self type hasn't yet been
    // computed (e.g. inside a struct's parameter list). This is not allowed.
    emitter.emitError(getLoc(), "'Self' type is not available in this context");
    return {};
  }

  // Notify the listener that the Self is a reference of the parent
  // struct.
  emitter.shared.notifyListenerOnRef(astDecl, "Self", getRange());

  // Once we have the type in question we can just return its Self type as an
  // PValue.  This already includes bound parameters etc.
  assert(astDecl->resolvedness >= DeclResolvedness::signature);
  return emitter.emitResult(astDecl->getTypeDeclSelf(), this, dest);
}

// Handle generation of a constructor call to one of::
//     IntLiteral[value: __mlir_type.`!pop.int_literal`]
//     FloatLiteral[value: __mlir_type.`!pop.float_literal`]
//     StringLiteral[value: __mlir_type.`!kgen.string`]
static CValue handleIntFPStringLiteral(TypedAttr value, ASTType type,
                                       const ExprNode *expr, ExprDest &dest,
                                       IREmitter &emitter) {
  if (sugarIsa<TypeCheckErrorType>(type))
    return {}; // Sanity check the returned declaration.
  ASTDecl *decl = type.getDecl(emitter.shared);

  auto litStruct = dyn_cast_if_present<StructDeclOp>(decl->getIfOperation());
  if (!litStruct || litStruct.getParams().size() != 1 ||
      !sugarIsa<POP::IntLiteralType, POP::FloatLiteralType, StringType>(
          litStruct.getParams()[0].getType())) {
    emitter.emitError(expr->getLoc(), "malformed Literal type");
    return {};
  }

  // Bind the IntLiteral[value] parameter.
  type = litStruct.bindReference({value});

  // Create an instance of IntLiteral[val]()
  return emitter.emitConstructorCall(
      type, CallOperands(CallSyntax::kTypeCall, expr, std::move(dest)));
}

AnyValue IntLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  APInt value = Lexer::getIntegerLiteralValue(spelling);
  // Values produced are sometimes produced unsigned, so we must add an extra
  // sign bit.
  if (value.slt(APInt::getZero(value.getBitWidth())))
    value = value.zext(value.getBitWidth() + 1);
  auto attr = POP::IntLiteralAttr::get(emitter.getContext(), IPInt(value));

  // Look up the IntLiteral type.
  ASTType type = emitter.shared.lookupBuiltinType("IntLiteral",
                                                  emitter.declScope, getLoc());
  return handleIntFPStringLiteral(attr, type, this, dest, emitter);
}

AnyValue FloatLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  IPRational value = Lexer::getFloatLiteralValue(spelling);
  auto attr = POP::FloatLiteralAttr::get(emitter.getContext(), value);
  ASTType type = emitter.shared.lookupBuiltinType("FloatLiteral",
                                                  emitter.declScope, getLoc());
  return handleIntFPStringLiteral(attr, type, this, dest, emitter);
}

/// The value of a string is the concatenated value with escapes and quotes
/// removed.
std::string StringLiteralNode::getValue() const {
  std::string result;
  for (auto spelling : spellings)
    result += Lexer::getStringLiteralValue(spelling);
  return result;
}

/// Emit a constructor call for a string literal with the specified data, that
/// does not include enclosing quotes.  The specified expression specifies the
/// location but need not by a StringLiteral.
CValue StringLiteralNode::emitCtorCall(StringRef bytes, const ExprNode *expr,
                                       ExprDest &dest, IREmitter &emitter) {
  auto attr = StringAttr::get(bytes, StringType::get(emitter.getContext()));
  ASTType type = emitter.shared.lookupBuiltinType(
      "StringLiteral", emitter.declScope, expr->getLoc());
  return handleIntFPStringLiteral(attr, type, expr, dest, emitter);
}

AnyValue StringLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // Flatten multiple concat'd strings, remove """'s and "'s etc.
  std::string value = getValue();
  return emitCtorCall(value, this, dest, emitter);
}

/// Get the source range for a t-string, from start to the closing quote.
SourceRange TStringExprNode::getRange() const { return {startLoc, endLoc}; }

/// Emit IR for a t-string.
///
/// Lowers t"Hello, {name}!" to __make_tstring["Hello, {}"](name).
/// The format template is built at compile time with {} placeholders for each
/// interpolation, then passed as a comptime string parameter to
/// __make_tstring. The interpolated expressions become runtime arguments.
AnyValue TStringExprNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // Build the format template string and collect interpolated expressions.
  SmallString<128> formatTemplate;
  SmallVector<const ExprNode *> interpolatedExprs;

  for (const Part &part : parts) {
    std::visit(Overloaded{
                   [&](const LiteralPart &literal) {
                     formatTemplate += Lexer::getTStringLiteralValue(
                         literal.text, literal.isRaw);
                   },
                   [&](const InterpolationPart &interpolation) {
                     formatTemplate += "{}";
                     interpolatedExprs.push_back(interpolation.expr);
                   },
               },
               part);
  }

  // Emit IR for each interpolated expression.
  SmallVector<ASTExprAnd<AnyValue>> interpolatedValues;
  ExprDest exprDest(EC_OperatorOperandValue);
  for (const ExprNode *expr : interpolatedExprs) {
    AnyValue exprValue = emitter.emitExpr(expr, exprDest);
    if (!exprValue)
      return {};
    interpolatedValues.push_back({std::move(exprValue), expr});
  }

  // Bind the format string as a compile-time parameter.
  ParamBindings paramBindings(emitter.declScope, this);
  auto formatStringAttr = StringAttr::get(
      formatTemplate, KGEN::StringType::get(emitter.getContext()));
  paramBindings.add(this, formatStringAttr);

  // Look up __make_tstring and create an overload set.
  constexpr auto kMakeTStringFnName = "__make_tstring";
  auto fnDecls = emitter.shared.getBuiltinFunction(
      emitter.declScope, {"std", "format", "tstring"}, kMakeTStringFnName,
      getLoc());
  if (fnDecls.empty())
    return {};

  OverloadSet os(kMakeTStringFnName, fnDecls, std::move(paramBindings),
                 CallSyntax::kDirectCall);

  // Build the call operands from the interpolated values.
  CallOperands operands(CallSyntax::kDirectCall, this, std::move(dest));
  for (const auto &interpolated : interpolatedValues)
    operands.add(interpolated);
  return os.emitCall(std::move(operands), emitter);
}

bool Operand::isPositionalStringLiteral(StringRef str) const {
  if (isPositional())
    if (auto *strExpr = dyn_cast<StringLiteralNode>(expr))
      return strExpr->getValue() == str;
  return false;
}

AnyValue SyntheticNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  assert(irValue && "emitIR is undefined for synthetic nodes.");
  return emitter.emitResult(irValue, this, dest);
}

LogicalResult DeclRefNode::emitDestructuringPValue(PValue toBind,
                                                   IREmitter &emitter) const {
  assert(toBind && "No PValue provided when binding a parameter?");
  ASTDecl &paramASTDecl = emitter.getDeclResolver().addFullyResolvedDecl(
      toBind, spelling, getLoc(), &emitter.getDeclScope());
  emitter.shared.notifyListenerOnVariableDecl(paramASTDecl, getLoc());
  return success();
}

ExprNode::ELVIITResult DeclRefNode::emitLCVIR(ExprDest &dest,
                                              IREmitter &emitter,
                                              bool isSpeculative) const {
  return emitUnqualLookup(spelling, this, emitter.declScope, dest, emitter,
                          isSpeculative);
}

/// For a given ASTDecl, return the struct ASTDecl that it's about:
/// - If it's a struct's ASTDecl, return success with that ASTDecl.
/// - If it's an extension's ASTDecl, return success with the target struct's
/// ASTDecl.
/// - If it's anything else, return success with nullptr.
/// - If signature resolution fails, return failure.
static FailureOr<ASTDecl *> getTargetStructDecl(ASTDecl *referencedDecl,
                                                IREmitter &emitter) {
  if (isa_and_nonnull<StructDeclOp>(referencedDecl->getIfOperation())) {
    return referencedDecl;
  }
  if (auto extensionDeclOp =
          dyn_cast_or_null<ExtensionDeclOp>(referencedDecl->getIfOperation())) {
    // Signature resolve the extension so targetStruct gets populated.
    if (failed(emitter.shared.declResolver->resolve(
            *referencedDecl, DeclResolvedness::signature,
            referencedDecl->getLoc()))) {
      return failure();
    }

    SymbolRefAttr targetStructRef = extensionDeclOp.getTargetStruct().value();
    return &emitter.shared.declResolver->getDeclForTypeSymbol(targetStructRef);
  }
  return nullptr;
}

/// Find a unique in-scope spelling within edit distance of `spelling`, walking
/// parent scopes the same way unqualified lookup does. Returns empty when there
/// is no close match, or when several names tie for best distance.
static StringRef findClosestInScopeSpelling(StringRef spelling,
                                            ASTDecl &lookupScope) {
  // Short names are too ambiguous for useful suggestions (`x` vs `y`).
  if (spelling.size() < 3)
    return {};

  // Cap at 2 edits; scale down for shorter names (length 3 → at most 1).
  unsigned maxDistance = (spelling.size() + 2) / 3;
  if (maxDistance > 2)
    maxDistance = 2;
  StringRef best;
  unsigned bestDistance = maxDistance + 1;
  bool tied = false;

  for (ASTDecl *scope = &lookupScope; scope; scope = scope->getParentDecl()) {
    for (auto &[nameAttr, decls] : scope->getDeclsInScope()) {
      if (decls.empty())
        continue;
      StringRef candidate = nameAttr.getValue();
      if (candidate.empty() || candidate == spelling)
        continue;

      unsigned distance = spelling.edit_distance(
          candidate, /*AllowReplacements=*/true, bestDistance);
      if (distance > maxDistance)
        continue;
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
        tied = false;
      } else if (distance == bestDistance && candidate != best) {
        tied = true;
      }
    }
  }

  return tied ? StringRef() : best;
}

/// Utility function to diagnose unknown declarations, providing fixits or hints
/// whenever possible.
static ExprNode::ELVIITResult
diagnoseUnknownDeclaration(StringRef spelling, ASTDecl &lookupScope,
                           ArrayRef<ASTDecl *> failureDecls, IREmitter &emitter,
                           const ExprNode *expr) {
  auto loc = expr->getLoc();
  if (!failureDecls.empty()) {
    // Reject unqualified struct field references.
    if (auto fieldOp = dyn_cast_or_null<StructFieldOp>(
            failureDecls[0]->getIfOperation())) {
      emitter.emitError(loc, "cannot access instance field '")
          << spelling << "' directly; did you mean 'self.'?" << expr->getRange()
          << FixIt::insertBeforeToken(loc, "self.");
      return {};
    }

    // Reject unqualified struct/trait method/alias references.
    ASTDecl *parentDecl = failureDecls[0]->getParentDecl();
    if (parentDecl && isa_and_nonnull<StructDeclOp, TraitDeclOp>(
                          parentDecl->getIfOperation())) {
      const char *declKind = "";
      const char *replacement =
          isa<StructDeclOp>(parentDecl->getIfOperation()) ? "self." : "Self.";
      // References to static methods can always use capital Self.
      Operation *firstCandidate = failureDecls[0]->getIfOperation();
      if (auto fnCandidate = dyn_cast_or_null<FnOp>(firstCandidate)) {
        declKind = "method ";
        if (fnCandidate.getIsStatic())
          replacement = "Self.";
      } else if (isa_and_nonnull<AliasDeclOp>(firstCandidate)) {
        declKind = "comptime ";
        // Aliases are properties of the type, so `Self` is more idiomatic.
        // However `self.` is also supported.
        replacement = "Self.";
      }

      // References /from/ static methods can only use capital Self.
      if (auto curFn = dyn_cast_or_null<FnOp>(lookupScope.getIfOperation()))
        if (curFn.getIsStatic())
          replacement = "Self.";

      emitter.emitError(loc, "cannot access ")
          << declKind << "'" << spelling << "' directly; did you mean '"
          << replacement << "'?" << expr->getRange()
          << FixIt::insertBeforeToken(loc, replacement);
      return {};
    }
  }

  if (auto lookupOp = lookupScope.getIfOperation();
      lookupOp && isa<PackageOp, FileModuleOp>(lookupOp)) {
    auto diag = emitter.emitError(loc);
    if (auto packageOp = dyn_cast<PackageOp>(lookupOp))
      diag << "package '" << packageOp.getName();
    else {
      diag << "module '" << cast<FileModuleOp>(lookupOp).getName();
    }
    diag << "' has no declaration '" << spelling << "'" << expr->getRange();
    return {};
  }

  auto diag = emitter.emitError(loc, "use of unknown declaration '")
              << spelling << "'" << expr->getRange();
  if (spelling == "__type_of" || spelling == "_type_of" ||
      spelling == "typeof") {
    diag << "; did you mean 'type_of'?" << FixIt::replaceToken(loc, "type_of");
  } else if (spelling == "__origin_of" || spelling == "_origin_of" ||
             spelling == "originof") {
    diag << "; did you mean 'origin_of'?"
         << FixIt::replaceToken(loc, "origin_of");
  } else if (StringRef nearMiss =
                 findClosestInScopeSpelling(spelling, lookupScope);
             !nearMiss.empty()) {
    diag << "; did you mean '" << nearMiss << "'?"
         << FixIt::replaceToken(loc, nearMiss);
  } else {
    // The name is unambiguously part of a standard-library package that wasn't
    // imported; point the user at the missing import.
    std::optional<std::string> importPath =
        emitter.getDeclResolver().findUniqueStdlibImportFor(spelling);
    if (importPath)
      diag << "; did you mean to import it from '" << *importPath
           << "'? Add 'from " << *importPath << " import " << spelling << "'";
  }
  return {};
}

/// Merge original declared bounds with comptime-assumption refinements.
/// Returns the combined list of original + refined trait symbols, canonicalized
/// to include ancestor traits. Returns empty if no refinement applies (i.e.,
/// no new traits were added beyond the original bounds).
static SmallVector<TraitSymbolAttr>
mergeOriginalAndRefinedBounds(TraitType origBound, TraitType refinedBound,
                              SharedState &shared) {

  SmallVector<TraitSymbolAttr> traitSymbols(origBound.getSymbols());
  size_t origCount = traitSymbols.size();

  // Append refined symbols and fix up the ordering. Both sides are already
  // ancestor-closed, so the union is too — we only need sort+dedup, not a
  // second full canonicalize pass.
  llvm::append_range(traitSymbols, refinedBound.getSymbols());
  sortAndDeduplicateTraitSymbols(traitSymbols);

  // No new traits added — no refinement needed.
  if (traitSymbols.size() == origCount)
    return {};
  return traitSymbols;
}

/// Return the concrete trait bound carried by a type value's metatype, if one
/// is available.
static TraitType getTypeValueTraitBound(Type metaType) {
  if (auto traitType = dyn_cast<TraitType>(getCanonicalType(metaType)))
    return traitType;

  // A generic variadic element such as `Self.element_types[i]` has a parametric
  // metatype like `!kgen.param<:!lit.anytrait<Base> element_trait>`.  The
  // parameter value itself is not a concrete `TraitType`, but its metatype
  // still records the bound that every element in the list satisfies.
  if (auto paramType = dyn_cast<ParamType>(getCanonicalType(metaType))) {
    Type paramMetaType = getCanonicalType(paramType.getParam().getType());
    if (auto anyTrait = dyn_cast<AnyTraitType>(paramMetaType))
      return anyTrait.getTraitType();
  }

  return {};
}

/// Refine a non-reference ParamType using assumptions in `declScope`.
static Type maybeRefineParamType(Type varType, ParamType paramType,
                                 ASTDecl &declScope) {
  // Resolve the underlying ParamDeclRefAttr so assumption lookup matches
  // against the plain type parameter. `extractParamDeclRef` peels UpcastAttr
  // and TypeParamAttr; peel DowncastAttr here as well so that incoming
  // already-refined types (from prior def-time refinement or an explicit
  // user `downcast[T, Trait]`) still route through the main refinement path.
  // Falls back to the raw paramRef for associated-type paths.
  TypedAttr paramRef = paramType.getParam();
  TypedAttr unwrapped = paramRef;
  if (auto existingDowncast = dyn_cast<DowncastAttr>(unwrapped))
    unwrapped = existingDowncast.getInputTypeValue();
  ParamDeclRefAttr paramDeclRef = extractParamDeclRef(unwrapped);
  TypedAttr lookupAttr = paramDeclRef ? TypedAttr(paramDeclRef) : paramRef;

  SmallVector<ConstraintAttr> assumptions;
  declScope.getKnownAssumptionsIncludingParents(assumptions);
  if (assumptions.empty())
    return varType;

  TraitType refinedBound = getTraitBoundFromAssumptions(
      lookupAttr, declScope.getShared(), assumptions);
  if (!refinedBound)
    return varType;

  // Seed the merge from `paramRef` (not `lookupAttr`) so that any traits
  // already present on the incoming type — including an existing DowncastAttr
  // from prior refinement or an explicit user `downcast[T, Trait]` — are
  // preserved. `DowncastAttr::get` canonicalizes `downcast(downcast(x))` to
  // `downcast(x)`, so wrapping an already-refined paramRef stays flat.
  //
  // Preserve the bounds already visible on the incoming type value.  Plain type
  // parameters have a concrete trait metatype (e.g. `T: Base`).  Generic
  // variadic elements have a parametric metatype, but that metatype still
  // records the element trait bound (e.g. `element_trait: type_of(Base)`).
  //
  // If we cannot recover an original trait bound, conservatively decline to
  // refine.  Refining to only `refinedBound` would silently drop any original
  // constraints that downstream type checking expects to remain visible.
  TraitType origBound = getTypeValueTraitBound(paramRef.getType());
  if (!origBound)
    return varType;

  SmallVector<TraitSymbolAttr> traitSymbols = mergeOriginalAndRefinedBounds(
      origBound, refinedBound, declScope.getShared());
  if (traitSymbols.empty())
    return varType;

  TraitType traitType = TraitType::get(varType.getContext(), traitSymbols);
  TypedAttr downcast = DowncastAttr::get(traitType, paramRef);
  return ParamType::get(downcast);
}

Type LIT::maybeRefineTypeWithAssumptions(Type varType, ASTDecl &declScope) {
  // Handle RefType by unwrapping, refining the element, and re-wrapping.
  // This is needed for tuple elements that are references (e.g., from zip()).
  // Use sugarDynCast to see through type aliases like _ListIter[T].Element.
  if (auto refType = sugarDynCast<RefType>(varType)) {
    Type refinedElt =
        maybeRefineTypeWithAssumptions(refType.getElementType(), declScope);
    if (refinedElt != refType.getElementType())
      return refType.getWithElement(refinedElt);
    return varType;
  }

  // Check if the type is a parametric type.
  // Use sugarDynCast to see through type aliases.
  auto paramType = sugarDynCast<ParamType>(varType);
  if (!paramType)
    return varType;
  return maybeRefineParamType(varType, paramType, declScope);
}

TypedAttr LIT::maybeRefineTypeValueWithAssumptions(TypedAttr typeValue,
                                                   ASTDecl &declScope) {
  if (!typeValue || !LIT::isTypeExpr(typeValue))
    return typeValue;

  // `ASTType(TypedAttr)` turns the type value into the type it represents via
  // `ParamType::get(...)`. That builder canonicalizes: unresolved parameter
  // refs stay as `ParamType`, but constant type values can fold to the
  // represented MLIR type. Rebuild the refined type value through `PValue` so
  // both cases are handled uniformly.
  ASTType originalType(typeValue);
  Type refinedType =
      maybeRefineTypeWithAssumptions(originalType.mlirType, declScope);
  if (refinedType == originalType.mlirType)
    return typeValue;

  return PValue(refinedType).get();
}

Value LIT::maybeEmitRefinementRebind(Value value, ASTDecl &declScope,
                                     OpBuilder &builder, Location loc) {
  Type refinedType = maybeRefineTypeWithAssumptions(value.getType(), declScope);
  if (refinedType == value.getType())
    return value;
  return RebindOp::create(builder, loc, refinedType, value);
}

CValue LIT::maybeEmitRefinementRebind(ASTExprAnd<CValue> value,
                                      IREmitter &emitter) {
  if (!emitter.builder)
    return value.ir;
  Value ssa = value.ir.getMlirValue();
  if (!ssa)
    return value.ir;
  Type refinedType =
      maybeRefineTypeWithAssumptions(ssa.getType(), emitter.declScope);
  if (refinedType == ssa.getType())
    return value.ir;
  return emitter.rebindValue(value, refinedType);
}

/// Apply type refinement to a type expression used as an attribute receiver.
///
/// This creates a local shadow for this member lookup only: `T.member` can see
/// `downcast(T, Trait)` under a matching `conforms_to(T, Trait)` assumption,
/// while unrelated uses of `T` in the surrounding scope remain unchanged.
static void maybeRefineTypeAttributeBase(CValue &baseVal, ASTType &baseRVType,
                                         ASTDecl &declScope) {
  PValue basePValue = baseVal.getIfPValue();
  if (!basePValue)
    return;

  TypedAttr refinedTypeValue =
      maybeRefineTypeValueWithAssumptions(basePValue.get(), declScope);
  if (refinedTypeValue == basePValue.get())
    return;

  baseVal = CValue(PValue(refinedTypeValue));
  baseRVType = ASTType(refinedTypeValue);
}

/// Look up `spelling` across each container decl (the struct/trait plus any
/// visible extensions) and accumulate matching member decls into
/// `memberDecls`. Returns failure only on an erroneous lookup (already
/// diagnosed); a successful lookup that finds no members is reported by
/// `memberDecls` being empty on return.
static LogicalResult
lookupMembersInDecls(SmallVectorImpl<ASTDecl *> &memberDecls,
                     ArrayRef<ASTDecl *> structAndExtensionsDecls,
                     StringRef spelling, SMLoc loc, SharedState &shared) {
  memberDecls.clear();
  for (ASTDecl *containerDecl : structAndExtensionsDecls) {
    LookupResult lookup = shared.lookupAndResolveDecl(
        spelling, loc, *containerDecl, /*searchParentScopes=*/false);
    if (lookup.isErroneous())
      return failure(); // Error already diagnosed.
    if (!lookup.isSuccess())
      continue;

    // The param decls from structs are duplicated into the extension's
    // ASTDecl. This fact is inconvenient here, because we would see both the
    // ParamDeclRefAttr from the struct and the extension, and it would look
    // like a conflict further below. So filter out the param refs from
    // extensions to avoid duplicates.
    bool isExtension =
        isa_and_nonnull<ExtensionDeclOp>(containerDecl->getIfOperation());
    for (ASTDecl *decl : lookup.getIfSuccess()) {
      if (decl->isDisabled())
        continue;
      if (isExtension)
        if (!decl->getIfOperation())
          if (PValue cv = decl->getIfIRValue().getIfPValue())
            if (sugarIsa<ParamDeclRefAttr>(cv.get()))
              continue;
      memberDecls.push_back(decl);
    }
  }
  return success();
}

ExprNode::ELVIITResult
DeclRefNode::emitUnqualLookup(StringRef spelling, const ExprNode *expr,
                              ASTDecl &lookupScope, ExprDest &dest,
                              IREmitter &emitter, bool isSpeculative) {
  auto loc = expr->getLoc();

  // If this decl is part of a pattern that is wrapped in a var or ref, then
  // we need to emit a VarDeclOp for it in this scope.
  if (dest.getPatternDeclKind() != PatternDeclKind::kNone) {
    // Always return this node back on a speculative lookup, because we don't
    // have the contextual type available yet.
    if (isSpeculative)
      return expr;

    // Fail in cases like "var x = []" which is ambiguous.
    ASTType varType = dest.getIfInitializerType();
    if (!varType) {
      emitter.emitError(loc, "cannot declare '")
          << spelling << "' without a contextual type from its initializer"
          << expr->getRange();
      return {};
    }
    // We need to be in a function body to declare a variable.
    if (!emitter.builder || !lookupScope.getNearestDeclOfType<FnOp>()) {
      emitter.emitError(loc, "cannot declare '")
          << spelling << "' in this context" << expr->getRange();
      return {};
    }
    // This is either a var or ref pattern binding.
    bool isRefOrBind = dest.getPatternDeclKind() == PatternDeclKind::kRef ||
                       dest.getPatternDeclKind() == PatternDeclKind::kBind;
    VarDeclKind declKind;
    switch (dest.getPatternDeclKind()) {
    default:
      assert(0 && "unhandled pattern decl kind");
      [[fallthrough]];
    case PatternDeclKind::kRef:
      declKind = VarDeclKind::Ref;
      break;
    case PatternDeclKind::kVar:
      declKind = VarDeclKind::Var;
      break;
    case PatternDeclKind::kBind:
      declKind = VarDeclKind::Bind;
      break;
    }

    // Tuple unpacking is currently the only aggregate destructuring path. It
    // builds an aggregate LValue from element destinations before decomposing
    // the RHS tuple. Keep those element destinations unrefined so the aggregate
    // LHS remains store-compatible with `Tuple[T, ...]`; otherwise it would
    // expect `Tuple[downcast(T, Trait), ...]`. Bare `downcast(T, Trait)` and
    // `T` are convertible in targeted places, but we don't recursively treat
    // arbitrary `Struct[T]` and `Struct[downcast(T, Trait)]` as
    // interchangeable. The store path refines the individual bound elements
    // after initialization instead.
    // TODO: If we add other aggregate destructuring forms, model this as an
    // explicit aggregate-destructure destination property instead of keying off
    // `EC_TupleElement`.
    if (dest.getContext() != EC_TupleElement)
      varType = maybeRefineTypeWithAssumptions(varType, emitter.declScope);

    // For bind/ref patterns, wrap with placeholder origin (replaced at store
    // time). The ultimate reference will be a !lit.ref<!lit.ref<T>> but we
    // don't know the origin of the input value type yet.
    if (isRefOrBind)
      varType = RefType::getAnyOrigin(varType, true);

    VarDeclOp varDecl = emitter.emitVarDecl(
        spelling, varType, expr->getLocation(emitter), declKind);
    ASTDecl &varASTDecl = emitter.getDeclResolver().addFullyResolvedDecl(
        DeclIRValue(varDecl), varDecl.getNameAttr(), loc, &lookupScope);
    emitter.shared.notifyListenerOnVariableDecl(varASTDecl, loc);

    CValue result = isRefOrBind ? CValue(RLValue(varDecl)) : MLValue(varDecl);
    return emitter.emitCResult(result, expr, dest);
  }

  // Notify the listener of a normal decl reference lookup.
  emitter.shared.notifyListenerOnMemberLookup(lookupScope, loc,
                                              /*searchParentScopes=*/true);

  // Perform a lookup of the specified decl in the current lookupScope.
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      spelling, loc, lookupScope, /*searchParentScopes=*/true);

  // Only report this when we have a contextual type: we don't want to suggest
  // turning "x = {}" into "var x = {}", which is a different error.
  if (lookup.isFailure() && dest.getIfInitializerType()) {
    auto diag = emitter.emitError(loc);
    diag << "implicit declaration of '" << spelling << "' is not allowed; ";
    diag << "add 'var' to declare a new name";
    diag << FixIt::insertBeforeToken(loc, "var ");
    diag << expr->getRange();
    // An assignment outside any function is rejected before name resolution.
    if (ASTDecl *fn = lookupScope.getNearestDeclOfType<FnOp>())
      emitter.getDeclResolver().addErroneousDecl(spelling, loc, fn);
    return {};
  }

  ArrayRef<ASTDecl *> declsRef = lookup.getIfSuccess();

  // Backs `declsRef` when we resolve a deprecated intra-package reference
  // below. A LookupResult points into a scope's symbol table; the decls we
  // resolve here (a sibling module, or a name exposed by __init__) are not in
  // the empty package scope, so we need our own stable storage.
  SmallVector<ASTDecl *, 1> deprecatedDeclStorage;
  if (declsRef.empty()) {
    if (lookup.isErroneous())
      return {}; // Error already diagnosed.

    // If this was a speculative LValue lookup, don't fail, just wait for the
    // caller to try again with a type so we can synthesize a decl.
    if (isSpeculative)
      return expr;

    // Otherwise, diagnose the unknown declaration, and try to provide fixits.
    return diagnoseUnknownDeclaration(spelling, lookupScope,
                                      lookup.getIfFailure(), emitter, expr);
  }

  // We might modify this further below to filter out extensions and replace
  // them with the struct they're talking about.
  llvm::SmallVector<ASTDecl *, 16> decls(declsRef.begin(), declsRef.end());
  emitter.shared.notifyListenerOnRef(decls, spelling, expr);
  ASTDecl *firstExistingDecl = decls.front();

  // If the first result is a struct or an extension, their intention is
  // to use the struct.
  if (isa_and_nonnull<StructDeclOp>(firstExistingDecl->getIfOperation()) ||
      isa_and_nonnull<ExtensionDeclOp>(firstExistingDecl->getIfOperation())) {
    // We'll assume the first one is the one they intend to import, and any
    // conflicting later imports are the problem.
    ASTDecl *intendedReferencedDecl = firstExistingDecl;
    FailureOr<ASTDecl *> intendedTargetStructResult =
        getTargetStructDecl(intendedReferencedDecl, emitter);
    if (failed(intendedTargetStructResult))
      return {}; // Signature resolution failed, return early.
    ASTDecl *intendedTargetStructDecl = *intendedTargetStructResult;
    // Should be impossible because of the isa_and_nonnull checks above.
    assert(intendedTargetStructDecl && "no target struct");

    for (ssize_t i = decls.size() - 1; i >= 0; i--) {
      ASTDecl *otherReferencedDecl = decls[i];
      FailureOr<ASTDecl *> otherTargetStructResult =
          getTargetStructDecl(otherReferencedDecl, emitter);
      if (failed(otherTargetStructResult))
        return {}; // Signature resolution failed, return early.
      ASTDecl *otherTargetStructDecl = *otherTargetStructResult;
      if (!otherTargetStructDecl) {
        // Skip non-struct/extension declarations (where getTargetStructDecl
        // returns null). This can happen when lookup returns mixed declaration
        // types (e.g., a struct and a function with the same name).
        continue;
      }
      // We've now established that the otherTargetStructDecl is a struct.
      // Let's check if it's referring to the same struct they intended.
      // This shouldn't happen. If it could, tests ALCFRRS and/or ALCFTSVOE
      // would trigger this.
      // TODO(MOCO-522): Add another test in the cross-import extensions PR
      // to see if a renaming import might trigger this
      assert(intendedTargetStructDecl == otherTargetStructDecl &&
             "Ambiguous lookup for two structs/extensions");
    }

    decls = {intendedTargetStructDecl};
    // continue on
  }

  // Functions form an address, and may be overloaded.
  if (auto firstCandidate =
          dyn_cast_or_null<FnOp>(decls[0]->getIfOperation())) {
    // Form an overload set value with all the candidates.
    auto result = OverloadSetUValue::create(
        spelling, decls, ParamBindings(emitter.getDeclScope(), expr),
        CallSyntax::kDirectCall);
    return emitter.emitResult(result, expr, dest);
  }

  assert(decls.size() == 1 && "Only functions may be overloaded");
  ASTDecl *decl = decls[0];

  // Check for deprecation and stability warnings on the referenced decl.
  // Overloaded declarations like functions can't be handled here. They are
  // handled when overload sets are resolved in CallEmission.cpp.
  checkDeclUsageWarnings(*decl, loc, emitter.getDeclScope(), emitter.shared,
                         expr->getRange(), CallSyntax::kDirectCall, {});

  // Aliases form a PValue.
  if (auto param = dyn_cast_or_null<AliasDeclOp>(decl->getIfOperation())) {
    PValue result =
        resolveAliasReference(param, spelling, /*bindings=*/{}, loc, emitter);

    // Maintain alias sugar, e.g. print "UInt8" as UInt8 instead of SIMD[..].
    // We don't maintain sugar for aliases whose name start with _, because that
    // is assumed to be an internal implementation detail. This also covers
    // important things like _mlir_type.
    if (result && !isInternalName(param.getParamDecl().getName().strref())) {
      auto sugared = ParamDeclRefAttr::get(param.getParamDecl());
      // Some param refs (eg to unqualified parameters of structs) are already
      // fully sugared.
      if (result.get() != sugared)
        result = SugarAttr::getAlias(sugared, result);
    }

    return emitter.emitCResult(result.get(), expr, dest);
  }

  // If this is a type declaration, return it as a type.
  if (auto structOp = dyn_cast_or_null<StructDeclOp>(decl->getIfOperation()))
    return emitter.emitCResult(structOp.bindReference(), expr, dest);
  if (auto traitOp = dyn_cast_or_null<TraitDeclOp>(decl->getIfOperation()))
    return emitter.emitCResult(traitOp.getCanonicalTrait(), expr, dest);

  // If this is an ImportOp, form a module reference pointing to the ImportOp's
  // own symbol. The ImportOp gates child access via allowedChildren.
  if (isa_and_nonnull<ImportOp>(decl->getIfOperation())) {
    PValue result(ModuleAttr::get(LIT::ModuleType::get(decl->getSymbolRef())));
    return emitter.emitCResult(result, expr, dest);
  }

  // If this is a module or package declaration, form a module reference.
  // Cross-package access goes through an ImportOp (handled above); this is
  // reached for a sibling module resolved via the deprecated intra-package
  // fallback above (a bare sibling reference with no explicit import).
  if (isa_and_nonnull<FileModuleOp, PackageOp>(decl->getIfOperation())) {
    PValue result(ModuleAttr::get(LIT::ModuleType::get(decl->getSymbolRef())));
    return emitter.emitCResult(result, expr, dest);
  }

  // Error if this is a direct reference to a parameter declared on an enclosing
  // struct.  It should use Self.param.
  if (auto pvalue = decl->getIfIRValue().getIfPValue()) {
    if (auto declRef = dyn_cast<ParamDeclRefAttr>(pvalue.get())) {
      auto [parentDecl, paramDecls, paramIdx] =
          decl->lookupParamReference(declRef);
      // If the StructDecl does not have a self type, that means it is still
      // being signature resolved (and the ASTDecl we're looking at is the
      // temporary one made for signature emission). In this case, we allow
      // unqualified access to parameters.
      if (parentDecl && isa<StructDeclOp>(parentDecl->getIfOperation()) &&
          parentDecl->getTypeDeclSelf()) {
        auto diag =
            emitter.emitError(loc, "unqualified access to struct parameter '")
            << spelling << "'; use 'Self." << spelling << "' instead";
        diag << FixIt::replaceToken(loc, "Self." + spelling);

        for (auto paramDecl : parentDecl->lookupInCurrentScope(spelling)) {
          diag.attachNote(paramDecl->getLoc())
              << "parameter '" << spelling << "' declared here";
        }
      }
    }
    return emitter.emitCResult(pvalue, expr, dest);
  }

  // If this is a capture we need to emit a capture value.
  ASTDecl *declRef = nullptr;
  if (decls.size() == 1 && !isa_and_nonnull<FnOp>(decls[0]->getIfOperation())) {
    assert(decls.size() == 1 && "Only functions may be overloaded");
    declRef = decls[0];
  }

  // Find the nearest escaping closure, if there is one.
  ASTDecl *nearestEscapingFnOrNone =
      declRef ? lookupScope.getNearestDeclOfType<FnOp>() : nullptr;
  bool needsCapture = false;
  if (nearestEscapingFnOrNone) {
    needsCapture = true;
    for (ASTDecl *parentOfDeclOfRef = declRef->getParentDecl();
         parentOfDeclOfRef;
         parentOfDeclOfRef = parentOfDeclOfRef->getParentDecl()) {
      if (parentOfDeclOfRef == nearestEscapingFnOrNone)
        needsCapture = false;
    };
  }
  if (needsCapture) {
    FnOp parent = cast<FnOp>(nearestEscapingFnOrNone->getIfOperation());
    if (usesClosurePipeline(parent)) {
      if (!emitter.shared.captureInstanceExistsInScope(*nearestEscapingFnOrNone,
                                                       spelling)) {
        CaptureConvention defaultConvention =
            emitter.shared.defaultCaptureConventionInScope(
                *nearestEscapingFnOrNone);
        if (defaultConvention != CaptureConvention::kConventionUnspecified) {
          decl = ClosureEmitter::addCaptureValue(emitter.shared,
                                                 *nearestEscapingFnOrNone,
                                                 spelling, expr->getLoc());
          if (!decl)
            return {};
        } else if (emitter.builder) {
          // There is no default set nor this capture was registered already.
          // This is an error.
          emitter.shared.emitError(
              expr->getLoc(),
              "Could not infer capture convention of the captured value ")
              << spelling;
          return {};
        }
      }
      // do not attempt to re-register the capture under other closure types.
      needsCapture = false;
    }
  }

  // Narrow the decl to a CValue.
  CValue value;
  if (auto var = dyn_cast_or_null<VarDeclOp>(decl->getIfOperation())) {
    // Normal 'var' declarations are MLValues, but 'ref' declarations hold the
    // reference as its value and need to be loaded.
    if (var.getKind() == VarDeclKind::Bound) {
      value = MBValue(var); // Resolved "bind" values are immutable.
    } else if (var.getKind() != VarDeclKind::Ref) {
      value = MLValue(var);
    } else {
      if (!emitter.builder) {
        emitter.emitErrorForDynamicValueInParameter(expr);
        return {};
      }
      auto ref =
          RefLoadOp::create(*emitter.builder, expr->getLocation(emitter), var);
      value = CValue::getMValueForRef(ref);
    }
  } else if (auto cv = decl->getIfIRValue()) {
    value = cv;
  } else {
    emitter.emitError(loc, "use of declaration '")
        << spelling << "' as a value isn't supported yet" << expr->getRange();
    return {};
  }

  // Apply use-time type refinement. Variables declared in an outer scope may
  // gain additional conforms_to constraints when referenced in an inner scope
  // (e.g., inside comptime if/comptime assert blocks). This emits a
  // kgen.rebind with DowncastAttr when the current scope has additional
  // trait constraints beyond what was applied at declaration time.
  value = maybeEmitRefinementRebind({value, expr}, emitter);

  // If the declaration is a type check error, its error has already been
  // diagnosed. Squelch any downstream issues.
  if (sugarIsa<TypeCheckErrorType>(value.getRValueType()))
    return {};

  // If this is a reference to a value from an outer function scope, record
  // the capture.
  if (needsCapture) {
    emitter.shared.addCaptureToScope(
        *nearestEscapingFnOrNone, declRef,
        Capture(value, CaptureConvention::kConventionRead, spelling));
  }

  return emitter.emitCResult(value, expr, dest);
}

/// This uses the MLIR parser to turn the specified MLIR type name into an MLIR
/// type.
static ASTType parseMLIRType(StringRef name, const ExprNode *node,
                             SharedState &shared) {
  Type result;
  std::vector<Diagnostic> typeDiagnostics;
  {
    // Capture errors thrown by parseType.
    auto diagHandler = [&](Diagnostic &diag) {
      typeDiagnostics.push_back(std::move(diag));
    };
    mlir::ScopedDiagnosticHandler handler(shared.getContext(),
                                          std::move(diagHandler));
    result = mlir::parseType(name, shared.getContext());
  }
  if (!result) {
    MojoInflightDiag diagnostic =
        shared.emitError(node->getLoc(), "invalid MLIR type: ")
        << name << node->getRange();
    for (Diagnostic &diag : typeDiagnostics) {
      std::string str;
      llvm::raw_string_ostream os(str);
      diag.print(os);
      diagnostic.attachNote(node->getLoc()) << "MLIR error: " << str;
    }
  }
  return result;
}

/// Emit a reference to a stored field with a base that is known not to be a
/// dynamic lvalue.
CValue AttributeRefNode::emitStoredFieldRef(ASTExprAnd<CValue> base,
                                            StructFieldOp fieldOp,
                                            const ExprNode *expr,
                                            ExprDest &dest,
                                            IREmitter &emitter) {
  assert(!base.ir.getIfDLValue() &&
         "Dynamic lvalues should already be handled");
  auto mlirLoc = expr->getLocation(emitter);

  // Keep things in the parameter expression domain if we can.
  if (PValue baseMV = base.ir.getIfPValue()) {
    // StructExtract expects the value to be of struct type, so strip away any
    // sugar if present.
    baseMV = ParamOperatorAttr::getRebind(
        baseMV.get(), SugarAttr::strip(baseMV.get().getType()));

    auto extractVal = LIT::StructExtractAttr::get(baseMV.get(), fieldOp);
    return emitter.emitCResult(PValue(extractVal), expr, dest);
  }

  // Okay, handle dynamic field references.
  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(expr);
    return {};
  }

  // If we got a trivial SSA value, extract the field and return as SBValue.
  if (base.ir.isSValue() &&
      base.ir.getRValueType().isTrivial(expr->getLoc(), emitter.shared)) {
    Value baseVal = base.ir.getSValueRegister();
    // StructExtract expects the value to be of struct type, so strip away any
    // sugar if present.
    baseVal = emitter.emitRebindOpIfNeeded(
        baseVal, SugarAttr::strip(baseVal.getType()), expr->getLoc());
    auto structType = cast<StructType>(baseVal.getType());
    auto fieldType = fieldOp.getReboundType(
        structType, &emitter.shared.getEvaluationContext());
    auto extractVal = StructExtractOp::create(
        *emitter.builder, mlirLoc, fieldType, baseVal, fieldOp.getNameAttr());
    return emitter.emitCResult(SBValue(extractVal), expr, dest);
  }

  // Otherwise emit this to memory, and work with that.
  MBValue baseBVal = emitter.emitMBValue(base, dest.getContext());
  if (!baseBVal)
    return {};

  // RefStructGEROp requires the base to be a StructType: rebind away sugar.
  if (!isa<LIT::StructType>(baseBVal.getRValueType())) {
    auto baseRefType = cast<RefType>(baseBVal.getType());
    auto newEltType = SugarAttr::strip(baseBVal.getRValueType());
    baseBVal = emitter.emitRebindOpIfNeeded(
        baseBVal, baseRefType.getWithElement(newEltType), expr->getLoc());
  }

  auto fieldRefType =
      RefStructGEROp::getFieldType(cast<RefType>(baseBVal.getType()), fieldOp,
                                   &emitter.shared.getEvaluationContext());
  Value fieldRef = RefStructGEROp::create(
      *emitter.builder, mlirLoc, fieldRefType, fieldOp.getNameAttr(), baseBVal);

  // Apply type refinement to struct field accesses whose element type is a
  // parametric type with comptime assumptions (e.g., Self.T in a struct
  // method with `where conforms_to(Self.T, Trait)`).
  fieldRef = maybeEmitRefinementRebind(fieldRef, emitter.declScope,
                                       *emitter.builder, mlirLoc);

  // Result kind depends on the input kind.
  CValue result;
  if (base.ir.getIfMLValue())
    result = MLValue(fieldRef);
  else if (base.ir.getIfMBPValue())
    result = MBPValue(fieldRef);
  else
    result = MBValue(fieldRef);
  return emitter.emitCResult(result, expr, dest);
}

/// Emit an operand as an expression, handling the _ and other cases. This
/// typically will return a PValue but can also return a UValue when the operand
/// is (eg) an overload set or init list expression.
static AnyValue emitSingleParamExpr(const Operand &operand,
                                    IREmitter &srcEmitter,
                                    ExprContext context) {
  // Evaluate foo() as a param call not dynamic call if in a dynamic context.
  auto emitter = srcEmitter.getParamEmitter(EC_TypeParamValue);
  if (operand.expr->kind == ExprNode::kDiscardLiteral) {
    if (operand.unpackStyle == ArgUnpackStyle::kStarStar) {
      srcEmitter.shared.emitError(operand.expr->getLoc())
          << "'**_' not supported on parameter list, using '...' instead"
          << operand.expr->getRange();
      return {};
    }
    // Handle `_` syntax as an unbound parameter.
    return UnboundAttr::get(UnresolvedType::get(emitter.getContext()));
  }

  // Handle unpacked variadics in the form of `*x`.
  if (operand.expr->kind == ExprNode::kUnpack) {
    auto *unpackExpr = cast<UnaryOpNode>(operand.expr);

    if (unpackExpr->subExpr->kind == ExprNode::kDiscardLiteral) {
      srcEmitter.shared.emitError(operand.expr->getLoc())
          << "'*_' not supported on parameter list, using '...' instead"
          << operand.expr->getRange();
      return {};
    }

    // Only variadic-typed parameters can be unpacked. Emit the subexpression.
    // `expectedType` isn't needed here, because a variadic value would have
    // already had its element type disambiguated.
    PValue value = emitter.emitExprPValue(unpackExpr->subExpr, context);
    if (!value)
      return {};
    return UnpackedAttr::get(value, /*kwOnly=*/false, value.getType());
  }

  return emitter.emitExpr(operand.expr, context);
}

static LogicalResult parseParameterBindings(
    ArrayRef<Operand> operands, IREmitter &emitter,
    ParamBindings &paramBindings,
    ParamBindings::BindingKind initialBindingKind = ParamBindings::kStandard) {
  // When parsing a user-defined binding, we always start with the most
  // conservative binding kind.
  paramBindings.bindingKind = initialBindingKind;

  bool seenEllipsis = false;
  for (auto op : operands) {
    auto value = emitSingleParamExpr(op, emitter, EC_TypeParamValue);
    if (!value)
      return failure();

    if (!op.name && seenEllipsis) {
      // Regardless of the pog that the binding will be applied to, we can only
      // allow passing by keyword after `...`. Otherwise, it leads to ambiguity.
      return emitter.emitError(op.expr->getLoc())
             << "parameter after `...` must be passed by keyword"
             << op.expr->getRange();
    }
    if (isa_and_nonnull<EllipsisAttr>(value.getIfPValue().get())) {
      paramBindings.relaxBindingKindTo(ParamBindings::kWithEllipsis);
      seenEllipsis = true;
    } else {
      paramBindings.add(op.expr, value, op.name);
    }
  }

  return success();
}

/// Given a value of type type, substitute parameters into the type, producing
/// a more concrete type.  This syntax is `SomeType[1, 4, Int]`.
static PValue substituteParametersIntoUserDefinedType(
    PValue typeValue, const ExprNode *expr, ArrayRef<Operand> operands,
    SMLoc lhsLoc, SMLoc rhsLoc, IREmitter &emitter,
    ParamBindings::BindingKind initialBindingKind) {
  auto metaType = sugarCast<StructMetaType>(typeValue.getType());
  ASTDecl *typeDecl = ASTType(typeValue).getDecl(emitter.shared);

  // Notify the listener on the parameter binding.
  emitter.shared.notifyListenerOnParameterBinding(typeDecl, rhsLoc, operands);

  // Build up a ParamBindings set to validate and check the bindings.
  TypeSignatureType sig = metaType.getSignature();

  ParamBindings paramBindings(emitter.getDeclScope(), expr);
  if (failed(parseParameterBindings(operands, emitter, paramBindings,
                                    initialBindingKind)))
    return {};

  // Check the bindings.
  // Now we see a `[]`, it must be a concrete binding.
  ParamInf inference(paramBindings, sig.getParamTypes(),
                     sig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, typeDecl,
                     /*discardError=*/false,
                     /*deferredTypingContext=*/emitter.deferredTypingContext);

  VerifiedParamBindings bindings = inference.inferForStruct();
  if (!bindings)
    return {};

  // Ok, we succeeded at reparameterizing the type.
  return bindings.specializeGenerator(typeValue);
}

/// Bind parameter operands to a callable parameter.
static PValue
bindToGeneratorValue(PValue callable, GeneratorType sig, const ExprNode *expr,
                     ArrayRef<Operand> operands, IREmitter &emitter,
                     const SourceRange &range,
                     ParamBindings::BindingKind initialBindingKind) {
  // Build up a ParamBindings set to validate and check the bindings.
  ParamBindings paramBindings(emitter.getDeclScope(), expr);
  if (failed(parseParameterBindings(operands, emitter, paramBindings,
                                    initialBindingKind)))
    return {};

  // Check the bindings.
  // FIXME: The error messages are bad for partial binding, because the
  // diagnostic emitter points to the original struct definition.
  ParamInf inference(paramBindings, sig.getInputParamTypes(),
                     sig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, nullptr,
                     /*discardError=*/false,
                     /*deferredTypingContext=*/emitter.deferredTypingContext);
  VerifiedParamBindings newBindings = inference.inferForStruct();
  if (!newBindings)
    return {};

  // Applying arguments to a parametric alias whose right-hand side is a closure
  // type must substitute the arguments into the closure trait's captured
  // aliases and yield a freshly specialized closure trait.
  if (ASTDecl *moduleDecl =
          emitter.getDeclScope().getNearestDeclOfType<FileModuleOp>()) {
    if (TraitType specialized =
            emitter.shared.getClosureEmitter().getSpecializedClosureTrait(
                sig, newBindings.getValues(), *moduleDecl, expr->getLoc())) {
      return PValue(
          TypeParamAttr::get(specialized, AnyTraitType::get(specialized)));
    }
  }

  return newBindings.specializeGenerator(callable);
}

/// Given a base value, emit access to a base value element using either a
/// reference-producing-method or getter-and-setter-methods using the provided
/// operands.
///
/// This prefers the reference method if present.  If not, and if a getter is
/// present on the base type but a setter is not, this method immediately emits
/// a getter call.
///
/// Otherwise, it returns a SubscriptDLValue for later materializing calls to
/// the getter or setter as appropriate. When doing this it  takes ownership of
/// the operands because it might move them to a SubscriptDLValue, if emitted.
AnyValue emitGetterSetterAccess(const ExprNode *node, ASTExprAnd<CValue> base,
                                ArrayRef<Operand> exprOperands, ExprDest &dest,
                                IREmitter &emitter) {
  ASTType baseType = base.ir.getRValueType();

  // This is either a SubscriptNode for x[i,j] or a AttributeRefNode for x.name.
  bool isSubscript = isa<SubscriptNode>(node);

  // Subscripts on types are ill-formed
  if (isSubscript && LIT::isMetaType(baseType)) {
    emitter.emitError(node->getLoc())
        << "types are not subscriptable" << node->getRange();
    return {};
  }

  CallSyntax syntax =
      isSubscript ? CallSyntax::kSubscript : CallSyntax::kAttribute;

  auto lookupError = [&] {
    auto diagType = baseType;
    // Complain about "SomeType" in 'SomeType.foo' not 'AnyStruct[SomeType]'.
    if (auto meta = sugarDynCast<MetaType>(diagType)) {
      diagType = meta.getType();
    } else if (auto generator = sugarDynCast<GeneratorType>(diagType)) {
      if (auto mt = dyn_cast<MetaType>(generator.getBody())) {
        diagType = ASTType(mt.getType())
                       .getWithUnknownParametersReplaced(emitter.shared);
      }
    }

    auto diag = emitter.emitError(node->getLoc())
                << diagType << base.expr->getRange();

    if (isSubscript) {
      diag << " is not subscriptable, it does not implement the "
              "`__getitem__`/`__setitem__` methods";
    } else {
      auto *expr = exprOperands[0].expr->getWithoutParens();
      diag << " value has no attribute '"
           << cast<StringLiteralNode>(expr)->getValue() << "'";
    }
  };

  // As an extension to Python, we allow types to define getattr/getitem where
  // the indices are interpreted as parameters instead of arguments.  This is
  // important for types like Tuple and ThreadIdx that need comptime values for
  // their indices. However, some types VariadicParamList can support both
  // static and dynamic indices, and the implementations have different
  // constraints, and they can't be overloaded.
  //
  // We solve this by having a special name for the static index version, then
  // doing some checks to see if the indices are all static, and if so, using
  // it. Note that the base may be a dynamic value, and we may be in a dynamic
  // context; our support here is just about whether the indices are static.
  StringRef staticGetterName =
      isSubscript ? "__getitem_param__" : "__getattr_param__";
  LookupResult staticGetters = emitter.shared.lookupAndResolveDecl(
      staticGetterName, node->getLoc(), baseType,
      /*searchParentScopes=*/false);
  if (staticGetters.isSuccess()) {
    // The indices may have dynamic values in them that can't be emitted in a
    // parameter context, but could still be emitted to a __getitem__. If the
    // type has a __getitem__ then we have to decide what path to use.
    if (emitter.shared.typeHasMember(baseType, "__getitem__", node->getLoc())) {
      // Check if the indices are all static.
      bool anyDynamic = false;
      size_t numEmitted = 0;
      for (const Operand &operand : exprOperands) {
        emitter.emitExpressionWithoutEvaluatingIt(
            operand.expr, EC_Origin, [&](CValue result, IREmitter &emitter) {
              anyDynamic |= !result.getIfPValue();
              ++numEmitted;
            });
      }
      if (numEmitted != exprOperands.size())
        return {}; // Operand had an error emitting it.
      if (anyDynamic)
        staticGetters.setToFailure();
    }
  }
  // Ok, if we haven't veto'd the static index version, emit it.
  if (staticGetters.isSuccess()) {
    // emit base.__getitem_param__[indices]() or base.__getitem_param__[indices]
    // depending on whether the decl we found is a def or a comptime.
    SyntheticNode baseNode(node->getLoc(), base.ir);
    AttributeRefNode getItemNode(
        &baseNode, node->getLoc(),
        StringAttr::get(emitter.getContext(), staticGetterName));
    SubscriptNode subscriptNode(&getItemNode, node->getLoc(), exprOperands,
                                node->getLoc());
    if (isa_and_nonnull<AliasDeclOp>(
            staticGetters.getIfSuccess().front()->getIfOperation())) {
      return emitter.emitExpr(&subscriptNode, dest);
    }

    CallNode callNode(&subscriptNode, node->getLoc(),
                      /*no operands*/ ArrayRef<Operand>(), node->getRangeEnd());
    return emitter.emitExpr(&callNode, dest);
  }

  // Look up the getter and setter candidate list on the self type.
  StringRef getterName = isSubscript ? "__getitem__" : "__getattr__";
  OverloadSet getterSet = OverloadSet::lookup(emitter.getDeclScope(), baseType,
                                              getterName, node, syntax);
  if (getterSet.isErroneous())
    return {}; // Ignore already emitted errors.

  StringRef setterName = isSubscript ? "__setitem__" : "__setattr__";
  OverloadSet setterSet = OverloadSet::lookup(emitter.getDeclScope(), baseType,
                                              setterName, node, syntax);
  if (setterSet.isErroneous())
    return {}; // Ignore already emitted errors.

  // If there is no getter or setter, then we need to fail.
  if (!setterSet && !getterSet) {
    lookupError();
    return {};
  }

  if (!getterSet) {
    auto diag = emitter.emitError(node->getLoc())
                << baseType << " has '" << setterName << "' but no '"
                << getterName << "' method" << node->getRange();
    diag.attachNote(node->getLoc())
        << "used in an expression here" << node->getRange();
    return {};
  }

  // Ok, we'll be calling the getter and/or setter, passing the indices as
  // dynamic arguments.
  CallOperands operands(CallSyntax::kMethodCall, node, EC_Subscript);
  operands.addSelf(base);

  for (const Operand &operand : exprOperands) {
    ExprNode *expr = operand.expr;
    AnyValue exprVal = emitter.emitExpr(expr, EC_Subscript);
    if (!exprVal)
      return {};
    operands.add(operand.name, ASTExprAnd<AnyValue>{exprVal, expr},
                 operand.unpackStyle);
  }

  // If we /just/ have a getter, or we already know we're assigning this into
  // some destination, emit this as a call to the getter. This gives us nice
  // tuned diagnostics.  We also do this in a comptime context, because the
  // setter will never be called, and we want PValues here, not DLValues.
  if (!setterSet || !emitter.builder || dest.isSpecified()) {
    operands.dest = std::move(dest);
    return getterSet.emitCall(std::move(operands), emitter);
  }

  // Okay, we definitely have both a setter and a getter.  The problem
  // is that we don't know in which context this expression will be used - it
  // could be loaded from, stored to, or both (with a mut argument), and it
  // might even have computed contextual parameters.

  // Resolve the getter to discover the element type.
  auto getter =
      getterSet.filterOverloadSet(operands,
                                  /*emitDiagnosticOnFailure*/ true, emitter);
  if (!getter.isYes()) // Error already emitted.
    return {};

  // ElementType is the result of the getter, processing by-ref results and
  // ignoring the variant for raising functions.
  ASTType elementType = getter.getYes().getType().getSignatureUserResultType();

  // Also look through ref results.
  if (FnOrFnLiteralTypeGeneratorType::get(getter.getYes().getType())
          .isRefResult())
    elementType = sugarCast<RefType>(elementType).getElementType();

  // Ok, now that we know the elementType, we can look up any setter that we
  // need to use.

  // If the accessors are defined with the new value as a keyword-only
  // argument (eg because the indices are variadic), then we need to pass as a
  // keyword, otherwise we can pass as a positional argument.  This is a bit
  // awkward for overload resolution because we don't know what name each
  // overload might use.  It seems reasonable to require that all overloads of
  // __setitem__ use the same name for their value argument, so we just sniff
  // at the first entry of the set to see what it uses and assume the rest use
  // the same name.
  auto firstFnSig = setterSet.fnDecls.front()->getDeclFullSignature();

  // Find the last user declared argument.
  auto argNo = firstFnSig.getNumArguments();
  do {
    --argNo;
    // Can't use the self argument as the new value.
    if (argNo == 0) {
      auto diag = emitter.emitError(setterSet.fnDecls.front()->getLoc())
                  << setterName
                  << " must take at least one argument for the value to set"
                  << node->getRange();
      diag.attachNote(node->getLoc())
          << "used in an expression here" << node->getRange();
      return {};
    }
    // Ignore the byref return and error arguments.
  } while (isResultSlot(firstFnSig.getArgConvention(argNo)));

  StringAttr setterValueName = firstFnSig.getArgName(argNo);
  if (operands.findKwArg(setterValueName)) {
    auto diag = emitter.emitError(node->getLoc())
                << "keyword argument " << setterValueName
                << " may not be specified in the index list, it is needed "
                   "for the new value"
                << node->getRange();
    return {};
  }

  // Otherwise, this expression may be used as an LValue so form it.
  DLValue result(RCRef<SubscriptDLValue>::create(
      getter.getYes(), setterValueName, std::move(operands), elementType));
  return emitter.emitResult(result, node, dest);
}

/// Emit a qualified attribute reference like "x.y" to MLIR. These can always be
/// emitted eagerly in the LHS of an assignment because the base expression can
/// never have an inferred type.
auto AttributeRefNode::emitLCVIR(ExprDest &dest, IREmitter &emitter,
                                 bool isSpeculative) const -> ELVIITResult {
  auto &shared = emitter.shared;

  // In order to allow parameter expressions which technically include a runtime
  // reference, i.e `x.static_field` we allow some values which would otherwise
  // produce a value in a parameter context to still propagate up.
  AnyValue baseAny = emitter.emitExpr(base, EC_AttributeRefBase);
  if (!baseAny)
    return {};

  // If the base is an inferred-base attribute (e.g. `.red.opacity`), defer the
  // whole attribute reference until a contextual type is available.
  if (baseAny.getIfInferredBaseAttrRef())
    return emitter.emitResult(InferredBaseAttrRefUValue(this), this, dest);

  CValue baseVal = emitter.emitCValue({baseAny, base}, EC_AttributeRefBase);
  if (!baseVal)
    return {};

  // Figure out what type is being accessed.  'hasTypeBase' is when the base
  // expression is itself a type, e.g. `Int.__add__`.
  ASTType baseRVType;
  bool hasTypeBase = false;

  // Handle member references on types, like Int.member.
  if (ASTType baseType = baseVal.getIfTypeValue()) {
    baseRVType = baseType;
    hasTypeBase = true;
    maybeRefineTypeAttributeBase(baseVal, baseRVType, emitter.declScope);
  } else {
    // Otherwise, it must be an access to a field of a value.  Look up in the
    // RValueType of the value.
    baseRVType = baseVal.getRValueType();
  }

  // Find the decl for the type we're looking up into.
  // Note: Type refinement for parametric types is handled at value creation
  // time (call results in UncheckedCallEmission, function args in
  // DeclResolution, local vars in maybeRefineTypeWithAssumptions) and at
  // variable use time (emitUnqualLookup applies maybeEmitRefRefinement for
  // comptime if/assert scopes), so baseRVType should already be refined if
  // applicable.
  ASTDecl *typeDecl = baseRVType.getDecl(shared);
  if (!typeDecl) {
    // If the attribute spelling is empty, we couldn't find a name to look up.
    // This was already diagnosed during initial parsing, so we can just bail
    // here.
    if (spelling.empty())
      return {};

    // If there is no decl, the type is an MLIR type.
    Type baseMLIRType = baseRVType.mlirType;
    if (sugarIsa<TypeCheckErrorType>(baseMLIRType))
      return {}; // An already-diagnosed error.

    // Handle __mlir_op.`xxx` references, lazily synthesizing values when
    // they are referenced.
    if (sugarIsa<MagicMLIRAttrType>(baseMLIRType)) {
      PValue result = synthesizeMLIRAttrFromString(spelling, getLoc(), shared);
      return emitter.emitResult(result, this, dest);
    }
    if (sugarIsa<MagicMLIRAttrType, MagicMLIRDeferredAttrType>(baseMLIRType)) {
      /// `__mlir_deferred_attr` always behaves like `__mlir_attr` when used
      /// with backticks. Don't want to strictly enforce that, but user should
      /// be aware that use of `__mlir_attr` is preferred
      emitMLIRDeferredAttrToMLIRAttrWarning(getLoc(), emitter);
      PValue result = synthesizeMLIRAttrFromString(spelling, getLoc(), shared);
      return emitter.emitResult(result, this, dest);
    }
    if (sugarIsa<MagicMLIROpType>(baseMLIRType)) {
      PValue result = synthesizeMLIROpFromString(spelling, emitter);
      return emitter.emitResult(result, this, dest);
    }
    if (sugarIsa<MagicMLIRTypeType>(baseMLIRType)) {
      ASTType result = parseMLIRType(spelling, this, shared);
      return emitter.emitResult(result, this, dest);
    }

    // Diagnose use of not-fully-bound generators with a specific error message.
    if (auto paramType = sugarDynCast<ParamType>(baseMLIRType)) {
      if (isa<GeneratorAttr>(paramType.getParam())) {
        emitter.emitError(getLoc())
            << baseRVType
            << " needs more parameters bound before accessing attributes"
            << base->getRange();
        return {};
      }
    }

    emitter.emitError(getLoc())
        << baseRVType << " has no attributes" << base->getRange();
    return {};
  }

  // Notify the listener of a member lookup.
  shared.notifyListenerOnMemberLookup(*typeDecl, getIdentifierLoc());

  // If the attribute spelling is empty, we couldn't find a name to look up.
  // This was already diagnosed during initial parsing, so we can just bail
  // here.
  if (spelling.empty())
    return {};

  // This could be null for TraitType.
  auto typeDeclOp = typeDecl->getIfOperation();

  // Handle import-gated module references. ImportOp gates access to the
  // real package: nested ImportOps control which child modules are accessible.
  if (auto importOp = dyn_cast_or_null<ImportOp>(typeDeclOp)) {
    // The gate may be only signature-resolved when reached here (e.g., reloaded
    // from a precompiled package) so body-resolve it before looking in its
    // scope.
    (void)shared.declResolver->resolveBody(*typeDecl, getLoc());
    // Look for a child ImportOp matching the member name via decl lookup.
    ArrayRef<ASTDecl *> children = typeDecl->lookupInCurrentScope(spelling);
    for (ASTDecl *childDecl : children) {
      if (isa_and_nonnull<ImportOp>(childDecl->getIfOperation())) {
        PValue result(
            ModuleAttr::get(LIT::ModuleType::get(childDecl->getSymbolRef())));
        return emitter.emitCResult(result, this, dest);
      }
    }

    // Not a child ImportOp. Resolve the underlying module and look the member
    // up there. For a package this resolves through its __init__'s scope (the
    // package's public surface), so only names that __init__ defines or
    // re-exports are reachable.
    ASTDecl &realDecl = shared.importModule(
        SharedState::ImportPath::fromAttr(importOp.getModulePath()),
        /*currentPackage=*/nullptr,
        shared.diags.convertLocToSMLoc(importOp->getLoc()));
    auto memberLookup = shared.lookupAndResolveDecl(
        spelling, getLoc(), realDecl, /*searchParentScopes=*/false);
    if (memberLookup.isSuccess()) {
      ASTDecl *memberDecl = memberLookup.getIfSuccess().front();
      // A re-exported submodule surfaces as a nested ImportOp. Hand
      // back that op's module directly; it is in scope only because __init__
      // re-exported it.
      if (isa_and_nonnull<ImportOp>(memberDecl->getIfOperation())) {
        PValue result(
            ModuleAttr::get(LIT::ModuleType::get(memberDecl->getSymbolRef())));
        return emitter.emitCResult(result, this, dest);
      }
      if (isa_and_nonnull<PackageOp, FileModuleOp>(
              memberDecl->getIfOperation())) {
        // A raw submodule bound in __init__'s scope. It is reachable precisely
        // because __init__ bound it, so access is allowed. Bind a child
        // ImportOp over it and hand that back (not the raw module) so deeper
        // access stays gated.
        OpBuilder gateBuilder = typeDecl->getDeclEndBuilder();
        SharedState::ImportPath childPath =
            SharedState::ImportPath::fromAttr(importOp.getModulePath());
        childPath.components.push_back(spelling);
        ASTDecl &childGate = shared.declResolver->createImportOp(
            *typeDecl, gateBuilder,
            StringAttr::get(shared.getContext(), spelling),
            childPath.toAttr(shared.getContext()), importOp->getLoc());
        return emitter.emitCResult(PValue(ModuleAttr::get(LIT::ModuleType::get(
                                       childGate.getSymbolRef()))),
                                   this, dest);
      }
    }

    // Non-module member - delegate to the real module.
    return DeclRefNode::emitUnqualLookup(spelling, this, realDecl, dest,
                                         emitter, false);
  }

  // Member access on a raw module/package. This is reached for intra-package
  // references - a file naming a sibling submodule registered by the directory
  // scan in an ancestor package scope. (Cross-package access goes through an
  // ImportOp, handled above.)
  // FIXME: Can we remove this?
  if (isa_and_nonnull<PackageOp, FileModuleOp>(typeDeclOp))
    return DeclRefNode::emitUnqualLookup(spelling, this, *typeDecl, dest,
                                         emitter, false);

  // We can only look up in something of struct or trait type.
  if (!isa_and_nonnull<StructDeclOp, TraitDeclOp>(typeDeclOp) &&
      !sugarIsa<TraitType>(typeDecl->getIfTypeValue())) {
    emitter.emitError(getLoc(), "cannot access attribute in type ")
        << baseVal.getType() << base->getRange();
    return {};
  }

  // Find the member being accessed. Search through the struct and also through
  // any of its visible extensions.
  SmallVector<ASTDecl *, 4> structAndExtensionsDecls =
      emitter.getDeclScope().collectTypeAndExtensions(baseRVType, getLoc());
  SmallVector<ASTDecl *> memberDecls;
  if (failed(lookupMembersInDecls(memberDecls, structAndExtensionsDecls,
                                  spelling, getLoc(), shared)))
    return {};

  // If the struct has no static member of the required name, try to look for
  // dynamic lookup attribute methods (__getattr__ etc) on the type.
  if (memberDecls.empty()) {
    // Convert the attribute name to a StringLiteralNode since that's how the
    // operand will be emitted.  We need to allocate it persistently to enable
    // DLValues to capture it.
    std::string quotedName = '"' + spelling.str() + '"';
    StringRef quotedNameRef = shared.getPersistentCopy(StringRef(quotedName));
    // StringLiteral wants ArrayRef<StringRef>.
    auto strRefs = shared.getPersistentCopy(ArrayRef(quotedNameRef));
    auto *strLiteral = shared.allocPersistent<StringLiteralNode>(strRefs);
    // Put the expression in a ParenNode so it has a location.
    auto *parenNode =
        shared.allocPersistent<ParenNode>(getLoc(), strLiteral, getLoc());
    return emitGetterSetterAccess(
        this, {baseVal, base},
        Operand(parenNode, getLoc(), ArgUnpackStyle::kPositional), dest,
        emitter);
  }
  shared.notifyListenerOnRef(memberDecls, spelling, this);

  // We place this here rather than earlier in the function because we want to
  // emit an error if the trait member doesn't exist at all (handled right above
  // in emitGetterSetterAccess).
  //
  // TODO(MOCO-2303): We do want to support this eventually for trait methods as
  // we want to be a able to call defaulted trait methods directly with a value
  // of a conforming struct, for example:
  //
  // trait Foo
  //   def foo(self):
  //     print("Foo.foo")
  //
  // struct Bar(Foo):
  //   def foo(self):
  //     print("Bar.foo")
  //
  // var b = Bar()
  // Foo.foo(b) # want this to print Foo.foo
  if (mlir::isa_and_present<TraitType>(baseVal.getIfTypeValue())) {
    emitter.emitError(getLoc(),
                      "Direct access of trait members is not supported.");
    return {};
  }

  // Handle method references, which might be overloaded.
  if (memberDecls[0]->isCallableDecl()) {
    // Build an overload set of all matching function declarations.

    // TODO(ParameterizedType): This representation is subtly wrong.  We should
    // be inferring Self parameters from the expression later rather than
    // installing "getForDeclaredType", because this won't work correctly with
    // nonmaterializable types that need an implicit conversion.
    //
    // We currently need to bind the Self parameters here so that subsequent
    // parameters are bound correctly.  Consider something like:
    //     foo.dyn_cast[Int]()
    // If typeof(foo) has parameters A and B, we need to form a parameter list
    // of `[A, B, Int]`.  If we had ParameterizedType then we could model this
    // correctly as have an unspecified first set of bindings for the type,
    // and the Int binding could go in a subsequent parameter list.
    //
    // When fixing this, remember to check out SubscriptNode::emitLCVIR as well,
    // as it uses similar logic.
    auto bindings = ParamBindings::getForDeclaredType(emitter.getDeclScope(),
                                                      baseRVType, this);
    /// Only bind the `Self`, leave the rest unbound.
    bindings.relaxBindingKindTo(ParamBindings::kWithEllipsis);

    auto result = OverloadSetUValue::create(
        spelling, memberDecls, std::move(bindings), CallSyntax::kDirectCall);

    // If the callee is a static method, we can directly reference it
    // without binding a self parameter.  If this is an instance method, we
    // bind the base value and the symbol together into a callable.
    if (!hasTypeBase) {
      result->baseValue = {baseVal, base};
      result->syntax = CallSyntax::kMethodCall;
    }
    return emitter.emitResult(result, this, dest);
  }

  // Check deprecation and stability for non-function members (fields, aliases).
  // Functions are checked in CallEmission.cpp::getCallee after overload
  // resolution.
  for (ASTDecl *memberDecl : memberDecls)
    if (memberDecl->getIfOperation())
      checkDeclUsageWarnings(*memberDecl, getIdentifierLoc(),
                             emitter.getDeclScope(), shared, getRange(),
                             CallSyntax::kDirectCall, {});

  assert(memberDecls.size() == 1 && "only methods may be overloaded");
  // Maintain sugar.  The base might be dynamic so we turn it into
  // "Type.member" from a sugar perspective.
  auto getParamMemberSugar = [&](PValue value) -> PValue {
    // Propagate failures, never sugar aliases that start with an _.  These
    // are internal implementation details of types and eliding this
    // chops gigabytes of IR out of .mlir files for the std.
    if (!value || isInternalName(spelling))
      return value;

    auto memberName = StringAttr::get(emitter.getContext(), spelling);
    return SugarAttr::getMemberAlias(baseRVType, memberName, value);
  };

  // References to an AliasDecl form a PValue.
  if (isa_and_nonnull<AliasDeclOp>(memberDecls[0]->getIfOperation())) {
    // Handle accessing an alias in a struct or an extension.
    // Can not be a TraitDeclOp, member lookup should be routed with a TraitType
    // Decl, and should be handled below;
    assert(isa_and_nonnull<StructDeclOp>(typeDeclOp) ||
           isa_and_nonnull<ExtensionDeclOp>(
               memberDecls[0]->getParentDecl()->getIfOperation()));
    PValue result = resolveAliasReference(
        cast<AliasDeclOp>(memberDecls[0]->getIfOperation()), spelling,
        baseRVType.getParamBindings(), getLoc(), emitter);
    return emitter.emitResult(getParamMemberSugar(result), this, dest);
  } else if (WitnessDecl *witness = memberDecls[0]->getIfWitness()) {
    assert(sugarIsa<TraitType>(typeDecl->getIfTypeValue()) &&
           "Trait witness lookup must be routed via TraitType decl.");

    PValue basePValue = baseVal.getIfPValue();
    if (!basePValue) {
      emitter.emitError(getLoc(), "can't access '")
          << spelling << "' in non-parameter " << baseRVType << getRange();
      return {};
    }

    Type aliasType = witness->getWitnessEntry().witnessType;
    StringAttr witnessName = witness->getWitnessEntry().witnessName;
    TraitSymbolAttr traitSymbol = witness->traitSymbol;

    // If the base has a parametric type, use the trait type instead.
    if (auto paramType = sugarDynCast<ParamType>(baseRVType))
      basePValue = PValue(paramType.getParam());

    // Upcast the composition into the trait that defined the alias so that
    // types match.
    ASTDecl *traitDecl =
        &shared.declResolver->getDeclForTypeSymbol(traitSymbol.getSymbol());
    basePValue = emitter.emitMetaTypeToTraitConversion(
        {basePValue, base},
        cast<TraitDeclOp>(traitDecl->getIfOperation()).getCanonicalTrait());

    // In the case that aliasDeclOpParam is a dependent associated type alias,
    // we need to first rebind `_Self` to `T`.
    //
    // In the following example:
    //
    // trait Trait:
    //   alias T1 : AnyType
    //   alias T2 : T1
    //
    // When emitting
    // def foo[
    //   T : Trait
    //   t : T.T2
    // ](...)
    //   To emit `T.T2`, we need to fold
    //
    //   # get_witness_attr<T, "Trait", "T2"> :
    //       #get_witness_attr<"_Self", "Trait", "T1">: AnyType
    //
    //   # to
    //
    //   # get_witness_attr<T, "Trait", "T2"> :
    //       #get_witness_attr<T, "Trait", "T1">: AnyType

    // TraitDeclOp always have one input parameter `_Self`
    ParamDeclAttr traitSelf =
        cast<TraitDeclOp>(traitDecl->getIfOperation()).getParamsAttr().back();
    ParameterEvaluator selfReplacer = shared.getParameterEvaluator();
    selfReplacer.setDeclBinding(traitSelf.getName(), basePValue);
    aliasType = selfReplacer.getReboundType(aliasType);

    auto witnessEntryResult =
        shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
            basePValue, traitSymbol, witnessName, aliasType);
    return emitter.emitResult(getParamMemberSugar(witnessEntryResult), this,
                              dest);
  }

  ASTDecl &memberDecl = *memberDecls[0];
  // If the field is a variable, emit a reference to it.
  if (auto fieldOp =
          dyn_cast_or_null<StructFieldOp>(memberDecl.getIfOperation())) {
    if (hasTypeBase || sugarIsa<StructMetaType>(baseRVType)) {
      emitter.emitError(getLoc(), "cannot access instance field '")
          << spelling << "' without an instance of " << baseRVType
          << getRange();
      return {};
    }

    // We know that baseVal is a CValue, so handle all the cases.

    // If the base is a DLValue, we need to emit this as a projected DLValue.
    // This allows to emit a get and/or set as needed.
    if (DLValue baseLV = baseVal.getIfDLValue()) {
      // The base is a known StructType because we got the ASTDecl from it.
      ASTType elementType = fieldOp.getReboundType(
          sugarCast<LIT::StructType>(baseRVType.mlirType),
          &emitter.shared.getEvaluationContext());
      DLValue result(RCRef<StoredAttributeRefDLValue>::create(
          ASTExprAnd<DLValue>{baseLV, base}, fieldOp, elementType, this));
      return emitter.emitResult(result, this, dest);
    }

    // Otherwise, emit the stored field reference.
    return emitStoredFieldRef({baseVal, base}, fieldOp, this, dest, emitter);
  }

  // This parameter will refer to the generic parameter on the base type decl,
  // e.g the base struct. We need to substitute it for the "real" parameter used
  // to construct this specific type, not the shared type on the struct.
  auto structDeclOp = dyn_cast<StructDeclOp>(typeDecl->getIfOperation());
  if (auto parameter = memberDecl.getIfIRValue().getIfPValue();
      parameter && structDeclOp) {
    auto paramRef = cast<ParamDeclRefAttr>(parameter.get());
    auto baseRVMetaType = baseRVType.extractMetaType();

    // Two possible cases: 1) a partially bound struct type, 2) a generator
    auto paramValues = ArrayRef<TypedAttr>();
    if (auto baseDecl = sugarDynCast<StructMetaType>(baseRVMetaType)) {
      paramValues = baseDecl.getParamValues();
    } else if (auto genType = sugarDynCast<GeneratorType>(baseRVMetaType)) {
      // Get the param values for the generator attr body.
      auto genBodyType = sugarDynCast<StructMetaType>(genType.getBody());
      if (genBodyType) {
        paramValues = ASTType(genBodyType.getType())
                          .getWithUnknownParametersReplaced(shared)
                          .getParamBindings();
      }
    }

    if (!paramValues.empty()) {
      for (auto [name, value] :
           llvm::zip(structDeclOp.getParams(), paramValues)) {
        // If this binding is for this parameter propagate the bound
        // parameter.
        if (name.getName() == paramRef.getName()) {
          if (isa<UnboundAttr>(value)) {
            // Detect if we are accessing a unbound parameter.
            emitter.emitError(getLoc(), "'")
                << spelling << "' refers to an unbound parameter in "
                << baseRVType;
            return {};
          }
          return emitter.emitResult(value, this, dest);
        }
      }
    }
  }

  // Reference to some non-function/struct member of the type.
  emitter.emitError(getLoc(), "reference to unknown member '")
      << spelling << "'" << getRange();
  return {};
}

/// Emission for ".foo" expressions, which are inferred attribute references.
auto InferredAttributeRefNode::emitLCVIR(ExprDest &dest, IREmitter &emitter,
                                         bool isSpeculative) const
    -> ELVIITResult {
  // Emission is trivial: we just wrap it in a UValue so type checking is
  // deferred until we resolve the contextual type (or not).
  return emitter.emitResult(InferredBaseAttrRefUValue(this), this, dest);
}

/// Decode a canonical `_type=` value (shared by the dot-syntax and f-string
/// `__mlir_op` paths). The bare `__mlir_deferred_type` marker sets
/// `forceDeferred` and adds no type; a `Tuple[...]` appends each element; any
/// other type appends itself. Emits an error at `loc` and fails if a value
/// isn't a type.
static LogicalResult decodeMLIRResultTypes(TypedAttr value, SMLoc loc,
                                           IREmitter &emitter,
                                           SmallVectorImpl<Type> &out,
                                           bool &forceDeferred) {
  value = getCanonicalAttr(value);
  if (sugarIsa<MagicMLIRDeferredTypeType>(ASTType(value).mlirType)) {
    forceDeferred = true;
    return success();
  }
  auto pushType = [&](TypedAttr type, const Twine &message) -> LogicalResult {
    if (!LIT::isTypeExpr(type)) {
      emitter.emitError(loc, message);
      return failure();
    }
    out.push_back(ASTType(type));
    return success();
  };
  if (auto valueMetaType = sugarDynCast<StructMetaType>(value.getType())) {
    ASTType tupleType =
        emitter.shared.lookupBuiltinType("Tuple", emitter.declScope, loc);
    if (valueMetaType.getSymbol() ==
        sugarCast<StructMetaType>(tupleType.extractMetaType()).getSymbol()) {
      auto tca = sugarCast<TypeParamAttr>(value);
      auto drt = sugarCast<LIT::StructType>(tca.getMlirType());
      ArrayRef<TypedAttr> paramValues = drt.getParamValues();
      assert(paramValues.size() == 2 && "Tuple ParamValues must be size 2");
      auto variadic = sugarCast<ParamListAttr>(paramValues[0]);
      for (TypedAttr type : variadic.getValues())
        if (failed(pushType(type, "value in _type tuple is not a type")))
          return failure();
      return success();
    }
  }
  return pushType(value, "_type value is not a type");
}

/// Given a call to an UnboundMLIROperator, generate an MLIR operation with
/// the operands as SSA values.
static AnyValue emitMLIROperatorCall(const CallNode &call,
                                     UnboundMLIROperationAttr unboundOp,
                                     ExprDest &dest, IREmitter &emitter) {
  auto *context = emitter.getContext();
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(&call);

  // Emit all the arguments so we can encode them as SSA values.
  SmallVector<Value> opOperands;
  for (const Operand &argument : call.operands) {
    if (!argument.isPositional()) {
      emitter.emitError(argument.getLoc(),
                        "MLIR operators only support positional arguments");
      return {};
    }
    Value value = emitter.emitExprSRValue(argument.expr, EC_MLIRMagic);
    if (!value)
      return {};
    // We strip sugar types off all arguments, so they don't interfere with
    // operator invariants.
    value = emitter.emitRebindOpIfNeeded(
        value, getCanonicalType(value.getType()), argument.expr->getLoc());
    opOperands.push_back(value);
  }

  StringAttr opName = unboundOp.getName();
  OperationState state(call.getLocation(emitter), opName);

  // Do special preprocessing of a specified type attribute. That needs to
  // happen before useDeferredOp is called, because useDeferredOp will analyze
  // result types stored in the state.
  bool hadTypeSpec = false;
  // Set by `_type=__mlir_deferred_type`: force `kgen.deferred` without
  // adding any result type.
  bool forceDeferred = false;
  for (auto &attr : unboundOp.getAttrs()) {
    if (attr.getName() == "_type") {
      // We expect either a single type, `None`, or a `Tuple` of types.
      if (isa<NoneAttr>(attr.getValue())) {
        hadTypeSpec = true;
        continue;
      }

      auto value = dyn_cast<TypedAttr>(attr.getValue());
      if (!value) {
        emitter.emitError(call.getLoc(), "unknown _type value");
        return {};
      }
      if (failed(decodeMLIRResultTypes(value, call.getLoc(), emitter,
                                       state.types, forceDeferred)))
        return {};
      hadTypeSpec = true;
      continue;
    }
  }

  // Return true if any attribute or result type is deferred.
  auto useDeferredOp = [&]() {
    // If any attribute of the operation is deferred, the operation is deferred
    // too.
    if (llvm::any_of(unboundOp.getAttrs(), [](NamedAttribute attr) {
          if (auto typedAttr = dyn_cast<TypedAttr>(attr.getValue()))
            return sugarIsa<DeferredType>(typedAttr.getType());
          return false;
        })) {
      return true;
    }

    // A result type of `!kgen.deferred_type` requires a deferred op since
    // the concrete type is not yet known; verification must be deferred.
    if (llvm::any_of(state.types, [&](Type type) {
          assert(!sugarIsa<DeferredType>(type) &&
                 "Deferred type is not allowed in return");
          return sugarIsa<MLIRDeferredType>(type);
        })) {
      return true;
    }
    return false;
  };

  bool isDeferredOp = false;
  NamedAttrList attrs;
  if (forceDeferred || useDeferredOp()) {
    // Change operation name within a state to `kgen.deferred`
    state.name = mlir::OperationName(DeferredOp::getOperationName(), context);
    isDeferredOp = true;
  }

  // Set up the OperationState for the thing we're building.
  state.addOperands(opOperands);

  // Process the attributes
  std::optional<Attribute> propsAttr = std::nullopt;
  for (auto &attr : unboundOp.getAttrs()) {
    if (attr.getName() == "_type") {
      // This attribute has been already processed.
      continue;
    }
    if (attr.getName() == "_region") {
      // A region is specified for an MLIR operation by using the `_region`
      // special attribute to refer to a function declaration.
      auto bodyRef = dyn_cast<StringAttr>(attr.getValue());
      if (!bodyRef) {
        emitter.emitError(call.getLoc(),
                          "MLIR operation region must be a region reference");
        return {};
      }
      // Lookup the operation body.
      LookupResult result = emitter.shared.lookupAndResolveDecl(
          bodyRef, call.getLoc(), emitter.declScope,
          /*searchParentScopes=*/false);
      ArrayRef<ASTDecl *> results = result.getIfSuccess();
      if (result.isFailure() || results.size() != 1 ||
          !isa<LIT::UnboundRegionOp>(results.front()->getIfOperation())) {
        emitter.emitError(call.getLoc(), "MLIR operation region reference did "
                                         "not resolve to a region body");
        return {};
      }
      auto unboundRegion =
          cast_or_null<LIT::UnboundRegionOp>(results.front()->getIfOperation());
      auto region = std::make_unique<Region>();
      region->takeBody(unboundRegion.getRegion());
      unboundRegion.erase();
      results.front()->setIRValue(PValue(BoolAttr::get(context, false)));
      state.addRegion(std::move(region));
      continue;
    }
    if (attr.getName() == "_properties") {
      propsAttr = attr.getValue();
      continue;
    }
    if (isDeferredOp) {
      // For deferred operation, fill the dictionary with all attributes
      // regardless if attribute itself is deferred or not. That allows to have
      // any number of any attributes for deferred operation.
      attrs.set(attr.getName(), attr.getValue());
      continue;
    }
    state.addAttributes(attr);
  }

  if (isDeferredOp) {
    auto deferredName =
        mlir::OperationName(DeferredOp::getOperationName(), context);
    state.addAttribute(DeferredOp::getOpNameAttrName(deferredName),
                       unboundOp.getName());
    state.addAttribute(DeferredOp::getOpAttrsAttrName(deferredName),
                       attrs.getDictionary(context));
    // `kgen.deferred` has no property storage; stash `_properties` on it so
    // the elaborator can re-apply it to the reconstructed op.
    if (propsAttr) {
      state.addAttribute(DeferredOp::getOpPropertiesAttrName(deferredName),
                         *propsAttr);
      propsAttr.reset();
    }
  }

  // Finally, if we don't already have a type, figure out the return types
  // using InferTypeOpInterface if the operation is registered and if it is
  // present.
  auto inferType = [&]() -> LogicalResult {
    auto opNameInfo =
        mlir::RegisteredOperationName::lookup(unboundOp.getName(), context);
    if (!opNameInfo)
      return failure();

    // Check to see if this implements the ZeroResults trait.
    if (opNameInfo->hasTrait<mlir::OpTrait::ZeroResults>())
      return success(); // We know there are zero results.

    // Otherwise, check for InferTypeOpInterface.
    auto inferTypesItf = opNameInfo->getInterface<mlir::InferTypeOpInterface>();
    if (!inferTypesItf)
      return failure();

    SmallVector<char> propStorage(opNameInfo->getOpPropertyByteSize());
    mlir::PropertyRef properties = mlir::PropertyRef(
        opNameInfo->getOpPropertiesTypeID(), propStorage.data());
    auto attributes = state.attributes.getDictionary(context);

    // Initialize the properties storage
    if (!propStorage.empty() && !state.attributes.empty()) {
      auto emitError = [&]() {
        return mlir::emitError(state.location)
               << " failed properties conversion while building "
               << state.name.getStringRef() << " with `" << attributes << "`: ";
      };
      if (failed(opNameInfo->setOpPropertiesFromAttribute(
              state.name, properties, attributes, emitError)))
        return failure();
    }

    if (failed(inferTypesItf->inferReturnTypes(
            context, state.location, state.operands, attributes, properties,
            state.regions, state.types)))
      return failure();

    return success(
        llvm::all_of(state.types, [](Type t) { return t != Type(); }));
  };

  if (!hadTypeSpec) {
    if (failed(inferType())) {
      emitter.emitError(call.getLoc(),
                        "unable to infer result type from MLIR operation ")
          << unboundOp.getName() << call.getRange();
      return {};
    }
    if (state.types.size() > 1) {
      emitter.emitError(
          call.getLoc(),
          "cannot use operations with multiple inferred results (yet) ")
          << unboundOp.getName() << call.getRange();
      return {};
    }
  }

  for (Type type : state.types) {
    if (!ASTType(type).isRegisterPassable(call.getLoc(), emitter.shared)) {
      emitter.emitError(call.getLoc())
          << ASTType(type)
          << " cannot be returned directly from __mlir_op as it is not a "
             "'RegisterPassable' types";
      return {};
    }
  }

  // Check for an unregistered operation, because otherwise MLIR will crash when
  // assertions are enabled.
  if (!state.name.getDialect() &&
      !emitter.getContext()->allowsUnregisteredDialects()) {
    emitter.emitError(call.getLoc(), "use of unregistered MLIR operation ")
        << unboundOp.getName() << call.getRange();
    return {};
  }

  Operation *resultOp = emitter.builder->create(state);

  // Check if the attributes specified are all inherent attributes.
  DenseSet<StringAttr> inherentAttrs;
  inherentAttrs.insert_range(resultOp->getName().getAttributeNames());
  for (NamedAttribute &attr : state.attributes) {
    if (!inherentAttrs.contains(attr.getName())) {
      emitter.emitError(call.getLoc(), "attribute ")
          << attr.getName() << " is not an inherent attribute of "
          << unboundOp.getName();
      return {};
    }
  }

  // Set the properties if needed. We do this here, because errors result in a
  // crash in the op builder if we simply set state.propertiesAttr.
  if (propsAttr) {
    if (failed(resultOp->setPropertiesFromAttribute(
            *propsAttr, [&]() { return resultOp->emitError(); }))) {
      emitter.emitError(call.getLoc(), "cannot set property");
      return {};
    }
  }

  // Explicitly run the verifier on the new operation so we make sure to
  // catch problems early.
  std::string errorMessage;
  bool verificationError = false;
  // FIXME: Terminators expect certain parent operations and are only valid when
  // inlined into an operation's region. Don't verify them.
  if (!resultOp->hasTrait<OpTrait::IsTerminator>()) {
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(
        context, [&](Diagnostic &diag) { errorMessage = diag.str(); });
    // Verify that the resulting op is correctly constructed.  If not, we
    // fail.
    verificationError = failed(mlir::verify(resultOp));
  }
  if (verificationError) {
    emitter.emitError(call.getLoc(), "MLIR verification error: ")
        << errorMessage;
    return {};
  }

  // Helper to emit an SValue or PValue to the destination.
  auto emitResult = [&](CValue value) -> AnyValue {
    return emitter.emitResult(value, &call, dest);
  };

  // If we succeeded and have no types, then install a None value.
  if (resultOp->getNumResults() == 0)
    return emitResult(PValue(emitter.shared.getNoneAttr()));

  if (resultOp->getNumResults() == 1) {
    OpResult res = resultOp->getResult(0);
    ASTType resType = res.getType();

    // Check to see if we can fold this operation.  This enables use of
    // __mlir_op to produce meta-values without forcing them into the dynamic
    // value domain.
    SmallVector<Attribute, 4> constOperands(resultOp->getNumOperands());
    for (unsigned i = 0, e = constOperands.size(); i != e; ++i)
      matchPattern(resultOp->getOperand(i),
                   mlir::m_Constant(&constOperands[i]));
    SmallVector<OpFoldResult, 4> foldResults;
    if (succeeded(resultOp->fold(constOperands, foldResults)) &&
        foldResults.size() == 1) {
      auto folded = PointerUnion<Attribute, Value>(foldResults[0]);
      CValue result;
      // If the result was some other value that already exists, use it.
      if (auto val = dyn_cast<Value>(folded)) {
        result = SRValue(val);
      } else {
        // If it is a constant, make an PValue result.
        if (auto attr = dyn_cast<TypedAttr>(cast<Attribute>(folded)))
          result = PValue(attr);
        // If it isn't a TypedAttr, then we cannot fold to it.
      }

      if (result) {
        if (result.getRValueType().isEqualCanon(resType)) {
          resultOp->erase();
          return emitResult(result);
        }
        emitter.emitError(call.getLoc())
            << unboundOp.getName() << " operation folded to result type "
            << result.getRValueType() << " but we expected it to be " << resType
            << call.getRange();
        return {};
      }
    }

    // If folding failed, return the operation's result normally.
    return emitResult(SRValue(res));
  }

  // Pack results into a tuple and return it.
  auto tupleType = emitter.shared.lookupBuiltinType("Tuple", emitter.declScope,
                                                    call.getLoc());
  if (tupleType.isTypeCheckErrorType())
    return {};

  // Construct the Tuple type without parameters so we infer them.
  tupleType = tupleType.getWithoutParameters(emitter.shared);

  CallOperands operands(CallSyntax::kTypeCall, &call, std::move(dest));
  for (OpResult opResult : resultOp->getResults())
    operands.add({SRValue(opResult), &call});
  return emitter.emitConstructorCall(tupleType, std::move(operands));
}

AnyValue CallNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  AnyValue calleeVal = emitter.emitExpr(callee, EC_CallCalleeValue);
  if (!calleeVal)
    return {};

  // If the callee is an inferred-base attribute (e.g. `.hsb_to_rgb`), defer the
  // whole call until a contextual result type is available to resolve the base.
  if (calleeVal.getIfInferredBaseAttrRef())
    return emitter.emitResult(InferredBaseAttrRefUValue(this), this, dest);

  // If this is the invocation of an unbound MLIR operator, bind it into an
  // actual operator!
  if (auto mValue = calleeVal.getIfPValue()) {
    if (auto unboundOp = dyn_cast<UnboundMLIROperationAttr>(mValue.get()))
      return emitMLIROperatorCall(*this, unboundOp, dest, emitter);
  }

  /// Emit all the operands that we'll need.
  CallOperands operandsList(CallSyntax::kDirectCall, this, std::move(dest));
  for (const Operand &operand : operands) {
    ExprNode *operandExpr = operand.expr;
    if (operand.unpackStyle == ArgUnpackStyle::kStar) {
      auto unaryOp = cast<UnaryOpNode>(operand.expr);
      ExprNode *packedExpr = unaryOp->subExpr;
      if (packedExpr->kind == ExprNode::kDiscardLiteral) {
        emitter.emitError(operand.getLoc())
            << "unbound packs not supported yet in runtime arguments";
        operandsList.dest.resetForError(emitter);
        return {};
      }
      operandExpr = packedExpr;
    }

    ASTExprAnd<AnyValue> exprAndVal = {
        emitter.emitExpr(operandExpr, EC_CallArgValue), operandExpr};
    if (!exprAndVal) {
      operandsList.dest.resetForError(emitter);
      return {};
    }

    operandsList.add(operand.name, std::move(exprAndVal), operand.unpackStyle);
  }

  // If the callee is a type value (as in `T()` or `T[123]()`), then this is an
  // invocation of the initializer for the type, or a call to a custom op.
  if (ASTType calledType = calleeVal.getIfTypeValue()) {
    auto *calleeDecl = calledType.getDecl(emitter.shared);
    if (!calleeDecl) {
      emitter.emitError(getLoc(), "cannot construct type ")
          << calledType << callee->getRange();
      operandsList.dest.resetForError(emitter);
      return {};
    }

    // Check to see if we can invoke an __init__ method to convert it.
    operandsList.syntax = CallSyntax::kTypeCall;
    return emitter.emitConstructorCall(calledType, std::move(operandsList));
  }

  // If this is an overloaded operand, resolve it and call the result.
  if (auto overloads = calleeVal.getIfOverloadSet()) {
    emitter.shared.notifyListenerOnCall(overloads->fnDecls, rparenLoc,
                                        overloads->syntax, operandsList);
    operandsList.syntax = overloads->syntax;
    return overloads->emitCall(std::move(operandsList), emitter);
  }

  // Otherwise, we must have a concrete RValue, emit an indirect call.
  if (auto crVal = calleeVal.getIfCValue()) {
    operandsList.syntax = CallSyntax::kIndirectCall;
    return emitter.emitIndirectCall(crVal, std::move(operandsList));
  }

  emitter.emitError(getLoc(), "cannot call this unresolved expression");
  operandsList.dest.resetForError(emitter);
  return {};
}

AnyValue SliceLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  auto getOperand = [&](const ExprNode *expr) -> ASTExprAnd<AnyValue> {
    if (expr)
      return {emitter.emitExpr(expr, ExprContext::EC_SliceIndex), expr};

    // Missing expressions resolve into None.
    return {PValue(NoneAttr::get(emitter.getContext())), this};
  };

  // TODO: Generalize to more than 3 operands.  We might also want to turn this
  // into a well-known static method instead of overloading onto constructor.
  CallOperands operands(CallSyntax::kTypeCall, this, EC_SliceIndex);
  operands.add(getOperand(lower));
  operands.add(getOperand(upper));
  operands.add(getOperand(stride));
  if (!operands.values[0].ir || !operands.values[1].ir ||
      !operands.values[2].ir)
    return {};

  auto result = InitializerUValue::create(InitializerUValue::kSliceLiteral,
                                          std::move(operands));
  return emitter.emitResult(result, this, dest);
}

//===----------------------------------------------------------------------===//
// `__mlir_op[`f-string`, _type=...]` subscript syntax
//===----------------------------------------------------------------------===//
//
// Uses the op's natural assembly with `%{name}` / `%{type_of(name)}`
// resolving to Mojo identifiers in scope. Shapes:
//   __mlir_op[`fstring`]
//   __mlir_op[`fstring`, _type=...]
//   __mlir_op[_type=..., `fstring`]

/// Rewrite `%{name}` → `%arg<N>` / `%param<N>` and `%{type_of(name)}` → operand
/// type text, then lower now via `KGEN::lowerFStringMLIROp` or stash as
/// `kgen.deferred` for the elaborator (parametric types / comptime parameters).
static AnyValue emitFStringSubscriptMLIROp(const SubscriptNode &subscript,
                                           ExprDest &dest, IREmitter &emitter) {
  SMLoc loc = subscript.getLoc();
  MLIRContext *context = emitter.getContext();
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(&subscript);

  // 1. Validate shape: one or more positional backtick templates (concatenated
  //    at parse time, mirroring `__mlir_type[…]` / `__mlir_attr[…]`) plus an
  //    optional `_type=` keyword.
  SmallVector<const Operand *> fstringOps;
  const Operand *typeOp = nullptr;
  for (const Operand &op : subscript.operands) {
    if (op.isPositional()) {
      fstringOps.push_back(&op);
    } else if (op.isKeyword() && op.name && op.name.getValue() == "_type") {
      if (typeOp) {
        emitter.emitError(op.expr->getLoc(),
                          "duplicate '_type' in __mlir_op[...]");
        return {};
      }
      typeOp = &op;
    } else {
      emitter.emitError(op.expr->getLoc(),
                        "__mlir_op[...] only accepts backtick template "
                        "chunks and an optional '_type=' keyword");
      return {};
    }
  }
  if (fstringOps.empty()) {
    emitter.emitError(loc,
                      "__mlir_op[...] requires a backtick f-string template");
    return {};
  }

  // 2. Concatenate the backtick chunks; the multi-chunk path interns the
  //    concat via the shared allocator so derived `StringRef`s outlive
  //    diagnostics.
  StringRef tmpl;
  if (fstringOps.size() == 1) {
    auto *dre = dyn_cast<DeclRefNode>(fstringOps.front()->expr);
    if (!dre || dre->spelling.empty() ||
        dre->spelling.data()[dre->spelling.size()] != '`') {
      emitter.emitError(fstringOps.front()->expr->getLoc(),
                        "__mlir_op[...] template chunk must be a backtick "
                        "string");
      return {};
    }
    tmpl = dre->spelling;
  } else {
    std::string concat;
    for (const Operand *fop : fstringOps) {
      auto *dre = dyn_cast<DeclRefNode>(fop->expr);
      if (!dre || dre->spelling.empty() ||
          dre->spelling.data()[dre->spelling.size()] != '`') {
        emitter.emitError(fop->expr->getLoc(),
                          "__mlir_op[...] template chunk must be a backtick "
                          "string");
        return {};
      }
      concat.append(dre->spelling.data(), dre->spelling.size());
    }
    tmpl = emitter.shared.getPersistentCopy(StringRef(concat));
  }

  // 3. Process `_type` if present (single Type or Tuple of Types).
  SmallVector<Type> resultTypes;
  bool hadTypeSpec = false;
  bool forceDeferred = false;
  if (typeOp) {
    AnyValue typeAny = emitter.emitExpr(typeOp->expr, EC_MLIRMagic);
    if (!typeAny)
      return {};
    auto pv = typeAny.getIfPValue();
    if (!pv) {
      emitter.emitError(typeOp->expr->getLoc(), "unknown _type value");
      return {};
    }
    auto value = dyn_cast<TypedAttr>(pv.get());
    if (!value) {
      emitter.emitError(typeOp->expr->getLoc(), "unknown _type value");
      return {};
    }
    if (failed(decodeMLIRResultTypes(value, typeOp->expr->getLoc(), emitter,
                                     resultTypes, forceDeferred)))
      return {};
    hadTypeSpec = true;
  }

  // 4. Walk the template, resolving `%{name}` once per name:
  //      * SRValue → operand, emit `%arg<N>`.
  //      * PValue (comptime/parametric) → `ToStringDeferredAttr`, emit
  //        `%param<N>` for the elaborator to inline.
  //    `%{type_of(name)}` prints the operand's MLIR type now, unless the type
  //    is parametric (`MLIRDeferredType`), in which case the placeholder
  //    survives as `%type_of(arg<N>)` and the op is deferred.
  //    String literals pass through verbatim; `//` comments are not handled.
  llvm::DenseMap<StringRef, unsigned> nameToIdx;
  SmallVector<Value> operands;
  SmallVector<Attribute> paramAttrs;
  std::string rewritten;
  llvm::raw_string_ostream os(rewritten);
  bool operandTypeDeferred = false;

  // The callback decides each placeholder's rewrite; it signals errors (already
  // diagnosed) by setting `walkFailed` and returning -1, since it can't
  // `return` out of the enclosing function.
  bool walkFailed = false;
  KGEN::scanFStringTemplate(
      tmpl, os, [&](StringRef rest, llvm::raw_ostream &out) -> int {
        // `%{name}` / `%{type_of(name)}` — the only user-facing placeholders.
        if (rest.starts_with("{")) {
          size_t end = rest.find('}');
          if (end == StringRef::npos) {
            emitter.emitError(loc, "unterminated `%{...}` placeholder");
            walkFailed = true;
            return -1;
          }
          StringRef inner = rest.substr(1, end - 1);
          bool wantType = inner.starts_with("type_of(") && inner.ends_with(")");
          StringRef name = wantType ? inner.drop_front(8).drop_back(1) : inner;
          if (name.empty() || (!llvm::isAlpha(name[0]) && name[0] != '_') ||
              !llvm::all_of(name.drop_front(), [](char ch) {
                return llvm::isAlnum(ch) || ch == '_';
              })) {
            emitter.emitError(loc,
                              "invalid identifier in `%{...}` placeholder: '")
                << name << "'";
            walkFailed = true;
            return -1;
          }

          // Resolve once. PValue → `%param<N>` (deferred string), otherwise
          // emit an SRValue operand → `%arg<N>`. The high bit of
          // `nameToIdx[name]` tags literal vs operand so re-references share a
          // slot.
          static constexpr unsigned kParamTag = 1u << 31;
          auto it = nameToIdx.find(name);
          if (it == nameToIdx.end()) {
            DeclRefNode synthRef(name, /*isEscapedIdentifier=*/false);
            AnyValue val = emitter.emitExpr(&synthRef, EC_MLIRMagic);
            if (!val) {
              walkFailed = true;
              return -1;
            }
            if (PValue pv = val.getIfPValue()) {
              TypedAttr canon = getCanonicalAttr(pv.get());
              nameToIdx[name] =
                  static_cast<unsigned>(paramAttrs.size()) | kParamTag;
              paramAttrs.push_back(
                  ToStringDeferredAttr::get(canon, /*needElideType=*/true));
            } else {
              SRValue srv = emitter.emitSRValue({val, &synthRef}, EC_MLIRMagic,
                                                ASTType());
              if (!srv) {
                walkFailed = true;
                return -1;
              }
              Value v = emitter.emitRebindOpIfNeeded(
                  srv, getCanonicalType(Value(srv).getType()),
                  fstringOps.front()->expr->getLoc());
              if (!v) {
                walkFailed = true;
                return -1;
              }
              nameToIdx[name] = operands.size();
              operands.push_back(v);
            }
            it = nameToIdx.find(name);
          }
          unsigned encoded = it->second;
          bool isParam = (encoded & kParamTag) != 0;
          unsigned idx = encoded & ~kParamTag;

          if (wantType) {
            if (isParam) {
              emitter.emitError(loc, "`%{type_of(")
                  << name
                  << ")}` is not supported for comptime parameter placeholders";
              walkFailed = true;
              return -1;
            }
            // Parametric (`MLIRDeferredType`): keep the placeholder so the
            // elaborator can substitute after binding (bracketed index,
            // matching `%arg<N>` / `%param<N>`).
            if (sugarIsa<MLIRDeferredType>(operands[idx].getType())) {
              operandTypeDeferred = true;
              out << "%type_of(arg<" << idx << ">)";
            } else {
              operands[idx].getType().print(out);
            }
          } else if (isParam) {
            out << "%param<" << idx << ">";
          } else {
            out << "%arg<" << idx << ">";
          }
          return static_cast<int>(end + 1);
        }
        // `%arg<N>` / `%param<N>` / `%type_of(arg<N>)` are internal
        // placeholders the rewrites above own; reject them (and the unbracketed
        // `%argN`, which collides with the synthetic block args, and `%paramN`)
        // in user templates.
        if (rest.starts_with("arg<") || rest.starts_with("param<") ||
            rest.starts_with("type_of(") ||
            (rest.starts_with("arg") && rest.size() > 3 &&
             llvm::isDigit(rest[3])) ||
            (rest.starts_with("param") && rest.size() > 5 &&
             llvm::isDigit(rest[5]))) {
          emitter.emitError(
              loc, "reserved `%...` placeholder in __mlir_op template; "
                   "use `%{name}` or `%{type_of(name)}`");
          walkFailed = true;
          return -1;
        }
        return 0;
      });
  if (walkFailed)
    return {};

  // 5. Defer when anything in operand/result types is parametric, a `%param<N>`
  //    survived, or `forceDeferred` was requested. The elaborator re-runs
  //    `lowerFStringMLIROp` after binding.
  bool resultTypeDeferred = llvm::any_of(
      resultTypes, [](Type t) { return sugarIsa<MLIRDeferredType>(t); });
  bool isDeferred = forceDeferred || operandTypeDeferred ||
                    resultTypeDeferred || !paramAttrs.empty();

  auto emitR = [&](CValue value) -> AnyValue {
    return emitter.emitResult(value, &subscript, dest);
  };
  auto emitTupleOf = [&](Operation::result_range results) -> AnyValue {
    auto tupleType =
        emitter.shared.lookupBuiltinType("Tuple", emitter.declScope, loc);
    if (tupleType.isTypeCheckErrorType())
      return {};
    tupleType = tupleType.getWithoutParameters(emitter.shared);
    CallOperands callOperands(CallSyntax::kTypeCall, &subscript,
                              std::move(dest));
    for (Value r : results)
      callOperands.add({SRValue(r), &subscript});
    return emitter.emitConstructorCall(tupleType, std::move(callOperands));
  };

  if (isDeferred) {
    // Stash the rewritten template on `kgen.deferred`; the elaborator
    // detects it via `isFStringTemplate` and re-runs `lowerFStringMLIROp`
    // after concretization. Comptime parameters ride as `fstring_params`.
    OperationState dstate(subscript.getLocation(emitter),
                          DeferredOp::getOperationName());
    dstate.addOperands(operands);
    dstate.addTypes(resultTypes);
    auto deferredName =
        mlir::OperationName(DeferredOp::getOperationName(), context);
    dstate.addAttribute(DeferredOp::getOpNameAttrName(deferredName),
                        StringAttr::get(context, rewritten));
    DictionaryAttr opAttrs = DictionaryAttr::get(context, {});
    if (!paramAttrs.empty()) {
      NamedAttribute litsNA(
          StringAttr::get(context, KGEN::getFStringParamsAttrName()),
          ArrayAttr::get(context, paramAttrs));
      opAttrs = DictionaryAttr::get(context, {litsNA});
    }
    dstate.addAttribute(DeferredOp::getOpAttrsAttrName(deferredName), opAttrs);
    Operation *deferred = emitter.builder->create(dstate);

    if (deferred->getNumResults() == 0)
      return emitR(PValue(emitter.shared.getNoneAttr()));
    if (deferred->getNumResults() == 1)
      return emitR(SRValue(deferred->getResult(0)));
    return emitTupleOf(deferred->getResults());
  }

  // 6. Concrete path: lower via the shared helper now. `tmpl` is the original
  //    `%{name}` source, passed for diagnostics.
  std::string errorMsg;
  Operation *parsedOp = KGEN::lowerFStringMLIROp(
      *emitter.builder, subscript.getLocation(emitter), rewritten, operands,
      resultTypes, errorMsg, /*userTmpl=*/tmpl);
  if (!parsedOp) {
    emitter.emitError(loc) << errorMsg;
    return {};
  }

  if (hadTypeSpec && (parsedOp->getNumResults() != resultTypes.size() ||
                      !llvm::equal(parsedOp->getResultTypes(), resultTypes))) {
    emitter.emitError(loc, "_type does not match parsed op's result types");
    return {};
  }

  for (Type type : parsedOp->getResultTypes()) {
    if (!ASTType(type).isRegisterPassable(loc, emitter.shared)) {
      emitter.emitError(loc)
          << ASTType(type)
          << " cannot be returned directly from __mlir_op as it is not a "
             "'RegisterPassable' type";
      return {};
    }
  }

  if (parsedOp->getNumResults() == 0)
    return emitR(PValue(emitter.shared.getNoneAttr()));
  if (parsedOp->getNumResults() == 1)
    return emitR(SRValue(parsedOp->getResult(0)));
  (void)context;
  return emitTupleOf(parsedOp->getResults());
}

/// Emit a reference like "x[i, j]" to MLIR. These can always be
/// emitted eagerly in the LHS of an assignment because the base expression can
/// never have an inferred type.
auto SubscriptNode::emitLCVIR(ExprDest &dest, IREmitter &emitter,
                              bool isSpeculative) const -> ELVIITResult {
  // Bare `__mlir_op[...]` — new f-string subscript syntax. Intercept early
  // so we don't run through the OverloadSet / __call__ / param-binding path.
  if (auto *dre = dyn_cast<DeclRefNode>(base);
      dre && dre->spelling == "__mlir_op")
    return emitFStringSubscriptMLIROp(*this, dest, emitter);
  // Subscripting a generic function binds the parameter expressions.
  auto baseAnyValue = emitter.emitExpr(base, EC_SubscriptBase);
  if (!baseAnyValue)
    return {};

  // If the base is an inferred-base attribute (e.g. `.alpha_blended[42]`),
  // defer the whole subscript until a contextual type is available.
  if (baseAnyValue.getIfInferredBaseAttrRef())
    return emitter.emitResult(InferredBaseAttrRefUValue(this), this, dest);

  OverloadSetUValue overloads = baseAnyValue.getIfOverloadSet();

  // Is this a non-subscriptable struct instance with a __call__ method? If so,
  // bind parameters to it instead of fruitlessly trying to subscript.
  if (auto ty = baseAnyValue.getRValueTypeIfResolvable()) {
    auto hasMethod = [&](StringRef name) {
      auto ov = OverloadSet::lookup(emitter.getDeclScope(), ty, name, base,
                                    CallSyntax::kMethodCallSynthetic);
      return !ov.isNull();
    };

    if (auto structTy = dyn_cast<StructMetaType>(ty);
        (structTy && structTy.getSignature().getInputParamTypes().empty()) ||
        !structTy) {
      if (!hasMethod("__getitem__") && !hasMethod("__setitem__") &&
          !hasMethod("__getattr__")) {

        auto callOv =
            OverloadSet::lookup(emitter.getDeclScope(), ty, "__call__", base,
                                CallSyntax::kMethodCallSynthetic);

        if (!callOv.isNull()) {
          auto bindings = ParamBindings::getForDeclaredType(
              emitter.getDeclScope(), ty, this);
          assert(callOv.paramBindings.empty() &&
                 "parameter bindings should be empty at this point");
          callOv.paramBindings = std::move(bindings);
          auto callValue = OverloadSetUValue::create(std::move(callOv));
          callValue->baseValue = {baseAnyValue, base};
          overloads = std::move(callValue);
        }
      }
    }
  }

  // If the baseAnyValue has a bound callable symbol, then this is applying
  // (more?) parameter expressions to bind its parameters.
  if (overloads) {
    emitter.shared.notifyListenerOnParameterBinding(overloads->fnDecls,
                                                    rsquareLoc, operands);
    // Mutate the OverloadSet directly.  This is a bit gross, but we know we're
    // the only user of it.
    if (failed(parseParameterBindings(operands, emitter,
                                      overloads->paramBindings)))
      return {};

    return emitter.emitResult(overloads, this, dest);
  }

  // Otherwise, this must be a concrete value to be able to subscript it.
  CValue baseValue = emitter.emitCValue({baseAnyValue, base}, EC_SubscriptBase);
  if (!baseValue)
    return {};
  ASTType baseType = baseValue.getRValueType();

  if (auto value = baseValue.getIfPValue()) {
    // Check for attribute bindings to an MLIR operation.
    if (auto unboundOperator =
            sugarDynCast<UnboundMLIROperationAttr>(value.get())) {
      PValue result =
          bindAttributesToMLIROperatorCall(*this, unboundOperator, emitter);
      return emitter.emitResult(result, this, dest);
    }

    // If this is a parametric PValue, this is binding parameter values to the
    // generator value. Or if the sub-value is an unbound Type, try binding
    // parameters to it! Handle user-defined types and custom MLIR types.
    if (sugarIsa<GeneratorType>(baseType) ||
        (baseValue.getIfTypeValue() && sugarIsa<StructMetaType>(baseType))) {
      ParamBindings::BindingKind bindingKind = ParamBindings::kStandard;
      // In the form of S[xxx]() or S[xxx].field, we make it more admissible.
      if (dest.getContext() == ExprContext::EC_CallCalleeValue ||
          dest.getContext() == ExprContext::EC_AttributeRefBase)
        bindingKind = ParamBindings::kContextual;

      PValue result = [&]() -> PValue {
        if (auto sig = sugarDynCast<GeneratorType>(baseType)) {
          return bindToGeneratorValue(value, sig, this, operands, emitter,
                                      getIndexRange(), bindingKind);
        }
        return substituteParametersIntoUserDefinedType(value, this, operands,
                                                       lsquareLoc, rsquareLoc,
                                                       emitter, bindingKind);
      }();

      return emitter.emitResult(result, this, dest);
    }
  }

  // If the sub-value is an unbound Type, try binding parameters to it!
  if (Type typeValue = baseValue.getIfTypeValue()) {
    // Handle __mlir_type["foo"] and __mlir_attr["foo"].
    if (sugarIsa<MagicMLIRTypeType>(typeValue)) {
      std::string result = substituteMLIRMagic(*this, emitter);
      if (result.empty())
        return {};
      ASTType type = parseMLIRType(result, this, emitter.shared);
      return emitter.emitResult(type, this, dest);
    }
    if (sugarIsa<MagicMLIRDeferredTypeType>(typeValue)) {
      std::string str = substituteMLIRMagic(*this, emitter);
      if (str.empty())
        return {};
      // Silently try to parse the type immediately; if it succeeds the user
      // can be told that __mlir_type suffices.
      MLIRContext *ctx = emitter.getContext();
      Type parsed;
      {
        mlir::ScopedDiagnosticHandler suppress(ctx, [](Diagnostic &) {});
        parsed = mlir::parseType(str, ctx);
      }
      if (parsed) {
        emitter.emitWarning(getLoc())
            << "trivially constructable type. Use `__mlir_type` instead.";
        return emitter.emitResult(ASTType(parsed), this, dest);
      }
      // Type cannot be parsed now (parameters not yet concrete); wrap in
      // !kgen.deferred_type so the elaborator can resolve it after inlining.
      AttrCtorDeferredAttr deferredAttr =
          buildAttrCtorDeferredAttrFromMLIRAttr(*this, emitter);
      if (!deferredAttr)
        return {};
      Type deferredType = MLIRDeferredType::get(ctx, deferredAttr);
      return emitter.emitResult(ASTType(deferredType), this, dest);
    }
    if (sugarIsa<MagicMLIRAttrType, MagicMLIRDeferredAttrType>(typeValue)) {
      std::string result = substituteMLIRMagic(*this, emitter);
      if (result.empty())
        return {};
      const bool fallbackToDeferredAttr =
          sugarIsa<MagicMLIRDeferredAttrType>(typeValue);
      // When we are not allowed to fallback to deferred attrobute, report an
      // error if attribute cannot be constructed.
      PValue attr = synthesizeMLIRAttrFromString(
          result, getLoc(), emitter.shared,
          /* reportError = */ !fallbackToDeferredAttr);
#ifndef NDEBUG
      {
        // Ideally we want to build deferred attribute first and stringize it.
        // This will help us to keep everything in sync. The downside of this is
        // it will add unnecessary overhead as many attributes don't need to be
        // deferred.
        // Therefore, to make sure deferred attributes are built correctly, for
        // non-production build do extra check.
        auto deferredAttr =
            buildAttrCtorDeferredAttrFromMLIRAttr(*this, emitter);
        assert(result == getStringRepresentation(deferredAttr) &&
               "string representations of an attribute don't match");
      }
#endif // NDEBUG
      if (fallbackToDeferredAttr) {
        if (!attr) {
          auto deferredAttr =
              buildAttrCtorDeferredAttrFromMLIRAttr(*this, emitter);
          // Wrap this attribute with `#kgen.deferred` to let elaborator
          // concretize attributes that are not concrete now.
          attr = PValue(deferredAttr);
        } else {
          // If attribute can be constructed at this point, but user used
          // `__mlir_deferred_attr`, let user know that regular `__mlir_attr`
          // should be used instead.
          emitMLIRDeferredAttrToMLIRAttrWarning(getLoc(), emitter);
        }
      }
      return emitter.emitResult(attr, this, dest);
    }
  }

  // Otherwise, if there is no symbol, it is just an LValue or RValue being
  // subscript, invoking a dynamic subscript with __getitem__ and __setitem__.
  return emitGetterSetterAccess(this, {baseValue, base}, operands, dest,
                                emitter);
}

LogicalResult ParenNode::emitDestructuringPValue(PValue value,
                                                 IREmitter &emitter) const {
  return subExpr->emitDestructuringPValue(value, emitter);
}

AnyValue ParenNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  return emitter.emitExpr(subExpr, dest);
}

static AnyValue emitListLiteral(const ExprNode *expr,
                                ArrayRef<ExprNode *> elements,
                                IREmitter &emitter) {
  CallOperands operands(CallSyntax::kTypeCall, expr, EC_CollectionLiteral);
  for (ExprNode *expr : elements) {
    auto value = emitter.emitExpr(expr, EC_CollectionLiteral);
    if (!value)
      return {};
    operands.add({value, expr});
  }
  return InitializerUValue::create(InitializerUValue::kListLiteral,
                                   std::move(operands));
}

ExprNode::ELVIITResult ListLiteralNode::emitLCVIR(ExprDest &dest,
                                                  IREmitter &emitter,
                                                  bool isSpeculative) const {
  // Speculative emission is used for LHS patterns like `var [a, b] = …`. We
  // don't do anything with it, because `[var x:Int] = ...` isn't enough to
  // infer the type of the RHS anyway.  Just wait for the RHS to get emitted.
  if (isSpeculative)
    return {this};

  // If this is the LHS of an assignment, then we unpack the list pattern into
  // the subexpressions of the pattern.
  if (auto expectedType = dest.getIfInitializerType()) {
    if (sugarIsa<TypeCheckErrorType>(expectedType))
      return {};
    DLValue result(RCRef<ListPatternDLValue>::create(
        exprs, expectedType, dest.getPatternDeclKind(), this));
    return emitter.emitResult(result, this, dest);
  }

  // Normal RHS emission: `[a, b, c]` as a list literal initializer.
  return emitter.emitResult(emitListLiteral(this, exprs, emitter), this, dest);
}

LogicalResult
ListLiteralNode::emitDestructuringPValue(PValue value,
                                         IREmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot destructure into list patterns yet");
  return failure();
}

AnyValue DictLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  assert(!values.empty() && "empty syntax doesn't turn into dict literal");

  // We emit dictionary literal syntax like `{a:b, c:d}` as an initializer list
  // with a key and value list: {[a, c], [b, d], __dict_literal__: ()}. This
  // allows the rest of the compiler to infer the type of the dictionary literal
  // and infer the key/element types from the list literals recursively.
  SmallVector<ExprNode *> keyElts;
  SmallVector<ExprNode *> valueElts;
  for (auto [keyExpr, valueExpr] : values) {
    if (!keyExpr) {
      emitter.emitError(
          valueExpr->getLoc(),
          "TODO: unpack emission in dict literal not supported yet")
          << valueExpr->getRange();
      return {};
    }
    keyElts.push_back(keyExpr);
    valueElts.push_back(valueExpr);
  }

  auto keysListValue = emitListLiteral(this, keyElts, emitter);
  auto valuesListValue = emitListLiteral(this, valueElts, emitter);
  if (!keysListValue || !valuesListValue)
    return {};

  // Form the initializer list for the dictionary literal.
  CallOperands operands(CallSyntax::kTypeCall, this, EC_CollectionLiteral);
  operands.add({keysListValue, this});
  operands.add({valuesListValue, this});
  auto result = InitializerUValue::create(InitializerUValue::kDictLiteral,
                                          std::move(operands));
  return emitter.emitResult(result, this, dest);
}

AnyValue SetInitLiteralNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // We emit this as a simple initializer list directly, but resolution of the
  // UValue detects when this should be a set initializer and handles that.
  CallOperands operands(CallSyntax::kTypeCall, this, EC_CollectionLiteral);
  for (auto [keyExpr, valueExpr] : values) {
    StringAttr keyName;
    // The key needs to be an identifier, which gets parsed as a DeclRefNode.
    if (auto dre = dyn_cast_or_null<DeclRefNode>(keyExpr)) {
      keyName = StringAttr::get(emitter.getContext(), dre->spelling);
    } else if (keyExpr) {
      emitter.emitError(keyExpr->getLoc(),
                        "expected identifier in initializer list");
      return {};
    }
    auto value = emitter.emitExpr(valueExpr, EC_CollectionLiteral);
    if (!value)
      return {};
    operands.add(keyName, {value, valueExpr},
                 keyName ? ArgUnpackStyle::kKeyword
                         : ArgUnpackStyle::kPositional);
  }

  auto result = InitializerUValue::create(InitializerUValue::kSetInitLiteral,
                                          std::move(operands));
  return emitter.emitResult(result, this, dest);
}

/// Given an operator, return the SpecialFunctionInfo that implements it.
static SpecialFunctionInfo getOpSpecialFunctions(ExprNode::Kind kind,
                                                 bool isReversed) {

  // Use an if chain to find the right match.  We can't use switch here because
  // multiple special functions may implement the same kind, e.g. __add__ and
  // __radd__ special methods both implement kAdd.
#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS)              \
  if (kind == ExprNode::Kind::EXPRNODE) {                                      \
    auto info = SpecialFunctionInfo::get(SpecialFunctionKind::ENUM);           \
    if (info.isReversed() == isReversed)                                       \
      return info;                                                             \
  }
#include "Mojo/LITDialect/SpecialFunctions.def"
  // If everything fails we should return "normal".
  return SpecialFunctionInfo::get(SpecialFunctionKind::kNormal);
}

/// Emit the binary operation (with a `lhs`, `rhs` and `kind`) as a special
/// function call.
/// A special function call is one where the` kind` must corresponds to a valid
/// SpecialFunctionInfo when we invoke getOpSpecialFunctions(kind).
/// `callExpr` is the call like expression that results in the call.
//
/// This is an utility function to share code between BinOpNone and
/// ChainedCmpOpNode since the latter is a sequence of binary operations.
static AnyValue emitBinOpCall(ASTExprAnd<AnyValue> lhs,
                              ASTExprAnd<AnyValue> rhs, ExprNode::Kind kind,
                              ExprDest &dest, const ExprNode *callExpr,
                              IREmitter &emitter) {

  // Sugar for Type1 == Type2
  if (kind == ExprNode::Kind::kCmpEQ || kind == ExprNode::Kind::kCmpNE) {
    PValue lhsPV = lhs.ir.getIfPValue();
    PValue rhsPV = rhs.ir.getIfPValue();
    // TODO: maybe we can extend this to any type value at the same depth, for
    // now, just handle the most common case.
    if (lhsPV && rhsPV && LIT::isFirstLevelTypeExpr(lhsPV) &&
        LIT::isFirstLevelTypeExpr(rhsPV)) {
      TraitType anyTypeTrait =
          emitter.shared.lookupBuiltinTraitType("AnyType", callExpr->getLoc());
      // First upcast the type value to AnyType, this ensures us we encode mlir
      // type consistently.
      FailureOr<PValue> lhsOr =
          emitter.emitTypeValueUpCastToTrait({lhsPV, lhs.expr}, anyTypeTrait);
      FailureOr<PValue> rhsOr =
          emitter.emitTypeValueUpCastToTrait({rhsPV, rhs.expr}, anyTypeTrait);
      // A null result means the upcast diagnosed an error (e.g. a non-concrete
      // type value), and `failure()` means it does not apply at all.
      assert(succeeded(lhsOr) && succeeded(rhsOr));
      if (lhsOr->isNull() || rhsOr->isNull())
        return {};

      PValue res = ParamIdenticalAttr::get(*lhsOr, *rhsOr);
      if (kind == ExprNode::Kind::kCmpNE)
        res = ParamOperatorAttr::getNot(res);
      return emitter.emitBool({res, callExpr}, dest);
    }
  }

  // If this is a 'not in' emit the 'in' expression and then invert the result.
  //  We use this style to make sure that a direct emission emits into
  // the ExprDest directly.
  if (kind == ExprNode::Kind::kCmpNotIn) {
    ExprDest inDest(EC_OperatorOperandValue);
    auto inResult = emitBinOpCall(lhs, rhs, ExprNode::Kind::kCmpIn, inDest,
                                  callExpr, emitter);
    return UnaryOpNode::emitArith(ExprNode::Kind::kBoolNot, callExpr,
                                  {inResult, callExpr}, dest, emitter);
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/false);
  if (specialFnInfo.kind == SpecialFunctionKind::kNormal) {
    // This means that the operator is not defined in SpecialFunctions.def.
    emitter.shared.emitError(callExpr->getLoc(), "operator not yet supported");
    return {};
  }

  // Use one 'operands' set for the arg values even though we switch them back
  // and forth.  Resolving a set can mutate the argument list (e.g. emitting
  // PValues to dynamic values) even if the lookup fails and we don't want to
  // materialize them multiple times
  CallOperands operands(CallSyntax::kOperator, callExpr, std::move(dest),
                        {lhs, rhs});

  // `a in b` => `b.__contains__(a)` and there is no reversed form.
  if (kind == ExprNode::Kind::kCmpIn)
    std::swap(operands[0], operands[1]);

  // Check to see if we have a forward version of this function on the primary
  // receiver.
  if (auto lhsCV = lhs.ir.getIfCValue()) {
    if (PValue callee = OverloadSet::lookupAndResolve(
            lhsCV.getRValueType(), specialFnInfo.name, operands, emitter))
      return emitter.emitIndirectCall(callee, std::move(operands));
  }

  // Check to see if we have the reverse version of this operator.
  auto reversedFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/true);
  if (reversedFnInfo.kind != SpecialFunctionKind::kNormal) {
    // Swap the operand order.
    std::swap(operands[0], operands[1]);
    operands.syntax = CallSyntax::kReversedOperator;
    if (auto rhsCV = rhs.ir.getIfCValue()) {
      if (PValue callee = OverloadSet::lookupAndResolve(
              rhsCV.getRValueType(), reversedFnInfo.name, operands, emitter))
        return emitter.emitIndirectCall(callee, std::move(operands));
    }

    // Swap these back so we emit the right error.
    std::swap(operands[0], operands[1]);
    operands.syntax = CallSyntax::kOperator;
  }

  // Emit an error complaining about the forward version of the operator.
  return emitter.emitNamedMethodCall(specialFnInfo.name, std::move(operands));
}

/// Emit a simple assignment statement.
AnyValue BinOpNode::emitAssign(ExprDest &dest, IREmitter &emitter) const {
  // Assignments might need to infer the LHS from the RHS when the LHS is
  // unresolved, and the RHS from the LHS when it is known:
  //
  //    _, x = foo()            # infer typeof _ and x from foo()
  //    y = []                  # infer type of [] from type of y.
  //    y : List[_] = [1, 2, 3] # infer type of List.T from type of [1, 2, 3].
  //
  // This is handled by speculatively emitting the LHS to see if it has a
  // context-free known LValue, and resolve the RHS to it if so.
  //
  // Note: Python always emits the RHS before the LHS, as seen in things like:
  //
  //     def test1(): print("test1"); return 0
  //     def test2(): print("test2"); return 1
  //     a[test1()] = test2()
  //   ==> test2; test1
  //

  // Check out the LHS speculatively.
  ELVIITResult lhsResult = lhs->emitLValueIfImplicitlyTyped(
      emitter, PatternDeclKind::kNone, /*hasInferrableRHS=*/true);
  if (lhsResult.isFailure())
    return {};

  if (AnyValue av = lhsResult.getIfValue()) {
    // If this is an RValue, emit an error so ExprDest doesn't panic.
    if (!av.getIfLValue()) {
      emitter.emitError(lhs->getLoc(),
                        "expression must be mutable in assignment")
          << lhs->getRange();
      return {};
    }
  }

  // Figure out if the LHS is syntactically a var pattern, to tune diagnostics.
  auto isVarPat = [&](ExprNode *expr) -> bool {
    while (1) {
      if (expr->kind == kVarPat)
        return true;
      if (auto paren = dyn_cast<ParenNode>(expr)) {
        expr = paren->subExpr;
        continue;
      }
      if (expr->kind == kTypePattern) {
        expr = cast<BinOpNode>(expr)->lhs;
        continue;
      }
      return false;
    }
  };

  // Generate tailored messages when the LHS is a "var x" pattern.
  auto assignDestKind = isVarPat(lhs) ? EC_VarInit : EC_Assignment;

  // The RHS will be assigned into either the expression returned (resolving
  // it from the type of the RHS) or from the LValue returned (allowing the
  // RHS to infer from it) based on the lhsResult.
  ExprDest assignDest(assignDestKind);
  assert(!lhsResult.isFailure() && "Failures should be handled");
  if (AnyValue av = lhsResult.getIfValue()) {
    assignDest = ExprDest(av.getIfLValue(), assignDestKind);
  } else if (lhsResult.getIfExprNode()) {
    assignDest = ExprDest(lhsResult.getIfExprNode(), assignDestKind);
  } else {
    assignDest = ExprDest(lhsResult.getIfPartiallyBoundLV(), assignDestKind);
  }

  // Emit the RHS into the context of the LHS.  If we got an LValue, then we can
  // infer the type of the RHS from the LHS LValue.  If we got an unresolved
  // LHS, then we can resolve it from the RHS.  If neither can decided then we
  // have an ambiguity.
  auto resultValue = emitter.emitExpr(rhs, assignDest);
  if (!resultValue) {
    // If emitting the RHS failed, use a "type check error" expression as the
    // RHS so we can make sure to emit any vars declared, to silence downstream
    // errors.
    //     var x = <bad>
    //     use(x)  # Don't warn here.
    resultValue = UnknownAttr::get(emitter.shared.getTypeCheckErrorType());
  }

  // Support chained assignment like `x = y = 1`: the assignment operation
  // returns a borrowed version of the dest value.
  return emitter.emitResult(resultValue, this, dest);
}

/// Emit a walrus assignment expression like `x := y`.
///
/// Unlike general assignment, walrus does not use speculative bidirectional
/// type inference: the LHS is emitted directly as an LValue, then the RHS is
/// stored into it. It yields the RHS, whichever kind of LValue the target is.
AnyValue BinOpNode::emitWalrus(ExprDest &dest, IREmitter &emitter) const {
  LValue lhsLV = emitter.emitExprLValue(lhs, EC_Assignment);
  if (!lhsLV)
    return {};

  // Mutable memory LValue: store into it, then yield the stored value.
  if (lhsLV.getIfMLValue()) {
    ExprDest assignDest(lhsLV, EC_Assignment);
    auto resultValue = emitter.emitExpr(rhs, assignDest);
    return emitter.emitResult(resultValue, this, dest);
  }

  // Otherwise must be a DLValue.
  if (!lhsLV.getIfDLValue()) {
    emitter.emitError(lhs->getLoc(),
                      "walrus operator target must be a mutable memory "
                      "location");
    return {};
  }

  // Computed LValue (e.g. subscript / attribute): the setter takes the
  // right-hand side, which is also what the walrus yields.
  auto rhsRValue =
      emitter.emitExprRValue(rhs, EC_Assignment, lhsLV.getRValueType());
  if (!rhsRValue)
    return {};

  // The logic below won't work for SRValue's (because emitBValue takes
  // ownership of the value), so promote to MRValue if we have one.
  if (rhsRValue.getIfSRValue())
    rhsRValue = emitter.emitMRValue({rhsRValue, rhs}, EC_Assignment);

  // A setter taking a 'var' argument would consume the value we return.
  BValue bValue = emitter.emitBValue({rhsRValue, rhs}, EC_Assignment);
  if (!bValue)
    return {};
  if (!emitter.emitStoreToLValue({bValue, rhs}, lhsLV, EC_Assignment))
    return {};
  return emitter.emitResult(rhsRValue, this, dest);
}

/// Emit a inplace assignment statement like `x += y`. Python evaluates the RHS
/// of an assignment before the LHS, as seen in things like:
///    def test1(): print("test1"); return 0
///    def test2(): print("test2"); return 1
///    a[test1()] += test2()
///  ==> test1; test2
AnyValue BinOpNode::emitInplace(ExprDest &dest, IREmitter &emitter) const {
  // Inplace operations evaluate the LHS first.
  LValue lhsLV = emitter.emitExprLValue(lhs, EC_InplaceBinOpDest);
  if (!lhsLV)
    return {};
  // Then emit the right side.
  AnyValue rhsV = emitter.emitExpr(rhs, EC_OperatorOperandValue);
  if (!rhsV)
    return {};

  // Emit the call to the operator function like `__iadd__`.
  return emitBinOpCall({lhsLV, lhs}, {rhsV, rhs}, kind, dest, this, emitter);
}

ExprNode::ELVIITResult BinOpNode::emitLValueIfImplicitlyTyped(
    IREmitter &emitter, PatternDeclKind patKind, bool hasInferrableRHS) const {
  if (kind != kTypePattern)
    return this;

  // Emit the RHS as a type expression.
  ASTType type = emitter.emitExprType(rhs, hasInferrableRHS);
  if (!type)
    return {};

  if (hasInferrableRHS)
    return LValueContextualType{type, lhs};

  // Handle type patterns like "(xyz) : Type": "xyz" must be an lvalue that has
  // the specified type, so we can direct emit it now.
  ExprDest dest(LValueInitializerType{type}, EC_TypePattern);
  dest.setPatternDeclKind(patKind);
  if (auto lv = emitter.emitExprLValue(lhs, dest))
    return AnyValue(lv);
  return {}; // Failure emitting the LValue.
}

AnyValue BinOpNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  if (kind == kAsPat) {
    emitter.emitError(getLoc(), "'as' patterns are only valid in a 'case' "
                                "clause");
    return {};
  }

  // Handle weird binary operators specially if we have them.
  if (kind == kBoolAnd || kind == kBoolOr) // `x and y`, `x or y`
    return emitAndOr(dest, emitter);
  if (kind == kAssign) // `x = y`
    return emitAssign(dest, emitter);
  if (kind == kWalrus) // `x := y`
    return emitWalrus(dest, emitter);
  if (isAssignmentStmt()) // `x += y`
    return emitInplace(dest, emitter);

  if (kind == kTypePattern) { // "x: Type" not in an LValue position?
    auto result = emitLValueIfImplicitlyTyped(
        emitter, dest.getPatternDeclKind(), /*hasInferrableRHS=*/false);
    if (result.isFailure())
      return {};
    assert(result.getIfValue() && "Failed to resolve value?");
    return emitter.emitResult(result.getIfValue(), this, dest);
  }

  // Otherwise we emit the LHS followed by the RHS.
  AnyValue lhsRV = emitter.emitExpr(lhs, EC_OperatorOperandValue);
  AnyValue rhsRV = emitter.emitExpr(rhs, EC_OperatorOperandValue);
  if (!lhsRV || !rhsRV)
    return {};

  // Emit trait composition if both operands have AnyTrait types.
  if (kind == kAnd) {
    auto lhsTrait =
        sugarDynCastIfPresent<AnyTraitType>(lhsRV.getRValueTypeIfResolvable());
    auto rhsTrait =
        sugarDynCastIfPresent<AnyTraitType>(rhsRV.getRValueTypeIfResolvable());
    if (lhsTrait && rhsTrait) {
      SmallVector<TraitSymbolAttr> symbols(
          lhsTrait.getTraitType().getSymbols());
      llvm::append_range(symbols, rhsTrait.getTraitType().getSymbols());
      sortAndDeduplicateTraitSymbols(symbols);
      if (llvm::equal(symbols, lhsTrait.getTraitType().getSymbols())) {
        emitter.emitWarning(getLoc())
            << "redundant trait composition: "
            << ASTType(lhsTrait.getTraitType()) << " already implies "
            << ASTType(rhsTrait.getTraitType());
      } else if (llvm::equal(symbols, rhsTrait.getTraitType().getSymbols())) {
        emitter.emitWarning(getLoc())
            << "redundant trait composition: "
            << ASTType(rhsTrait.getTraitType()) << " already implies "
            << ASTType(lhsTrait.getTraitType());
      }
      return emitter.emitResult(TraitType::get(emitter.getContext(), symbols),
                                this, dest);
    }
  }

  return emitBinOpCall({lhsRV, lhs}, {rhsRV, rhs}, kind, dest, this, emitter);
}

/// This method emits the `x and y`, `x or y` operators.  These are
/// interesting in Python:
///
///   "Note that neither `and` nor `or` restrict the value and type they
///   return to False and True, but rather return the last evaluated argument.
///   This is sometimes useful, e.g., if `s` is a string that should be
///   replaced by a default value if it is empty, the expression `s or 'foo'`
///   yields the desired value.
///
/// Unlike Python, we have static types that could disagree.  Our policy on
/// this is to either return the pre-Bool'ified value when their types agree (or
/// can be converted to each other unambiguously) or to return the common Bool
/// type if they don't.
///
AnyValue BinOpNode::emitAndOr(ExprDest &dest, IREmitter &emitter) const {
  Location ifLoc = getLocation(emitter);

  // Emit the LHS value as a bool/i1 value.
  CValue lhsV = emitter.emitExprCValue(lhs, EC_OperatorOperandValue);

  // If the lhs is an RValue, decay to BValue before passing into emitI1,
  // because we can't have emitI1 consume it.
  if (lhsV.getIfSRValue() &&
      !lhsV.getRValueType().isTrivial(getLoc(), emitter.shared)) {
    // We cannot convert an SRValue directly to an SBValue because the latter
    // doesn't track lifetimes back to the original value correctly. Instead,
    // decay the SRValue to an MRValue first, which does.
    lhsV = emitter.emitMRValue({lhsV, lhs}, EC_OperatorOperandValue);
  }
  CValue lhsBVal = lhsV;
  if (auto mrVal = lhsV.getIfMRValue())
    lhsBVal = MBValue(mrVal);

  RValue lhsBoolVal =
      emitter.emitScalarBool({lhsBVal, lhs}, EC_OperatorOperandValue);
  PValue lhsBoolPVal = lhsBoolVal.getIfPValue();

  if (!emitter.builder) {
    lhsBoolPVal = emitter.emitPValue({lhsBoolVal, lhs}, EC_BoolCondition);
    if (!lhsBoolPVal)
      return {};
    CValue rhsV = emitter.emitExprCValue(rhs, EC_BoolCondition);

    // Try to coerce the true/false values into a compatible type if they
    // disagree. Pass a null SMLoc so no error is emitted on failure.
    if (emitter.coerceTypesToEachOther(SMLoc(), lhsV, lhs, rhsV, rhs, {})) {
      // The two types are incompatible (e.g. `comptime if someInt and
      // someString`). Fall back to Bool for both sides, matching the runtime
      // `and`/`or` behavior.
      ASTType boolType =
          emitter.shared.lookupBuiltinType("Bool", emitter.declScope, getLoc());

      if (!rhsV.getRValueType().isEqualCanon(boolType)) {
        RValue rhsI1Value =
            emitter.emitScalarBool({rhsV, rhs}, EC_OperatorOperandValue);
        rhsV = emitter.emitCValue({rhsI1Value, rhs}, EC_OperatorOperandValue,
                                  boolType);
      }

      if (!lhsV.getRValueType().isEqualCanon(boolType)) {
        lhsV = emitter.emitCValue({AnyValue(lhsBoolVal), lhs},
                                  EC_OperatorOperandValue, boolType);
      }

      if (!lhsV || !rhsV)
        return {};
    }

    PValue lhsPV = emitter.emitPValue({lhsV, lhs}, EC_OperatorOperandValue);
    PValue rhsPV = emitter.emitPValue({rhsV, rhs}, EC_OperatorOperandValue);
    if (!lhsPV || !rhsPV)
      return {};

    if (kind == kBoolOr) // and/or swap true/false operands
      std::swap(lhsPV, rhsPV);

    auto value = ParamOperatorAttr::get(POC::Cond, {lhsBoolPVal, rhsPV, lhsPV});
    return emitter.emitResult(value, this, dest);
  }

  SRValue lhsI1SRValue =
      emitter.emitSRValue({AnyValue(lhsBoolVal), lhs}, EC_BoolCondition);
  if (!lhsI1SRValue)
    return {};

  auto ifOp = HLCF::ElifOp::create(*emitter.builder, ifLoc,
                                   TypeRange{lhsV.getType()}, lhsI1SRValue);
  emitter.builder->createBlock(&ifOp.getThenRegion());
  emitter.builder->createBlock(&ifOp.getElseRegion());

  OpBuilder trueBuilder = ifOp.getThenBodyBuilder();
  OpBuilder falseBuilder = ifOp.getElseBodyBuilder();
  if (kind == kBoolOr) // and/or just treat the bool differently.
    std::swap(trueBuilder, falseBuilder);

  emitter.builder = trueBuilder;
  CValue rhsV = emitter.emitExprCValue(rhs, EC_BoolCondition);
  if (!rhsV)
    return {};

  /// If the types disagree, then we need to emit a conversion to a common
  /// type. See if one is convertible to the other, and if so, emit a
  /// conversion to get to a common type.
  auto configEmitter = [&](bool isLHS) {
    emitter.builder = isLHS ? falseBuilder : trueBuilder;
  };
  // Try to find compatibility between the raw values.  Pass in a null SMLoc
  // so that an error isn't diagnosed with an error message.
  if (emitter.coerceTypesToEachOther(SMLoc(), lhsV, lhs, rhsV, rhs,
                                     configEmitter)) {
    // If the two types are incompatible or ambiguously convertible to each
    // other, then the user wrote something like `if someInt and someString`.
    // This has no common type to return, but the result should still be
    // boolean-ish.  Handle this by extracting the boolean result out of the
    // second argument and converting that to a proper Bool result.
    ASTType boolType =
        emitter.shared.lookupBuiltinType("Bool", emitter.declScope, getLoc());

    // If the RHS is already a Bool, we're good, otherwise convert to i1 then
    // back to Bool with a ctor.
    if (!rhsV.getRValueType().isEqualCanon(boolType)) {
      RValue rhsI1Value =
          emitter.emitScalarBool({rhsV, rhs}, EC_OperatorOperandValue);
      emitter.builder = trueBuilder;
      rhsV = emitter.emitCValue({rhsI1Value, rhs}, EC_OperatorOperandValue,
                                boolType);
    }

    // Similarly, if the LHS was already a Bool then use it, otherwise convert
    // the i1 we already have back to Bool with a ctor.
    if (!lhsV.getRValueType().isEqualCanon(boolType)) {
      emitter.builder = falseBuilder;
      lhsV = emitter.emitCValue({lhsI1SRValue, lhs}, EC_OperatorOperandValue,
                                boolType);
    }

    if (!lhsV || !rhsV)
      return {};
  }

  // Detect unreachable code and warn about it. This is called after both arms
  // are emitted.
  auto deadCodeCheck = [&]() {
    if (!lhsBoolPVal)
      return;
    auto asBoolAttr = sugarDynCast<SIMDAttr>(lhsBoolPVal.get());
    if (!asBoolAttr)
      return;
    bool isZero = !asBoolAttr.getAsBool();
    if (kind == kBoolOr && !isZero) {
      emitter.emitWarning(this->getLoc())
          << "unreachable code on right side of 'True or ...'";
      markRegionUnreachable(&ifOp.getElseRegion(), ifOp.getLoc());
    } else if (kind == kBoolAnd && isZero) {
      emitter.emitWarning(this->getLoc())
          << "unreachable code on right side of 'False and ...'";
      markRegionUnreachable(&ifOp.getThenRegion(), ifOp.getLoc());
    } else {
      // This has no dead code, but let's still warn about a constant branch
      // condition.
      emitter.emitWarning(this->getLoc())
          << "constant value on left side of '" << (isZero ? "False" : "True")
          << " " << (kind == kBoolOr ? "or" : "and") << " ...'";
    }
  };

  // Now we know they have common types.
  auto resultType = lhsV.getRValueType();
  if (resultType.isRegisterPassable(lhs->getLoc(), emitter.shared)) {
    emitter.builder = trueBuilder;
    auto rhsSR = emitter.emitSRValue({rhsV, rhs}, EC_OperatorOperandValue);
    if (!rhsSR)
      return {};
    HLCF::YieldOp::create(*emitter.builder, ifLoc, rhsSR);
    // Emit the false side.
    emitter.builder = falseBuilder;
    auto lhsSR = emitter.emitSRValue({lhsV, rhs}, EC_OperatorOperandValue);
    if (!lhsSR)
      return {};
    HLCF::YieldOp::create(*emitter.builder, ifLoc, lhsSR);
    ifOp->getResult(0).setType(lhsSR.getType());
    emitter.builder->setInsertionPointAfter(ifOp);
    deadCodeCheck();
    return emitter.emitResult(SRValue(ifOp.getResult(0)), this, dest);
  }

  // If we have a memory only type, we have to handle the various issues with
  // the ExprDest.  It may specify an MLValue to emit into, it may be
  // ambiguous (like a call argument) or it may even be something like a
  // DLValue.  We handle this by projecting the ExprDest to an MLValue if we
  // can, but otherwise using a scratch buffer if not.
  emitter.builder->setInsertionPoint(ifOp);
  MLValue destBuffer = dest.getMLValueForResult(getLoc(), resultType, emitter);

  emitter.builder = falseBuilder;
  ExprDest falseDest(destBuffer, EC_CondExpr);
  (void)emitter.emitResult(lhsV, lhs, falseDest);
  HLCF::YieldOp::create(*emitter.builder, ifLoc);

  emitter.builder = trueBuilder;
  ExprDest trueDest(destBuffer, EC_CondExpr);
  (void)emitter.emitResult(rhsV, rhs, trueDest);
  HLCF::YieldOp::create(*emitter.builder, ifLoc);

  // MemoryOnly results don't need the 'elif' result.  There is no way to remove
  // results after creating it, so we create a new ElifOp and move IR over.
  emitter.builder->setInsertionPointAfter(ifOp);
  auto newIfOp =
      HLCF::ElifOp::create(*emitter.builder, ifLoc, TypeRange{}, lhsI1SRValue);
  deadCodeCheck();
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp->erase();

  return emitter.emitCResult(MRValue(destBuffer), this, dest);
}

/// Emit the x^ expression.
AnyValue UnaryOpNode::emitTransfer(AnyValue argValue, ExprDest &dest,
                                   IREmitter &emitter) const {
  auto loc = getLoc();
  // We don't track ownership in parameter expressions, because we don't have
  // lifetimes (e.g. don't run destructors). As such, transferring in a
  // parameter context isn't useful, just disallow it.
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(
        loc, "cannot transfer a value in a parameter context");

  // If the input is already an owned RValue, then there is no need to
  // transfer from the temporary.
  if (argValue.getIfRValue()) {
    if (argValue.getIfPValue()) {
      emitter.emitError(loc, "cannot transfer from a parameter expression; did "
                             "you want to introduce a local 'var'?");
      return {};
    }

    emitter.emitWarning(loc)
        << "transfer from an owned value has no effect and can be removed"
        << FixIt::remove(loc);
    return emitter.emitResult(argValue, this, dest);
  }

  // We don't support transferring from trivial values, since this won't end the
  // origin. CheckLifetimes doesn't and can't track these things because they
  // don't have consume operators, move operators, etc.
  if (CValue argCValue = argValue.getIfCValue();
      argCValue && argCValue.getRValueType().isTrivial(loc, emitter.shared)) {
    emitter.emitWarning(loc)
        << "transfer from a value of trivial register type "
        << argCValue.getRValueType() << " has no effect and can be removed"
        << FixIt::remove(loc);
    return emitter.emitResult(argValue, this, dest);
  }

  // The operand value must be in memory to have a origin.
  if (!argValue.isMValue()) {
    if (argValue.getIfSBValue()) {
      emitter.emitError(loc, "expression is a register value, "
                             "transfer requires ownership");
      return {};
    }
    // Note: a DLValue like a[i] could be copied into a local value, but that
    // would almost always return an RValue which need not be transferred
    // anyway.
    emitter.emitError(loc, "expression does not live in a memory location, so "
                           "it need not be transferred");
    return {};
  }

  // The transfer expression expects the result to be a ownable value that it
  // can launder into an RValue.
  Value value = argValue.getMValueReference();

  // Origin checking needs to understand this value or field.
  Value trackableValue;
  if (value)
    trackableValue = OriginTrackable::findUnderlyingValueFromField(value);
  if (!trackableValue) {
    emitter.emitError(loc,
                      "expression does not designate a value with an origin");
    return {};
  }

  // If the memory type isn't mutable, then we can't transfer out of it.
  if (!cast<RefType>(value.getType()).isMutableKnown(true)) {
    emitter.emitError(loc, "cannot transfer out of immutable reference");
    return {};
  }

  // Make sure the origin of the value is extended to at least here.  This
  // is a use, and the `_ = x^` pattern to extend the origin of something is
  // very common.
  OwnershipUseOp::create(*emitter.builder, getLocation(emitter), value);

  // For memory values, we can just treat the value as an MRValue, and whoever
  // consumes this can consume it directly.
  return emitter.emitResult(MRValue(value), this, dest);
}

ExprNode::ELVIITResult
UnaryOpNode::emitLValueIfImplicitlyTyped(IREmitter &emitter,
                                         PatternDeclKind parentKind,
                                         bool hasInferrableRHS) const {
  // Most unary operators are never LValues, so don't speculatively resolve
  // them.
  if (kind != kVarPat && kind != kRefPat)
    return this;

  // Warn if this is a recursively nested specifier like "var ref x".
  if (parentKind != PatternDeclKind::kNone) {
    emitter.emitWarning(getLoc()) << "nested 'var' or 'ref' patterns are "
                                     "redundant, remove the outer pattern";
    return this;
  }

  // This should never be possible to resolve because the 'var' or 'ref' should
  // only influence the type of implicitly declared variables which cannot be
  // speculatively resolved.  If we did speculatively resolve it, then this is
  // an unneeded marker, e.g. "(var _) = x"
  auto patKind =
      kind == kVarPat ? PatternDeclKind::kVar : PatternDeclKind::kRef;
  auto result =
      subExpr->emitLValueIfImplicitlyTyped(emitter, patKind, hasInferrableRHS);
  if (result.isFailure())
    return result;

  // If we did resolve it, then we need to emit a warning.
  if (result.getIfValue()) {
    emitter.emitWarning(getLoc())
        << (kind == kVarPat ? "'var'" : "'ref'")
        << " pattern didn't declare a new variable, it can be removed";
    return result;
  }

  auto *newSubExpr = result.getIfExprNode();
  if (newSubExpr == subExpr)
    return this;

  return emitter.shared.allocPersistent<UnaryOpNode>(
      kind, opLoc, const_cast<ExprNode *>(newSubExpr));
}

AnyValue UnaryOpNode::emitComptime(ExprDest &dest, IREmitter &emitter) const {
  // Reject comptime if already here, this reduces confusion and cruft.
  if (!emitter.builder) {
    emitter.emitError(getLoc(), "expression is already evaluated at compile "
                                "time; remove 'comptime' keyword")
        << FixIt::remove(getLoc());
    return {};
  }

  PValue subPVal = emitter.emitExprPValue(subExpr, dest.getContext());
  return emitter.emitResult(subPVal, this, dest);
}

AnyValue UnaryOpNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // var/ref patterns are special unary operators that affect their enclosing
  // lvalue.  They are not valid on the right side of an assignment.
  if (kind == kVarPat || kind == kRefPat) {
    if (dest.getPatternDeclKind() != PatternDeclKind::kNone &&
        dest.getPatternDeclKind() != PatternDeclKind::kBind) {
      emitter.emitWarning(getLoc()) << "nested 'var' or 'ref' patterns are "
                                       "redundant, remove the outer pattern";
      return {};
    }

    dest.setPatternDeclKind(kind == kVarPat ? PatternDeclKind::kVar
                                            : PatternDeclKind::kRef);
    auto result = emitter.emitExpr(subExpr, dest);

    if (result && !result.getIfLValue()) {
      emitter.emitError(getLoc())
          << (kind == kVarPat ? "'var'" : "'ref'")
          << " patterns are only valid on the left side of an assignment";
      return {};
    }
    return result;
  }

  if (kind == kComptime)
    return emitComptime(dest, emitter);

  auto exprRep = emitter.emitExpr(subExpr, EC_OperatorOperandValue);
  if (!exprRep)
    return {};

  if (kind == kTransfer)
    return emitTransfer(exprRep, dest, emitter);

  if (kind == kUnpack) {
    // Unpacks are only allowed inside parameter or argument lists.
    emitter.emitError(getLoc(), "can't use starred expression here")
        << getRange();
    return {};
  }

  return emitArith(kind, this, {exprRep, subExpr}, dest, emitter);
}

/// Emit a unary arithmetic operation as a dynamic expression.
AnyValue UnaryOpNode::emitArith(Kind kind, const ExprNode *expr,
                                ASTExprAnd<AnyValue> argValue, ExprDest &dest,
                                IREmitter &emitter) {
  if (!argValue.ir)
    return {};

  if (kind == kBoolNot) {
    // Turn this into a call to __bool__.
    argValue.ir = emitter.emitNamedMethodCall(
        "__bool__", CallOperands(CallSyntax::kMethodCall, expr,
                                 EC_OperatorOperandValue, argValue));
    if (!argValue.ir)
      return {};
    // Now that we know we bool-ized the expression, invert it with ~.
    return emitArith(kInvert, expr, argValue, dest, emitter);
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");

  return emitter.emitNamedMethodCall(
      specialFnInfo.name,
      CallOperands(CallSyntax::kOperator, expr, std::move(dest), argValue));
}

AnyValue IfElseOpNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  RValue condRVal = emitter.emitExprScalarBool(condExpr, EC_BoolCondition);
  Location ifLoc = getLocation(emitter);

  // This function is used to get a CValue for an operand, inferring the type of
  // a UValue from the other operand if not present.  This allows us to handle
  // things like "Int() if cond else {}".
  auto emitToCValueInferringType = [&](ASTExprAnd<AnyValue> value,
                                       AnyValue otherValue) -> CValue {
    // If we already have a CValue, leave it alone so we get coercion between
    // two concrete types.
    if (auto cVal = value.ir.getIfCValue())
      return cVal;

    // If we have a contextual type, use it.
    if (auto expectedType = dest.getExpectedTypeIfSpecified())
      return emitter.emitCValue(value, EC_CondExpr, expectedType);

    // If the other operand is a CValue, use its type.
    if (auto otherCVal = otherValue.getIfCValue())
      return emitter.emitCValue(value, EC_CondExpr, otherCVal.getRValueType());
    // Otherwise just emit it to get an error message.
    return emitter.emitCValue(value, EC_CondExpr);
  };

  // Returns true if `origin` is defined in a block that dominates `anchorOp`
  // (i.e., `anchorOp` or one of its ancestors lives in the same block as the
  // origin). Used to decide whether it is safe to union two MValue lifetimes
  // across an if-like op.
  auto isAcceptableSource = [&](Value origin, Operation *anchorOp) -> bool {
    // Strip off GERs and Rebinds and RefImmutOp, and get the block that
    // defines the operation or block argument.
    origin = OriginTrackable::findUnderlyingValueFromField(origin);
    if (!origin)
      return false;
    Block *originBlock = origin.getParentBlock();
    Operation *curOp = anchorOp;
    // Scan up the region tree.
    do {
      if (curOp->getBlock() == originBlock)
        return true; // Found a dominating block containing the origin.
      curOp = curOp->getParentOp();
    } while (curOp);
    return false;
  };

  // Handles the "both sides are MValues" case for an if-like op (HLCF::ElifOp
  // or ParamIfOp). `yieldValue(v)` emits the branch terminator that yields
  // the converted SSA value.
  auto handleTwoMValuesForIfOp =
      [&](Operation *ifLikeOp, CValue trueVal, CValue falseVal,
          function_ref<void(Value)> yieldValue) -> AnyValue {
    // This only applies to things that are already memory references.
    if (!falseVal.isMValue() || !trueVal.isMValue())
      return {};

    // If both operands are MRValues then we can move the value into the
    // destination instead of forming a reference that requires a copy. Maintain
    // RValues.
    if (falseVal.getIfMRValue() && trueVal.getIfMRValue())
      return {};

    // See if the true and false values directly union together.  This
    // requires the rvalue types to be the same but allows the reference
    // types to be different.
    RefType commonRefType = emitter.getCommonRefType(falseVal.getMValueType(),
                                                     trueVal.getMValueType());
    if (!commonRefType)
      return {};

    // Check to see if the two values dominate the 'if'.  We don't want to
    // form a union'ed origin that includes an origin for something in the
    // else block, like an RValue temporary.  Such things will require a copy.
    //
    // The ideal thing to do would be to have use-def chains on the origin
    // itself, which would allow us to handle ref results from functions, but
    // we don't have that.  Instead, find the underlying values and see if we
    // can reason about them from the IR tree.
    Value falseMVal = falseVal.getMValueReference();
    Value trueMVal = trueVal.getMValueReference();
    if (!isAcceptableSource(trueMVal, ifLikeOp) ||
        !isAcceptableSource(falseMVal, ifLikeOp))
      return {};

    // Ok, at this point we are committed. Emit a conversion to the common
    // type in each branch and produce the result as the right MValue type.
    // ifLikeOp->getRegion(0) is the then-region, getRegion(1) the else-region
    // for both HLCF::ElifOp and ParamIfOp.
    auto emitBranch = [&](Region &region, const ExprNode *expr, Value value) {
      emitter.builder->setInsertionPointToEnd(&region.front());
      auto conv =
          emitter.emitZeroCostConvert({SRValue(value), expr}, commonRefType);
      assert(conv && "getCommonRefType failed");
      auto convVal = conv.getIfSRValue();
      assert(convVal && "zero cost convert changed value type");
      yieldValue(convVal);
    };
    emitBranch(ifLikeOp->getRegion(0), trueExpr, trueMVal);
    emitBranch(ifLikeOp->getRegion(1), falseExpr, falseMVal);
    emitter.builder->setInsertionPointAfter(ifLikeOp);

    // Ensure the correct type is used.
    ifLikeOp->getResult(0).setType(commonRefType);

    // Compute the right IRValue type based on what we were given, we know the
    // inputs are some kind of MValue.
    AnyValue result;
    // TODO: CheckLifetimes cannot handle consumption of indirect RValues.
    // if (falseVal.getIfMRValue() && trueVal.getIfMRValue())
    //   result = MRValue(ifLikeOp->getResult(0));
    if (falseVal.getIfMLValue() && trueVal.getIfMLValue())
      result = MLValue(ifLikeOp->getResult(0));
    else if (falseVal.getIfMBPValue() && trueVal.getIfMBPValue())
      result = MBPValue(ifLikeOp->getResult(0));
    else
      result = MBValue(ifLikeOp->getResult(0));
    return emitter.emitResult(result, this, dest);
  };

  // Handles the register-passable case for an if-like op (HLCF::ElifOp or
  // ParamIfOp). Emits SRValue conversions in each branch, yields them, and
  // fixes up the op result type. Returns {} if not register-passable.
  auto handleRegPassableForIfOp =
      [&](Operation *ifLikeOp, CValue trueVal, CValue falseVal,
          function_ref<void(Value)> yieldValue) -> AnyValue {
    if (!trueVal.getRValueType().isRegisterPassable(trueExpr->getLoc(),
                                                    emitter.shared))
      return {};
    emitter.builder->setInsertionPointToEnd(&ifLikeOp->getRegion(1).front());
    auto falseSR = emitter.emitSRValue({falseVal, falseExpr}, EC_CondExpr);
    if (!falseSR)
      return {};
    yieldValue(falseSR);
    emitter.builder->setInsertionPointToEnd(&ifLikeOp->getRegion(0).front());
    auto trueSR = emitter.emitSRValue({trueVal, trueExpr}, EC_CondExpr);
    if (!trueSR)
      return {};
    yieldValue(trueSR);
    emitter.builder->setInsertionPointAfter(ifLikeOp);
    ifLikeOp->getResult(0).setType(trueSR.getType());
    return emitter.emitResult(SRValue(ifLikeOp->getResult(0)), this, dest);
  };

  // Handles the memory-only case for an if-like op. Emits stores into a shared
  // scratch buffer from each branch, then recreates the op without a result
  // (there is no way to remove results after op creation). `yieldEmpty()`
  // emits the empty branch terminator; `recreate()` creates the result-free
  // replacement op.
  auto handleMemoryOnlyForIfOp =
      [&](Operation *ifLikeOp, CValue trueVal, CValue falseVal,
          function_ref<void()> yieldEmpty,
          function_ref<Operation *()> recreate) -> AnyValue {
    emitter.builder->setInsertionPoint(ifLikeOp);
    MLValue destBuffer =
        dest.getMLValueForResult(getLoc(), trueVal.getRValueType(), emitter);

    emitter.builder->setInsertionPointToEnd(&ifLikeOp->getRegion(1).front());
    ExprDest falseDest(destBuffer, EC_CondExpr);
    (void)emitter.emitResult(falseVal, falseExpr, falseDest);
    yieldEmpty();

    emitter.builder->setInsertionPointToEnd(&ifLikeOp->getRegion(0).front());
    ExprDest trueDest(destBuffer, EC_CondExpr);
    (void)emitter.emitResult(trueVal, trueExpr, trueDest);
    yieldEmpty();

    emitter.builder->setInsertionPointAfter(ifLikeOp);
    Operation *newOp = recreate();
    newOp->getRegion(0).takeBody(ifLikeOp->getRegion(0));
    newOp->getRegion(1).takeBody(ifLikeOp->getRegion(1));
    ifLikeOp->erase();

    return emitter.emitCResult(MRValue(destBuffer), this, dest);
  };

  // Emit a branch expression under `assumption`, inserted into a fresh child
  // scope of the emitter's declScope so that `conforms_to`-based type
  // refinement applies within the branch without leaking the assumption to
  // sibling code. Keeps the emitter's builder insertion point in sync for the
  // dynamic path.
  auto emitBranchUnderAssumption = [&](const ExprNode *branchExpr,
                                       ConstraintAttr assumption) -> AnyValue {
    ASTDecl &branchScope = emitter.getDeclResolver().addFullyResolvedDecl(
        /*declVal=*/nullptr, StringAttr(), branchExpr->getLoc(),
        &emitter.declScope);
    branchScope.insertKnownAssumptions(assumption);
    if (emitter.builder) {
      IREmitter branchEmitter(branchScope, *emitter.builder);
      AnyValue result = branchEmitter.emitExpr(branchExpr, EC_CondExpr);
      // Propagate the advanced insertion point back to the parent emitter.
      emitter.builder = branchEmitter.builder;
      return result;
    }
    IREmitter branchEmitter(branchScope, emitter.paramContext,
                            emitter.deferredTypingContext);
    return branchEmitter.emitExpr(branchExpr, EC_CondExpr);
  };

  // Inside a parameter context, emit conditional expression.
  if (!emitter.builder) {
    PValue condPVal =
        emitter.emitPValue({condRVal, condExpr}, EC_BoolCondition);
    if (!condPVal)
      return {};
    // Emit the expressions, refining each branch under the (possibly inverted)
    // condition so `conforms_to` guards refine type parameters they gate.
    AnyValue trueRawVal = emitBranchUnderAssumption(
        trueExpr, buildBranchAssumption(condPVal.get(),
                                        /*invertCondition=*/false, ifLoc));
    AnyValue falseRawVal = emitBranchUnderAssumption(
        falseExpr,
        buildBranchAssumption(condPVal.get(), /*invertCondition=*/true, ifLoc));
    // Get CValues, resolving a UValue to the other operand's type.
    CValue trueVal =
        emitToCValueInferringType({trueRawVal, trueExpr}, falseRawVal);
    CValue falseVal =
        emitToCValueInferringType({falseRawVal, falseExpr}, trueRawVal);
    // Coerce the types to each other if they disagree.
    if (emitter.coerceTypesToEachOther(getLoc(), trueVal, trueExpr, falseVal,
                                       falseExpr, {},
                                       dest.getExpectedTypeIfSpecified()))
      return {};

    PValue truePVal = emitter.emitPValue({trueVal, trueExpr}, EC_CondExpr);
    PValue falsePVal = emitter.emitPValue({falseVal, falseExpr}, EC_CondExpr);
    if (!truePVal || !falsePVal)
      return {};

    auto value =
        ParamOperatorAttr::get(POC::Cond, {condPVal, truePVal, falsePVal});
    return emitter.emitResult(value, this, dest);
  }

  // If the condition is a comptime PValue, emit kgen.param.if instead of
  // hlcf.elif. During elaboration, processParamIfOp selects and inlines only
  // the live branch, preventing dead-branch ops (e.g. `comptime assert False`)
  // from ever being elaborated.
  if (PValue condPVal = condRVal.getIfPValue()) {
    if (SIMDAttr asBool = sugarDynCast<SIMDAttr>(condPVal.get())) {
      if (!asBool.getAsBool())
        emitter.emitWarning(this->getLoc())
            << "left hand side expression of 'if False' is dead";
      else
        emitter.emitWarning(this->getLoc())
            << "right hand side expression of 'if True' is dead";
    }

    // Create with a placeholder result (the condition's i1 type); the real
    // result type is fixed after emitting both branches. For the memory-only
    // path the op is recreated without a result at the end (same pattern as
    // the hlcf.elif memory-only path below).
    auto paramIfOp =
        ParamIfOp::create(*emitter.builder, ifLoc,
                          TypeRange{condPVal.get().getType()}, condPVal.get());

    emitter.builder->createBlock(&paramIfOp.getThenRegion());
    AnyValue trueRawVal = emitBranchUnderAssumption(
        trueExpr, buildBranchAssumption(condPVal.get(),
                                        /*invertCondition=*/false, ifLoc));

    emitter.builder->createBlock(&paramIfOp.getElseRegion());
    AnyValue falseRawVal = emitBranchUnderAssumption(
        falseExpr,
        buildBranchAssumption(condPVal.get(), /*invertCondition=*/true, ifLoc));

    CValue falseVal =
        emitToCValueInferringType({falseRawVal, falseExpr}, trueRawVal);
    emitter.builder->setInsertionPointToEnd(&paramIfOp.getThenRegion().front());
    CValue trueVal =
        emitToCValueInferringType({trueRawVal, trueExpr}, falseRawVal);

    if (!trueVal || !falseVal) {
      emitter.builder->setInsertionPointAfter(paramIfOp);
      return {};
    }

    if (AnyValue result =
            handleTwoMValuesForIfOp(paramIfOp, trueVal, falseVal, [&](Value v) {
              ParamYieldOp::create(*emitter.builder, ifLoc, ValueRange{v});
            }))
      return result;

    auto configEmitter = [&](bool isLHS) {
      Block &b = isLHS ? paramIfOp.getThenRegion().front()
                       : paramIfOp.getElseRegion().front();
      emitter.builder->setInsertionPointToEnd(&b);
    };
    if (emitter.coerceTypesToEachOther(getLoc(), trueVal, trueExpr, falseVal,
                                       falseExpr, configEmitter,
                                       dest.getExpectedTypeIfSpecified())) {
      dest.resetForError(emitter);
      return {};
    }

    if (AnyValue result = handleRegPassableForIfOp(
            paramIfOp, trueVal, falseVal, [&](Value v) {
              ParamYieldOp::create(*emitter.builder, ifLoc, ValueRange{v});
            }))
      return result;

    // Memory-only: allocate a destBuffer and store into it from each branch.
    // The paramIfOp carries no result value; recreate it without one.
    return handleMemoryOnlyForIfOp(
        paramIfOp, trueVal, falseVal,
        [&] { ParamYieldOp::create(*emitter.builder, ifLoc); },
        [&]() -> Operation * {
          return ParamIfOp::create(*emitter.builder, ifLoc, condPVal.get());
        });
  }

  Value condValue =
      emitter.emitSRValue({AnyValue(condRVal), condExpr}, EC_BoolCondition);

  if (!condValue)
    return {};

  // At this point since we don't know the type of trueExpr / falseExpr, use a
  // dummy type for the 'elif' result.  We'll fix it later.
  auto ifOp = HLCF::ElifOp::create(*emitter.builder, ifLoc,
                                   TypeRange{condValue.getType()}, condValue);

  // Emit the trueVal and falseVal's, coercing any UValue to the other operand
  // type if present, but otherwise not diagnosing conflicts or merging types
  // yet.
  emitter.builder->createBlock(&ifOp.getThenRegion());
  AnyValue trueRawVal = emitter.emitExpr(trueExpr, EC_CondExpr);

  emitter.builder->createBlock(&ifOp.getElseRegion());
  AnyValue falseRawVal = emitter.emitExpr(falseExpr, EC_CondExpr);
  // Get CValues, resolving a UValue to the other operand's type.
  CValue falseVal =
      emitToCValueInferringType({falseRawVal, falseExpr}, trueRawVal);
  emitter.builder->setInsertionPointToEnd(&ifOp.getThenBlock());
  CValue trueVal =
      emitToCValueInferringType({trueRawVal, trueExpr}, falseRawVal);

  if (!trueVal || !falseVal) {
    emitter.builder->setInsertionPointAfter(ifOp);
    return {};
  }

  // If both results were M values and both sides agree with the result type,
  // then we can propagate the result as an MValue that has a merged lifetime.
  if (AnyValue result =
          handleTwoMValuesForIfOp(ifOp, trueVal, falseVal, [&](Value v) {
            HLCF::YieldOp::create(*emitter.builder, ifLoc, v);
          }))
    return result;

  /// If the types disagree, then we need to emit a conversion to a common
  /// type. See if one is convertible to the other, and if so, emit a
  /// conversion to get to a common type.
  auto configEmitter = [&](bool isLHS) {
    Block &b = isLHS ? ifOp.getThenBlock() : ifOp.getElseBlock();
    emitter.builder->setInsertionPointToEnd(&b);
  };
  if (emitter.coerceTypesToEachOther(getLoc(), trueVal, trueExpr, falseVal,
                                     falseExpr, configEmitter,
                                     dest.getExpectedTypeIfSpecified())) {
    dest.resetForError(emitter);
    return {};
  }

  // RegisterPassable values get merged together as SSA registers in the result.
  if (AnyValue result =
          handleRegPassableForIfOp(ifOp, trueVal, falseVal, [&](Value v) {
            HLCF::YieldOp::create(*emitter.builder, ifLoc, v);
          }))
    return result;

  // If we have a memory only type, we have to handle the various issues with
  // the ExprDest.  It may specify an MLValue to emit into, it may be
  // ambiguous (like a call argument) or it may even be something like a
  // DLValue.  We handle this by projecting the ExprDest to an MLValue if we
  // can, but otherwise using a scratch buffer if not.
  return handleMemoryOnlyForIfOp(
      ifOp, trueVal, falseVal,
      [&] { HLCF::YieldOp::create(*emitter.builder, ifLoc); },
      [&]() -> Operation * {
        return HLCF::ElifOp::create(*emitter.builder, ifLoc, TypeRange{},
                                    condValue);
      });
}

/// Emit the comparison expression with operator ops[opIdx] and operands:
///  1. lastExpr: the SSA value of the last expression in the chain emitted
///     so far.
///  2. The next expression node in the chain to be emitted: expr[opIdx + 1].
///
///  lastCmpExpr is the SSA value of the previous comparison expression.
///  Example
///  If the whole chained expression is a < b < c, and hence
///  a < b and b < c,  emitNextCmp will emit b < c, using the SSA
///  value of b in the previous comparison (a < b). lastCmpExpr is the value
///  of a < b.
///  Note that a < b  is handled by ChainedCmpOpNode::emitIR.
RValue ChainedCmpOpNode::emitNextCmp(IREmitter &emitter, size_t opIdx,
                                     RValue prevCmpVal, AnyValue prevRHS,
                                     bool hasPrevIfOp, ExprDest &dest) const {
  ExprContext context = dest.getContext();
  bool isLastOne = opIdx + 1 == ops.size();
  SMLoc ifLoc = exprs[opIdx - 1]->getLoc();
  Location ifLocation = emitter.translateLocation(ifLoc);
  std::optional<OpBuilder> lastBuilder;
  if (emitter.builder)
    lastBuilder = emitter.builder.value();
  RValue prevCmpI1Value =
      emitter.emitScalarBool({prevCmpVal, this}, EC_BoolCondition);
  if (!prevCmpI1Value)
    return {};
  SRValue prevCmpI1SRValue;
  HLCF::ElifOp ifOp;
  if (emitter.builder) {
    prevCmpI1SRValue =
        emitter.emitSRValue({prevCmpI1Value, this}, EC_BoolCondition);
    if (!prevCmpI1SRValue)
      return {};
    // In the dynamic case we need to build the RHS evaluation in the Then
    // region of an ElifOp.  But if we end up having all parameters, it will not
    // have been necessary.
    ifOp =
        HLCF::ElifOp::create(*emitter.builder, ifLocation,
                             prevCmpVal.getType().mlirType, prevCmpI1SRValue);
    emitter.builder->createBlock(&ifOp.getThenRegion());
  }
  AnyValue newRHS = emitter.emitExpr(exprs[opIdx + 1], EC_OperatorOperandValue);
  if (!newRHS)
    return {};
  ExprDest newCmpDest(context);
  AnyValue newCmp =
      emitBinOpCall({prevRHS, exprs[opIdx]}, {newRHS, exprs[opIdx + 1]},
                    ops[opIdx], newCmpDest, this, emitter);
  RValue newCmpCRV = emitter.emitRValue({newCmp, exprs[opIdx]}, context);
  if (!newCmpCRV)
    return {};

  if (prevCmpVal.getIfPValue() && prevCmpI1Value.getIfPValue() &&
      newCmpCRV.getIfPValue()) {
    // Since we have PValues, we didn't actually need that ifOp after all. Let's
    // clean up before returning a PValue directly or recurring.
    if (emitter.builder) {
      ifOp.erase();
      emitter.builder = lastBuilder;
    }
    if (!prevCmpVal.getRValueType().isEqualCanon(newCmpCRV.getRValueType())) {
      emitter.emitError(
          ifLocation,
          "comparison result types of chained comparison must match");
      return {};
    }
    auto chainedBool = ParamOperatorAttr::get(
        POC::Cond,
        {prevCmpI1Value.getIfPValue(), /*trueVal=*/newCmpCRV.getIfPValue(),
         /*falseVal=*/prevCmpVal.getIfPValue()});
    RValue ret = isLastOne ? chainedBool
                           : emitNextCmp(emitter, opIdx + 1, chainedBool,
                                         newRHS, false, dest);
    if (hasPrevIfOp) {
      HLCF::YieldOp::create(*emitter.builder, ifLocation,
                            emitter.emitSRValue({ret, exprs[opIdx]}, context));
    }
    return ret;
  }

  // We need to return the result of the ElifOp as a RValue.
  // More concretely, it will be an SRValue or, for exotic memory-only bool
  // equivalents, one of the pointer type RValues.
  // But for simplicity, let's only support return values that can fit in an
  // SRValue.
  // TODO - make this more general.
  // To refuse memory types right now, check what (other) comparison results
  // are.
  if (!newCmpCRV.getRValueType().isRegisterPassable(ifLoc, emitter.shared)) {
    emitError(ifLocation,
              "chained comparison operator does not currently support "
              "memory-only return types");
    return {};
  }

  RValue newOrNextResult;
  if (isLastOne) {
    newOrNextResult = newCmpCRV;
    auto newCmpSRV = emitter.emitSRValue({newCmpCRV, exprs[opIdx]}, context);
    if (!newCmpSRV)
      return {};
    HLCF::YieldOp::create(*emitter.builder, ifLocation, newCmpSRV);
  } else {
    newOrNextResult =
        emitNextCmp(emitter, opIdx + 1, newCmpCRV, newRHS, true, dest);
    if (!newOrNextResult)
      return {};
  }

  if (!newOrNextResult.getRValueType().isEqualCanon(prevCmpVal.getType())) {
    emitter.emitError(
        ifLocation, "comparison result types of chained comparison must match");
  }
  emitter.builder->createBlock(&ifOp.getElseRegion());
  ifOp->getResult(0).setType(prevCmpVal.getType());

  auto newCmpSRV = emitter.emitSRValue({prevCmpVal, exprs[opIdx - 1]}, context);
  if (!newCmpSRV)
    return {};
  HLCF::YieldOp::create(*emitter.builder, ifLocation, newCmpSRV);
  if (lastBuilder)
    emitter.builder = lastBuilder;
  auto r0 = ifOp->getResult(0);
  if (hasPrevIfOp)
    HLCF::YieldOp::create(*emitter.builder, ifLocation, r0);

  return SRValue(r0);
}

AnyValue ChainedCmpOpNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  AnyValue e0Rep = emitter.emitExpr(exprs[0], EC_OperatorOperandValue);
  AnyValue e1Rep = emitter.emitExpr(exprs[1], EC_OperatorOperandValue);
  if (!e0Rep || !e1Rep)
    return {};

  ExprDest cmpDest(dest.getContext());
  AnyValue cmpe0e1RV =
      emitBinOpCall({e0Rep, exprs[0]}, {e1Rep, exprs[1]}, ops[0],
                    exprs.size() == 2 ? dest : cmpDest, this, emitter);
  if (exprs.size() == 2)
    return cmpe0e1RV;

  RValue lastCmpExpr =
      emitter.emitRValue({cmpe0e1RV, exprs[1]}, EC_BoolCondition);
  RValue e1RV = emitter.emitRValue({e1Rep, exprs[1]}, EC_OperatorOperandValue);
  if (!lastCmpExpr || !e1RV)
    return {};
  return emitter.emitResult(
      emitNextCmp(emitter, 1, lastCmpExpr, e1RV, false, dest), this, dest);
}

AnyValue LambdaNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  return emitter.getDeclResolver().resolveAnonymousClosure(this, emitter, dest);
}

AnyValue FunctionTypeNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // Parameters declared within the function type must be visible. Create a
  // dummy declaration.
  ASTDecl &dummyScope = emitter.getDeclResolver().addFullyResolvedDecl(
      nullptr, StringAttr(), getLoc(), &emitter.declScope);

  dummyScope.setExplicitParamScope();

  // Type check any parameters we have.
  ParsedParamList parsedParamList;
  parsedParamList.params = llvm::to_vector(parsedParams);
  parsedParamList.bodyConstraints = llvm::to_vector(parsedConstraints);
  std::optional<TypeCheckedParamList> paramListOrError =
      TypeCheckedParamList::create(parsedParamList, dummyScope);
  if (!paramListOrError.has_value())
    return {}; // Error already emitted.
  TypeCheckedParamList &paramList = *paramListOrError;

  // Reinflate a ParsedArgumentList.
  ParsedArgumentList argList;
  argList.parsedArgs = llvm::to_vector(parsedArgs);
  argList.resultArg = resultArg;
  argList.effects = effects;
  argList.isThin = isThin;
  argList.isExperimentalParamTrait = isExperimentalParamTrait;
  argList.thrownTypeExpr = const_cast<ExprNode *>(thrownTypeExpr);

  TypeCheckedFnSignature tcSignature(paramList, argList, originExpr,
                                     /*fnDecl=*/nullptr, StringAttr());

  // Compute the signature of the function.
  FnTypeGeneratorType signature = tcSignature.getFnTypeGeneratorType();
  if (!signature)
    return {}; // Error already emitted.

  // Check to see if any of the arguments were erroneous. If so, we don't want
  // to produce a function type with a nested type check error, just fail.  This
  // is unlike function defs which want to handle arguments that are invalid.
  for (auto &arg : tcSignature.argList.parsedArgs) {
    if (arg.isErroneous)
      return {};
  }

  // Set the value of the dummy scope to the generated signature so that we can
  // still resolve information about it in tools.
  dummyScope.setIRValue(PValue(signature));

  // The parsed FuncTypeGeneratorType is set to the pretty type that includes
  // implicit origins, we strip off the named origin decl references and replace
  // them with indices.
  signature = signature.replaceImplicitOriginsWithIndexes(
      tcSignature.implicitOriginDecls);

  if (argList.isClosureFunctionType()) {
    ASTDecl *moduleDecl =
        emitter.getDeclScope().getNearestDeclOfType<FileModuleOp>();
    if (argList.isExperimentalParamTrait) {
      TraitType traitType = emitter.shared.declResolver->getCanonicalTrait(
          emitter.bindParamsToClosureTraitFromSig(signature));
      return emitter.emitResult(ASTType(traitType), this, dest);
    }
    ASTDecl *trait = emitter.shared.getOrCreateClosureTrait(
        getLoc(), *moduleDecl, signature);
    Type traitType = trait->getTypeDeclSelf().extractMetaType();
    return emitter.emitResult(ASTType(traitType), this, dest);
  }
  return emitter.emitResult(ASTType(signature), this, dest);
}

AnyValue GeneratorTypeNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // Parameters declared within the generator type must be visible to the body
  // type expression. Create a dummy declaration scope for them.
  ASTDecl &dummyScope = emitter.getDeclResolver().addFullyResolvedDecl(
      nullptr, StringAttr(), getLoc(), &emitter.declScope);
  dummyScope.setExplicitParamScope();

  ParsedParamList parsedParamList;
  parsedParamList.params = llvm::to_vector(parsedParams);
  std::optional<TypeCheckedParamList> paramListOrError =
      TypeCheckedParamList::create(parsedParamList, dummyScope);
  if (!paramListOrError.has_value())
    return {}; // Error already emitted.
  TypeCheckedParamList &paramList = *paramListOrError;

  // Evaluate the body type in the param scope so generator params resolve.
  IREmitter typeEmitter(dummyScope, EC_Type);
  ASTType bodyType =
      typeEmitter.emitExprType(bodyTypeExpr, /*allowUnbound=*/true);
  if (!bodyType)
    return {};

  // Generator types reference their input parameters by index, not by name.
  IndexRefRemapper remapper(paramList.paramDeclAttrs, {});
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : paramList.paramDeclAttrs)
    inputParamTypes.push_back(remapper.replace(param.getType()));

  PogListAttr metadata = remapper.replace(paramList.getParamListAttr());
  Type genType = GeneratorType::get(
      inputParamTypes, remapper.replace(bodyType.mlirType), metadata);

  dummyScope.setIRValue(PValue(genType));
  return emitter.emitResult(ASTType(genType), this, dest);
}

/// Return true if the magic function kind is considered stable.
/// Stable magic functions do not emit warnings with --warn-on-unstable-apis.
static bool isStableMagicFunction(ExprNode::Kind kind) {
  switch (kind) {
  case ExprNode::kOriginOf:
  case ExprNode::kTypeOf:
  case ExprNode::kConformsTo:
  case ExprNode::kFunctionsInModule:
  case ExprNode::kIsRunInComptimeInterpreter:
    return true;
  default:
    return false;
  }
}

AnyValue MagicFunctionNode::emitIR(ExprDest &dest, IREmitter &emitter) const {
  // Emit stability warning for unstable magic functions.
  if (!isStableMagicFunction(kind))
    checkMagicFunctionAndWarn(spelling, getLoc(), emitter.getDeclScope(),
                              emitter.shared, getRange());

  // Helper lambda to check argument count and emit an error if mismatched.
  // Returns true if the argument count matches, false otherwise.
  auto checkArgCount = [&](size_t expected) -> bool {
    if (subExprs.size() == expected)
      return true;
    emitter.emitError(getLoc())
        << "expected " << expected << " arguments, but this call has "
        << subExprs.size() << " arguments" << getRange();
    return false;
  };

  if (kind == kOriginOf)
    return emitOriginOf(dest, emitter);

  if (kind == kFunctionsInModule) {
    if (!checkArgCount(0))
      return {};
    return emitFunctionsInModule(dest, emitter);
  }

  if (kind == kGetCurrentFunctionName) {
    if (!checkArgCount(0))
      return {};
    return emitGetCurrentFunctionName(dest, emitter);
  }

  if (kind == kIsRunInComptimeInterpreter)
    return emitIsRunInComptimeInterpreter(dest, emitter);

  if (kind == kConformsTo) {
    if (!checkArgCount(2))
      return {};
    return emitConformsTo(dest, emitter);
  }

  // struct_field_ref takes two arguments: index and struct reference.
  if (kind == kStructFieldRef) {
    if (!checkArgCount(2))
      return {};
    return emitStructFieldRef(dest, emitter);
  }

  // All other magic function types take exactly one argument.
  if (!checkArgCount(1))
    return {};

  if (kind == kTypeOf)
    return emitTypeOf(dest, emitter);

  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(this);

  // Emit the subexpression.
  ExprNode *subExpr = subExprs.front();
  CValue subExprValue = emitter.emitExprCValue(subExpr, dest.getContext());
  if (!subExprValue)
    return {};

  // __get_mvalue_as_litref(someMValue) returns the !lit.ref.
  if (kind == kGetMValueAsLitRef) {
    if (!subExprValue.isMValue()) {
      emitter.emitError(getLoc(), "cannot use non-memory value") << getRange();
      return {};
    }

    // Return the MValue as an SRValue since the ref itself is the result.
    Value refValue = subExprValue.getMValueReference();
    return emitter.emitResult(SRValue(refValue), this, dest);
  }

  // __get_litref_as_mvalue(someLITRef) returns an MValue.
  if (kind == kGetLitRefAsMValue) {
    Value exprVal =
        emitter.emitSRValue({subExprValue, subExpr}, dest.getContext());
    if (!exprVal || sugarIsa<TypeCheckErrorType>(exprVal.getType()))
      return {};
    if (!sugarIsa<RefType>(exprVal.getType())) {
      emitter.emitError(getLoc(), "operand isn't a '!lit.ref' type ")
          << ASTType(exprVal.getType()) << getRange();
      return {};
    }

    return emitter.emitResult(CValue::getMValueForRef(exprVal), this, dest);
  }

  // __get_address_as_uninit_lvalue and __get_address_as_owned_value take a
  // !kgen.pointer.
  RValue exprRVal =
      emitter.emitRValue({subExprValue, subExpr}, dest.getContext());
  if (!exprRVal || sugarIsa<TypeCheckErrorType>(exprRVal.getType()))
    return {};

  Value exprVal = emitter.emitSRValue({exprRVal, subExpr}, dest.getContext());
  if (!exprVal)
    return {};
  // Strip sugar that gets in the way of the pointer type.
  exprVal = emitter.emitRebindOpIfNeeded(
      exprVal, SugarAttr::strip(exprVal.getType()), subExpr->getLoc());

  if (!isa<PointerType>(exprVal.getType())) {
    emitter.emitError(getLoc(),
                      "operand must have '!kgen.pointer<T>' type, not ")
        << exprRVal.getRValueType() << getRange();
    return {};
  }

  // TODO(references): if we keep these functions, they should take a origin.
  auto immortal = emitter.builder->getAttr<AnyOriginAttr>(/*isMut=*/true);
  bool startsUninit = kind == ExprNode::kGetAddressAsUninitLValue;
  bool endsUninit = kind == ExprNode::kGetAddressAsOwned;
  exprVal =
      RefFromPointerOp::create(*emitter.builder, getLocation(emitter), exprVal,
                               immortal, startsUninit, endsUninit);

  /// __get_address_as_owned_value(ptr) # returns RValue
  if (kind == ExprNode::kGetAddressAsOwned)
    return emitter.emitResult(MRValue(exprVal), this, dest);

  // __get_address_as_uninit_lvalue(ptr) returns an MLValue
  assert(kind == kGetAddressAsUninitLValue);
  return emitter.emitResult(MLValue(exprVal), this, dest);
}

AnyValue MagicFunctionNode::emitOriginOf(ExprDest &dest,
                                         IREmitter &emitter) const {
  // Gather the origins of each subexpression value. If any of the origins
  // are immutable, then we mutcast the rest to immutable.
  SmallVector<TypedAttr> origins;
  CValue singleOrigin;

  for (ExprNode *subExpr : subExprs) {
    emitter.emitExpressionWithoutEvaluatingIt(
        subExpr, EC_Origin, [&](CValue result, IREmitter &emitter) {
          // If this is a value of std.Origin type, remember it.
          if (auto origin = ASTType(result.getType()).isOriginStruct())
            singleOrigin = result;

          if (auto origin = emitter.extractOriginOf(subExpr, result))
            origins.push_back(origin);
        });
  }

  // If we had a single value of Origin type, use it directly. This allows
  // parameters to match up more successfully when the origin is a parameter,
  // rather than getting an UnknownAttr.
  if (origins.size() == 1 && singleOrigin)
    return emitter.emitResult(singleOrigin, this, dest);

  // Form the final value of !lit.origin type.
  auto resultLitOrigin = OriginUnionAttr::get(emitter.getContext(), origins);
  return emitter.emitResult(
      emitter.getStdlibOriginOf(resultLitOrigin, getLoc()), this, dest);
}

AnyValue MagicFunctionNode::emitTypeOf(ExprDest &dest,
                                       IREmitter &emitter) const {
  // TypeOf can reference dynamic values even when in a parameter context.
  ASTType resultType;
  emitter.emitExpressionWithoutEvaluatingIt(
      subExprs.front(), EC_Origin, [&](CValue result, IREmitter &emitter) {
        resultType = result.getRValueType();
      });

  if (!resultType) // Error emitting subexpr.
    return {};

  return emitter.emitResult(PValue(resultType), this, dest);
}

AnyValue MagicFunctionNode::emitConformsTo(ExprDest &dest,
                                           IREmitter &emitter) const {
  ASSERT_STREAM(subExprs.size() == 2,
                << "conforms_to requires exactly two operands");

  // The first operand is the checked operand: either a single type value or a
  // `param_list` of type values (e.g. `Ts.values` for a variadic pack). The
  // second operand is the trait to check conformance against.
  ExprNode *checkedExpr = subExprs[0];
  PValue checkedValue = emitter.emitExprPValue(checkedExpr, EC_ConformsTo);
  if (!checkedValue) {
    emitter.emitError(checkedExpr->getLoc(),
                      "expected a type or param_list operand")
        << checkedExpr->getRange();
    return {};
  }

  TypedAttr checkedAttr = checkedValue.get();
  if (sugarIsa<ParamListType>(checkedAttr.getType())) {
    if (!LIT::isVariadicOfMetaType(checkedAttr.getType())) {
      emitter.emitError(checkedExpr->getLoc(),
                        "param_list operand must contain type expressions")
          << checkedExpr->getRange();
      return {};
    }
  } else if (!LIT::isFirstLevelTypeExpr(checkedAttr)) {
    // TODO: we should fold to constant when the metatype is `StructMetaType`,
    // that means we know the concrete type to check at parsing time.
    emitter.emitError(checkedExpr->getLoc())
        << checkedValue << " is not a type expression"
        << checkedExpr->getRange();
    return {};
  }

  // The second operand must be the trait to check conformance against.
  ExprNode *traitExpr = subExprs[1];
  PValue traitToCheck = emitter.emitExprPValue(traitExpr, EC_ConformsTo);
  if (!traitToCheck)
    return {};

  if (!sugarIsaAndNonNull<AnyTraitType>(traitToCheck.getType())) {
    emitter.emitError(traitExpr->getLoc(),
                      "expected a trait for the second argument")
        << traitExpr->getRange();
    return {};
  }

  TypedAttr conformToI1 =
      emitter.shared.getEvaluationContext().getAndFold<TypeConformsToTraitAttr>(
          checkedAttr, traitToCheck.get());
  return emitter.emitBool({conformToI1, this}, dest);
}

AnyValue
MagicFunctionNode::emitGetCurrentFunctionName(ExprDest &dest,
                                              IREmitter &emitter) const {
  // Intentional choice: return an empty string
  // if used outside of a function (e.g. in a comptime field initializer)
  std::string funcName;
  ASTDecl *fnDecl = emitter.declScope.getNearestDeclOfType<FnOp>();
  if (fnDecl) {
    FnOp fnOp = cast<FnOp>(fnDecl->getIfOperation());
    // This implementation should be kept in sync with the
    // evaluation of KGEN_GetSourceNameAttr in
    // IREvaluatorContext::evaluateGetSourceNameAttr().
    if (auto s = fnOp.getSourceName())
      funcName = s->str();
  }
  return StringLiteralNode::emitCtorCall(funcName, this, dest, emitter);
}

AnyValue
MagicFunctionNode::emitIsRunInComptimeInterpreter(ExprDest &dest,
                                                  IREmitter &emitter) const {
  // Emit a dynamic SSA value which cannot be used in comptime expression.
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(this);
  auto op = IsRunInComptimeInterpreterOp::create(*emitter.builder,
                                                 this->getLocation(emitter));
  // Wrap the scalar<bool> result in a Bool so callers can use it with 'not',
  // 'and', 'or', etc.
  ASTType boolType = emitter.shared.lookupBuiltinType(
      "Bool", emitter.getDeclScope(), getLoc());
  AnyValue i1Value = SRValue(op.getResult());
  CallOperands ctorOps(CallSyntax::kImplicitConvert, this, std::move(dest),
                       ArrayRef<ASTExprAnd<AnyValue>>{{i1Value, this}});
  return emitter.emitConstructorCall(boolType, std::move(ctorOps));
}

AnyValue MagicFunctionNode::emitFunctionsInModule(ExprDest &dest,
                                                  IREmitter &emitter) const {
  // We are basically emitting an expression of the form `Tuple(foo, bar)`,
  // where `foo` and `bar` are the functions declared in the module.

  // Collect a SyntheticNode for each function declaration in the module.
  SmallVector<SyntheticNode> funcValues;
  if (ASTDecl *fileModuleDecl =
          emitter.getDeclScope().getNearestDeclOfType<FileModuleOp>()) {
    for (auto &[name, decls] : fileModuleDecl->getDeclsInScope()) {
      for (ASTDecl *decl : decls) {
        // TODO: consider allowing imported functions. This could be a feature,
        // allowing one to run tests from across several files, simply by
        // importing them. If we decide to do so, it might be best to control
        // this behavior with a flag. For now, we keep the behavior simple.
        if (decl->getParentDecl() != fileModuleDecl)
          continue;

        auto fnOp = dyn_cast_or_null<FnOp>(decl->getIfOperation());
        if (!fnOp || fnOp.isSynthetic())
          continue;

        // TODO(MOCO-2556): consistently allow recursive references.
        if (name == "main")
          continue;

        // We need to resolve the signature, otherwise if the intrinsic is
        // called before the function is declared, the emission will crash.
        if (failed(
                emitter.getDeclResolver().resolveSignature(*decl, getLoc()))) {
          emitter.emitError(decl->getLoc(),
                            "failed to resolve signature for function ");
          return {};
        }

        // To handle overloads, we form a singleton overload set (since we have
        // the ASTDecl for each overload), emit it, and wrap the result in a
        // SyntheticNode, which will be passed to the `Tuple` constructor.
        auto singletonOverloadSet = OverloadSetUValue::create(
            name.strref(), ArrayRef<ASTDecl *>{decl},
            ParamBindings(emitter.getDeclScope(), this),
            CallSyntax::kDirectCall);
        auto funcValue = emitter.emitResult(singletonOverloadSet, this, dest);
        funcValues.emplace_back(SyntheticNode(getLoc(), funcValue));
      }
    }
  }

  // Build the expression.
  DeclRefNode tupleDeclRef("Tuple");
  SmallVector<Operand> callOperands;
  for (SyntheticNode &funcValueNode : funcValues) {
    callOperands.emplace_back(
        Operand(&funcValueNode, getLoc(), ArgUnpackStyle::kPositional));
  }
  CallNode tupleCallNode(&tupleDeclRef, getLoc(), callOperands, getLoc());

  // Finally, emit the IR for the expression.
  return tupleCallNode.emitIR(dest, emitter);
}

//===----------------------------------------------------------------------===//
// Struct field reflection magic functions.
//
// These magic functions provide type validation and cleaner syntax for struct
// field reflection. The underlying KGEN attributes (StructFieldTypesAttr,
// StructFieldNamesAttr) implement ContextuallyEvaluatedAttrInterface, which
// allows them to be evaluated during elaboration after generic type parameters
// have been specialized.
//
// See stdlib/std/reflection/reflection.mojo for detailed documentation.
static TraitType getAnyTypeTraitType(IREmitter &emitter, SMLoc loc) {
  TraitType anyType = emitter.shared.lookupBuiltinTraitType("AnyType", loc);
  if (!anyType) {
    emitter.emitError(loc, "can not locate 'AnyType'");
    return {};
  }
  return anyType;
}

AnyValue MagicFunctionNode::emitStructFieldRef(ExprDest &dest,
                                               IREmitter &emitter) const {
  auto mlirLoc = getLocation(emitter);

  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(this);
    return {};
  }

  TraitType anyTypeTraitType = getAnyTypeTraitType(emitter, getLoc());
  if (!anyTypeTraitType)
    return {};

  // First argument: compile-time index
  CValue indexCValue = emitter.emitIndex(subExprs[0], EC_MLIRMagic);
  if (!indexCValue)
    return {};

  PValue indexPValue = indexCValue.getIfPValue();
  if (!indexPValue) {
    emitter.emitError(subExprs[0]->getLoc(),
                      "struct_field_ref requires a compile-time integer index")
        << subExprs[0]->getRange();
    return {};
  }
  TypedAttr indexAttr = indexPValue.get();

  // Second argument: reference to struct
  AnyValue structExprValue = emitter.emitExpr(subExprs[1], dest.getContext());
  if (!structExprValue) {
    return {};
  }
  MBValue structRef =
      emitter.emitMBValue({structExprValue, subExprs[1]}, dest.getContext());
  if (!structRef) {
    emitter.emitError(subExprs[1]->getLoc(),
                      "struct_field_ref requires a reference to a struct")
        << subExprs[1]->getRange();
    return {};
  }

  // Get the struct type from the reference
  auto refType = cast<RefType>(structRef.getType());
  Type elementType = refType.getElementType();

  Value resultRef;
  MLIRContext *ctx = emitter.getContext();

  // If we have a concrete struct type and a concrete index, use the direct path
  auto structType = sugarDynCast<LIT::StructType>(elementType);
  ErrorOr<int64_t> indexValueOr = POP::getScalarIndexValue(indexAttr);
  if (structType && succeeded(indexValueOr)) {
    // RefStructGEROp requires the base to be a StructType: rebind away sugar.
    if (!isa<LIT::StructType>(structRef.getRValueType())) {
      auto newEltType = SugarAttr::strip(structRef.getRValueType());
      structRef = emitter.emitRebindOpIfNeeded(
          structRef, refType.getWithElement(newEltType), getLoc());
    }

    // Look up the struct declaration
    SymbolRefAttr structSymbol = structType.getSymbol();
    ASTDecl &structAstDecl =
        emitter.shared.declResolver->getDeclForTypeSymbol(structSymbol);
    auto structDecl =
        dyn_cast<LIT::StructDeclOp>(structAstDecl.getIfOperation());
    if (!structDecl) {
      std::string symbolStr;
      llvm::raw_string_ostream os(symbolStr);
      structSymbol.print(os);
      emitter.emitError(getLoc(), Twine("could not find struct declaration "
                                        "for ") +
                                      symbolStr);
      return {};
    }

    // Concrete index case: emit RefStructGEROp directly
    size_t idx = static_cast<size_t>(*indexValueOr);
    StructFieldOp fieldOp;
    size_t currentIdx = 0;
    for (Operation &op : structDecl.getFields().front()) {
      if (auto field = dyn_cast<StructFieldOp>(&op)) {
        if (currentIdx == idx) {
          fieldOp = field;
          break;
        }
        ++currentIdx;
      }
    }

    if (!fieldOp) {
      emitter.emitError(subExprs[0]->getLoc(), Twine("struct field index ") +
                                                   Twine(idx) +
                                                   " is out of bounds")
          << subExprs[0]->getRange();
      return {};
    }

    resultRef =
        RefStructGEROp::create(*emitter.builder, mlirLoc, structRef, fieldOp);
  } else {
    // Parametric index case (or parametric struct type): emit RefStructGEROp
    // with index access. This will be canonicalized to field name access after
    // the index is resolved during elaboration/canonicalization.

    // We need to compute the result type. The result element type is computed
    // as VariadicGet(StructFieldTypes(T), index). The origin is the container's
    // origin (not field-sensitive until canonicalization to field name access).
    RefType containerRefType = cast<RefType>(structRef.getType().mlirType);
    auto variadicType = ParamListType::get(anyTypeTraitType);
    TypedAttr structTypeAttr;
    if (structType) {
      structTypeAttr = TypeParamAttr::get(structType, anyTypeTraitType);
    } else {
      // For parametric types, wrap the element type as a type attribute
      structTypeAttr = TypeParamAttr::get(elementType, anyTypeTraitType);
    }
    auto fieldTypesAttr =
        emitter.shared.getEvaluationContext().getAndFold<StructFieldTypesAttr>(
            ctx, structTypeAttr, variadicType);

    // Compute element type as VariadicGet(fieldTypes, index)
    auto elementTypeAttr = ParamListGetAttr::get(fieldTypesAttr, indexAttr);
    Type resultElementType = ParamType::get(elementTypeAttr);

    // The result ref type uses the container's origin (not field-sensitive)
    // The proper field-sensitive origin is computed when canonicalized to
    // field name access
    RefType resultType =
        RefType::get(resultElementType, containerRefType.getOrigin(),
                     containerRefType.getAddressSpace());

    resultRef = RefStructGEROp::create(*emitter.builder, mlirLoc, resultType,
                                       indexAttr, structRef);
  }

  // Apply type refinement to struct field accesses whose element type is a
  // parametric type with comptime assumptions. This handles tuple subscript
  // where elements are reference types like ref[origin] T (and assumptions
  // introduced by `where`, `comptime if`, or `comptime assert`).
  resultRef = maybeEmitRefinementRebind(resultRef, emitter.declScope,
                                        *emitter.builder, mlirLoc);

  // Result kind depends on the input kind - preserve mutability
  CValue result;
  if (structExprValue.getIfMLValue())
    result = MLValue(resultRef);
  else if (structExprValue.getIfMBPValue())
    result = MBPValue(resultRef);
  else
    result = MBValue(resultRef);
  return emitter.emitCResult(result, this, dest);
}

LogicalResult TupleNode::emitDestructuringPValue(PValue toUnpack,
                                                 IREmitter &emitter) const {
  auto getTupleItem = [&](Type eltType, unsigned index) {
    // Get the item from the tuple into the corresponding LValue.
    ExprDest eltDest(eltType, EC_TupleElement);

    // Bind the i parameters.  Int explicitly constructs from index type now.
    TypedAttr indexAttr =
        IntegerAttr::get(IndexType::get(emitter.getContext()), index);

    CValue intIndexCValue =
        emitter.emitInt(ASTExprAnd<PValue>{PValue(indexAttr), this},
                        ExprContext::EC_CallParamValue);
    PValue intIndex = intIndexCValue.getIfPValue();
    assert(intIndex && "Int must be PValue when constructed from int attr");

    SyntheticNode indexExpr(getLoc(), intIndex);
    Operand exprOperand(&indexExpr, getLoc(), ArgUnpackStyle::kPositional);
    SubscriptNode subscript(this, this->getLoc(), {}, this->getLoc());

    // Emit the extraction from the tuple as a synthesized subscript with
    // this value as an index.
    auto elem = emitGetterSetterAccess(&subscript, {toUnpack, this},
                                       exprOperand, eltDest, emitter);
    if (!elem) {
      eltDest.resetForError(emitter);
      return PValue{};
    }
    assert(elem.getIfPValue() &&
           "expect PValue result when unpacking a PValue tuple");
    return elem.getIfPValue();
  };

  SmallVector<ASTType> eltTypes;

  ASTType expectedType = toUnpack.getType();
  ASTType tupleType = emitter.shared.lookupBuiltinType(
      "Tuple", emitter.getDeclScope(), getLoc());

  if (!tupleType.isEqualCanon(
          expectedType.getWithoutParameters(emitter.shared))) {
    emitter.emitError(getLoc(), "expected a tuple type to destructure, got ")
        << expectedType << getRange();
    return failure();
  }

  assert(expectedType.getParamBindings().size() == 2 &&
         "Tuple has two parameter");
  // This must be a fully resolved tuple type.
  auto vaAttr = sugarCast<ParamListAttr>(expectedType.getParamBindings()[0]);
  if (vaAttr.getValues().size() == exprs.size()) {
    for (auto [i, typeElt] : llvm::enumerate(vaAttr.getValues())) {
      PValue eltPVal = getTupleItem(ASTType(typeElt), i);
      if (!eltPVal ||
          failed(exprs[i]->emitDestructuringPValue(eltPVal, emitter)))
        return failure();
    }
  } else {
    // Make sure lhs matches with rhs in order to destructure.
    emitter.emitError(getLoc(), "cannot unpack value of ")
        << expectedType << " of " << vaAttr.getValues().size() << " element"
        << plural(exprs.size()) << " into " << exprs.size() << " value"
        << plural(exprs.size()) << getRange();
    return failure();
  }

  return success();
}

// There are two options. We are emitting an instance of Tuple.
// That is, (exp, exp) is sugar for Tuple[typeof(expr), typeof(expr)](exp, exp)
// and we want to emit a constructor call and infer the parameter types
// of Tuple.
auto TupleNode::emitLCVIR(ExprDest &dest, IREmitter &emitter,
                          bool isSpeculative) const -> ELVIITResult {
  auto formTupleDLValue =
      [&](ArrayRef<ASTExprAnd<AnyValue>> elements) -> AnyValue {
    SmallVector<Type> typeElts;
    for (ASTExprAnd<AnyValue> elt : elements)
      typeElts.push_back(elt.ir.getIfLValue().getRValueType());
    ASTType concretizedTupleType =
        emitter.getBuiltinTupleInstantiation(getLoc(), typeElts);
    if (!concretizedTupleType || concretizedTupleType.isTypeCheckErrorType())
      return {};
    DLValue result(
        RCRef<TupleDLValue>::create(elements, concretizedTupleType, this));
    return emitter.emitResult(result, this, dest);
  };

  // If this tuple is being speculatively emitted on the LHS of an assignment,
  // speculatively emit each subelement.  It is possible that some will remain
  // unresolvable, e.g. for `(var x, y) = foo()` when 'x' is declared but
  // 'y' is not.  When this happens, we need to bundle things up and return a
  // new TupleNode.
  if (isSpeculative) {
    SmallVector<ELVIITResult> eltResults;
    bool allEltsKnownLValue = true;
    bool anyExprChanged = false;
    for (const ExprNode *expr : exprs) {
      auto result = expr->emitLValueIfImplicitlyTyped(
          emitter, dest.getPatternDeclKind(), /*hasInferrableRHS=*/false);
      if (result.isFailure())
        return {};
      if (auto *newExpr = result.getIfExprNode()) {
        // Emitting the expr could result in a new subexpr that is different,
        // which forces us to rebuild this node.
        anyExprChanged |= newExpr != expr;
        allEltsKnownLValue = false;
      } else {
        anyExprChanged = true; // Successfully emitted a subexpr.
        allEltsKnownLValue &= !result.getIfValue().getIfLValue().isNull();
      }
      eltResults.push_back(result);
    }

    // If we successfully emitted everything to an LValue, bind and return the
    // resolved LValue for the aggregate so the RHS of the assignment can infer
    // from the known type of the tuple.
    if (allEltsKnownLValue) {
      SmallVector<ASTExprAnd<AnyValue>> lvElements;
      for (auto [result, expr] : llvm::zip(eltResults, exprs))
        lvElements.push_back({result.getIfValue().getIfLValue(), expr});
      return formTupleDLValue(lvElements);
    }

    // If no exprs changed, just keep the same node. (a,b,_) = foo() where both
    // are implicitly declared.
    if (!anyExprChanged)
      return this;

    // Otherwise, we need to rebuild a new node capturing the emitted subexpr
    // values so the remaining exprs can be emitted but they don't get
    // re-emitted.
    auto &shared = emitter.shared;
    SmallVector<ExprNode *> newExprs;
    for (auto [result, expr] : llvm::zip(eltResults, exprs)) {
      if (auto *newExpr = result.getIfExprNode())
        newExprs.push_back(const_cast<ExprNode *>(newExpr));
      else {
        assert(result.getIfValue() && "unexpected result kind");
        newExprs.push_back(shared.allocPersistent<SyntheticNode>(
            expr->getLoc(), result.getIfValue()));
      }
    }
    return shared.allocPersistent<TupleNode>(
        firstCommaLoc, shared.getPersistentCopy(ArrayRef(newExprs)));
  }

  // Otherwise, emit in a non-speculatively path, which could be an LValue or
  // RValue.  It could even be a type!
  ASTType tupleType =
      emitter.shared.lookupBuiltinType("Tuple", emitter.declScope, getLoc());
  if (sugarIsa<TypeCheckErrorType>(tupleType))
    return {};

  // If the tuple has an inferred type, as in `(a, b)=foo()`, propagate the
  // element types into the subexpressions if possible to enable implicit var
  // definition.
  SmallVector<ASTType> eltTypes;
  bool isLValueType = false;
  if (auto expectedType = dest.getExpectedTypeIfSpecified()) {
    isLValueType = !dest.getIfInitializerType().isNull();
    // Special case the element type of Tuple.  We could be more general than
    // this when there was a reason to, e.g. looking up a __getitem__
    // implementation.
    if (tupleType.isEqualCanon(
            expectedType.getWithoutParameters(emitter.shared))) {
      assert(expectedType.getParamBindings().size() == 2 &&
             "Tuple has a param_list and TypeList parameter");
      if (auto variadicAttr =
              sugarDynCast<ParamListAttr>(expectedType.getParamBindings()[0])) {
        if (variadicAttr.getValues().size() == exprs.size()) {
          for (auto typeElt : variadicAttr.getValues())
            eltTypes.push_back(ASTType(typeElt));
        }
      }
    } else if (isLValueType) {
      if (!isa<TypeCheckErrorType>(expectedType))
        emitter.emitError(getLoc(), "cannot unpack value of type ")
            << expectedType << " into " << exprs.size() << " value"
            << plural(exprs.size()) << getRange();
      return {};
    }
  }

  bool allEltsLValue = true;
  SmallVector<ASTExprAnd<AnyValue>> elements;
  for (auto [i, expr] : llvm::enumerate(exprs)) {
    // Use an inferred element type if we have one.
    ExprDest eltDest(EC_TupleElement);
    if (!eltTypes.empty()) {
      if (isLValueType) {
        eltDest = ExprDest(LValueInitializerType{eltTypes[i]}, EC_TupleElement);
      } else {
        eltDest = ExprDest(eltTypes[i], EC_TupleElement);
      }
    }

    // Propagate var/ref context.
    eltDest.setPatternDeclKind(dest.getPatternDeclKind());
    auto exprVal = emitter.emitExpr(expr, eltDest);
    if (!exprVal)
      return {};
    allEltsLValue &= !exprVal.getIfLValue().isNull();
    elements.push_back({std::move(exprVal), expr});
  }
  assert(allEltsLValue || !elements.empty());

  // If this is a tuple with all LValue elements, return a DLValue since we
  // can assign into this expression.
  // TODO: Add support for list LValues as well.
  if (allEltsLValue)
    return formTupleDLValue(elements);

  // The ASTType will carry around parameters bound, we want to unbind them so
  // they can be inferred from the elements.
  tupleType = tupleType.getWithoutParameters(emitter.shared);

  // Emit a call to the builtin type constructor as an implicit conversion.
  // The type parameters are inferred from the element types.
  return emitter.emitConstructorCall(
      tupleType,
      CallOperands(CallSyntax::kTypeCall, this, std::move(dest), elements));
}
