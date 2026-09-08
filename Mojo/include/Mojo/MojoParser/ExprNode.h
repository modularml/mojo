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
// This file defines the ExprNode base class and support classes used for
// emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_EXPRNODE_H
#define KGEN_MOJOPARSER_EXPRNODE_H

#include "Mojo/MojoParser/ExprDest.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <variant>

namespace mlir {
class raw_indented_ostream;
} // namespace mlir

namespace M {
class SourceRange;
} // namespace M

namespace M::KGEN::LIT {
using llvm::SMLoc;
class AnyValue;
class ASTType;
class IREmitter;
class ExprDest;

//===----------------------------------------------------------------------===//
// ExprNode
//===----------------------------------------------------------------------===//

/// Base class for all expression nodes.  Note that these nodes are not allowed
/// to own memory since they are bump pointer allocated and their destructors
/// are never run.
class ExprNode {
public:
  // This indicates the subclass.
  enum Kind {
    kSynthetic,            // There is no source corresponding to the IR.
    kIntLiteral,           // 42
    kFloatLiteral,         // 1.1
    kBoolLiteral,          // False
    kSelfLiteral,          // Self
    kStringLiteral,        // "Hello"
    kTStringLiteral,       // t"Hello, {name}!"
    kNoneLiteral,          // None
    kDiscardLiteral,       // _
    kEllipsisLiteral,      // ...
    kDeclRef,              // x
    kAttributeRef,         // x.y
    kInferredAttributeRef, // .y  (inferred base, e.g. foo(.f64))
    kParen,                // (x+y)
    kTuple,                // (), (x,), (x, y), etc
    kSliceLiteral,      // :, a:, :a, ::, a:b:c, etc.  Only valid in subscripts.
    kListLiteral,       // [x, y]
    kDictLiteral,       // {a: 1, b: 2, **dictUnpack}
    kSetInitLiteral,    // {x, y} and {z=4, "foo"}
    kListComprehension, // [x for x in range(10) if x & 1]
    kSetComprehension,  // {x for x in range(10) if x & 1}
    kDictComprehension, // {x: x * x for x in range(10) if x & 1}
    kCall,              // thing(a, b)
    kSubscript,         // thing[a, b:c]
    kSubscriptArrow,    // thing[x, y -> a, b]
    kChainedCmp,        // a < b <= c
    kFunctionType,      // async def[](owned Int, &F32) capturing raises -> F64
    kGeneratorType,     // __generator_type[Idx: Int] ToT
    kLambda,            // lambda [y: Int](x: Int) raises {mut} -> F64: x + y

    // Magic functions
    kGetMValueAsLitRef,          // __get_mvalue_as_litref(x)
    kGetLitRefAsMValue,          // __get_litref_as_mvalue(x)
    kGetAddressAsUninitLValue,   // __get_address_as_uninit_lvalue(x)
    kGetAddressAsOwned,          // __get_address_as_owned_value(x)
    kOriginOf,                   // origin_of(x)
    kTypeOf,                     // type_of(x)
    kConformsTo,                 // conforms_to(T, Trait)
    kFunctionsInModule,          // __functions_in_module()
    kGetCurrentFunctionName,     // __get_current_function_name()
    kStructFieldRef,             // __struct_field_ref(idx, s)
    kIsRunInComptimeInterpreter, // __is_run_in_comptime_interpreter
    kFirstMagicFunction = kGetMValueAsLitRef,
    kLastMagicFunction = kIsRunInComptimeInterpreter,

    // Prefix and Postfix unary expressions.
    kNeg,      // -x
    kPos,      // +x
    kInvert,   // ~x
    kUnpack,   // *x
    kBoolNot,  // not x
    kAwait,    // await x
    kTransfer, // x^
    kVarPat,   // var x
    kRefPat,   // ref x
    kComptime, // comptime (x)
    kFirstUnaryOp = kNeg,
    kLastUnaryOp = kComptime,

    // Binary expressions.
    kAdd,
    kSub,
    kMul,
    kMatMul,
    kTrueDiv,
    kFloorDiv,
    kMod,
    kBoolOr,  // x or y
    kBoolAnd, // x and y
    kCmpIn,
    kCmpNotIn,
    kCmpIs,
    kCmpIsNot,
    kCmpLT,
    kCmpLE,
    kCmpGT,
    kCmpGE,
    kCmpNE,
    kCmpEQ,
    kOr,
    kXor,
    kAnd,
    kLShift,
    kRShift,
    kPow,
    kWalrus,      // x := y aka walrus
    kTypePattern, // x: Int   (Used in assignment statements).
    kAsPat,       // x as y   (match patterns)

    // Assignment and Inplace operators.
    kAssign, // x = y aka assignment_expression
    kIAdd,
    kISub,
    kIMul,
    kIMatMul,
    kITrueDiv,
    kIFloorDiv,
    kIMod,
    kIAnd,
    kIOr,
    kIXor,
    kILShift,
    kIRShift,
    kIPow,

    // Ternary expressions.
    kIfElse,

    // Not a valid expression node.
    kInvalid,

    kFirstAssignStmt = kAssign,
    kLastAssignStmt = kIPow,

    kFirstBinOp = kAdd,
    kLastBinOp = kIPow,
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual llvm::SMLoc getLoc() const = 0;

  /// Return the 'loc' for this node translated to an MLIR location.
  Location getLocation(IREmitter &emitter) const;

  /// Return the source range spanned by this expression.
  virtual SourceRange getRange() const = 0;

  /// Print the expression node.
  virtual void print(mlir::raw_indented_ostream &os) const = 0;
  void print(raw_ostream &os) const;
  /// Dump the expression node for debugging.
  LLVM_DUMP_METHOD void dump() const;

  /// Return the start or end of the source range.
  llvm::SMLoc getRangeStart() const;
  llvm::SMLoc getRangeEnd() const;

  /// Recursively dig through noop paren nodes (if present) to find what is
  /// inside of them.
  ExprNode *getWithoutParens();
  const ExprNode *getWithoutParens() const {
    return const_cast<ExprNode *>(this)->getWithoutParens();
  }

  /// Return true if this is a TupleNode with no subexpressions.
  bool isEmptyTuple() const;

  /// To support destructuring patterns in comptime expressions, this value
  /// binds the RHS of the comptime declaration to the pattern being declared,
  /// e.g. consider: "comptime for (a, b) in list:", we destructure the result
  /// of __next__ into the pattern (a, b).
  virtual llvm::LogicalResult emitDestructuringPValue(PValue value,
                                                      IREmitter &emitter) const;

  /// Emit this expression as a match pattern against `subject`.
  ///
  /// On success, returns a boolean CValue that is true when the subject
  /// matches this pattern (and any bindings introduced by the pattern have
  /// been declared in the current scope). The default implementation rejects
  /// the expression as an invalid pattern.
  ///
  /// `patternKind` is the enclosing `var`/`ref` binding mode for this pattern
  /// (or `kNone` when none applies). Unary `var`/`ref` patterns update it and
  /// pass it down to their subpattern; binding sites such as identifiers
  /// consume it to decide how to declare.
  virtual CValue emitMatch(IREmitter &emitter, CValue subject,
                           PatternDeclKind patternKind) const;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// ExprDest indicates information about where to emit the expression result
  /// into, e.g. the a/b target in `def f(): (a,b) = (1,2)`.  On success, the
  /// ExprDest /must/ be emitted into.
  virtual AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const = 0;

  /// emitLValueIfImplicitlyTyped can return one of three things:
  ///   1) A CValue if it is inherently typed.
  ///   2) A (potentially changed!) ExprNode* if it is contextually typed,
  ///      unresolvable until the initializer expression is known.
  ///   3) Failure if the expression is invalid.
  ///
  /// This method is used with assignment expressions, which need to infer the
  /// LHS from the RHS when the LHS is unresolved, and the RHS from the LHS when
  /// it is known:
  ///
  ///     _, x = foo() # infer typeof _ and x from foo()
  ///     y = []       # infer type of [] from type of y.
  ///
  struct ELVIITResult {
    // Error cases can often propagate in a null AnyValue.
    ELVIITResult(AnyValue value) : storage(value) {}
    ELVIITResult(CValue value) : ELVIITResult(AnyValue(value)) {}
    ELVIITResult(LValueContextualType value) : storage(value) {}
    ELVIITResult(const ExprNode *value) : storage(value) {
      assert(value && "Cannot init with null ExprNode*");
    }
    /*implicit*/ ELVIITResult() : ELVIITResult(AnyValue()) {}

    bool isFailure() const {
      if (auto *ptr = std::get_if<AnyValue>(&storage))
        return ptr->isNull();
      return false;
    }
    AnyValue getIfValue() const {
      if (auto *ptr = std::get_if<AnyValue>(&storage))
        return *ptr;
      return AnyValue();
    }
    const ExprNode *getIfExprNode() const {
      if (auto *ptr = std::get_if<const ExprNode *>(&storage))
        return *ptr;
      return nullptr;
    }
    LValueContextualType getIfPartiallyBoundLV() const {
      if (auto *ptr = std::get_if<LValueContextualType>(&storage))
        return *ptr;
      return LValueContextualType{ASTType(), nullptr};
    }

  private:
    /// The failure case is encoded as a null AnyValue
    std::variant<AnyValue, const ExprNode *, LValueContextualType> storage;
  };

  /// If this expression is an LValue with an inherent type, emit it and return
  /// it. Otherwise return an ExprNode* to further resolve (with emitIR) or
  /// emit an error and return failure.
  ///
  /// 'kind' indicates whether this pattern is implicitly scoped to the function
  /// or whether it is explicitly a 'var' or 'ref' pattern.
  ///
  /// 'hasInferrableRHS' indicates whether the type of the pattern is allowed to
  /// have unbound parameters (e.g. `x: ParametricList` without explicit
  /// parameters), which happens when the type can be inferred from an
  /// initializer.
  virtual ELVIITResult
  emitLValueIfImplicitlyTyped(IREmitter &emitter, PatternDeclKind kind,
                              bool hasInferrableRHS) const {
    return ELVIITResult(this);
  }
};

/// This class is a helper class used for expression nodes might be resolvable
/// to a concrete type without an explicit initializer. This allows them to
/// simplify their implementation by providing a single hook for both the
/// "speculative" and non-speculative paths.
class LValueCapableExprNode : public ExprNode {
public:
  LValueCapableExprNode(Kind kind) : ExprNode(kind) {}

  // A default implementation is provided for these that forward to emitIR.
  ELVIITResult
  emitLValueIfImplicitlyTyped(IREmitter &emitter, PatternDeclKind kind,
                              bool hasInferrableRHS) const override;
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;

  virtual ELVIITResult emitLCVIR(ExprDest &dest, IREmitter &emitter,
                                 bool isSpeculative) const = 0;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPRNODE_H
