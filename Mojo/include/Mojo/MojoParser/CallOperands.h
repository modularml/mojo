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
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CALLOPERANDS_H
#define KGEN_MOJOPARSER_CALLOPERANDS_H

#include "Mojo/MojoParser/ExprDest.h"

namespace M::KGEN {
class PogListAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {

//===----------------------------------------------------------------------===//
// CallSyntax
//===----------------------------------------------------------------------===//

/// When emitting a function call, this enum is used to indicate why the call
/// happened in the first place.  This allows producing better-tuned
/// diagnostics.
enum class CallSyntax : uint8_t {
  kParamBindings,      //< symbol[x, val=y]  (not actually a call).
  kDirectCall,         //< f()
  kIndirectCall,       //< expr()
  kMethodCall,         //< x.f()
  kTypeCall,           //< T()
  kOperator,           //< -x and x + y
  kReversedOperator,   //< y + x          (where the method was looked up on x).
  kSubscript,          // v[1, 2]
  kAttribute,          // v.x             (where x is not a static member of v).
  kImplicitConvert,    //< Conversion in an argument context
  kImplicitCopyCtor,   //< Implicit copy constructor call.
  kImplicitMoveCtor,   //< Implicit move constructor call.
  kDestructor,         //< Destructor due to a value definition.
  kTupleGetItem,       //< Call to getitem in a tuple assignment.
  kMethodCallSynthetic //< Call to a method for synthetic checks.
};

StringRef stringifyCallSyntax(CallSyntax val);
raw_ostream &operator<<(raw_ostream &os, CallSyntax val);

//===----------------------------------------------------------------------===//
// CallOperands
//===----------------------------------------------------------------------===//

/// This is an operand record, maintaining the IR repre that might
struct OperandValue : public ASTExprAnd<AnyValue> {
  // Null for positional arguments.
  StringAttr keyword;
  // This indicates whether the operand is a keyword argument or a positional
  // argument and if it is unpacked.
  ArgUnpackStyle unpackStyle;

  OperandValue(StringAttr keyword, ASTExprAnd<AnyValue> value,
               ArgUnpackStyle unpackStyle)
      : ASTExprAnd<AnyValue>(std::move(value)), keyword(keyword),
        unpackStyle(unpackStyle) {
    assert((keyword != StringAttr()) ==
               (unpackStyle == ArgUnpackStyle::kKeyword) &&
           "Keyword is present iff keyword argument");
  }
};

using OperandValueList = SmallVector<OperandValue, 4>;

/// Struct that carries information necessary to look up and emit functions and
/// method calls. This includes the destination to emit into, the operands (both
/// positional and keyword), and the syntax of the call.
class CallOperands {
public:
  /// Initialize with the call. The syntax and an expression node are required
  /// this constructor supports an optional list of positional operands as a
  /// convenience, but those can be added later as well.
  CallOperands(CallSyntax syntax, const ExprNode *callExpr, ExprDest &&dest,
               ArrayRef<ASTExprAnd<AnyValue>> posOperands = {})
      : syntax(syntax), callExpr(callExpr), dest(std::move(dest)) {
    for (const auto &operand : posOperands)
      add(operand);
  }

  // Initialize with an existing CallOperands and a new destination.
  CallOperands(const CallOperands &existing, ExprDest &&dest)
      : syntax(existing.syntax), callExpr(existing.callExpr),
        dest(std::move(dest)), values(existing.values),
        hasSelfOperand(existing.hasSelfOperand) {}

  CallOperands(CallOperands &&) = default;
  CallOperands &operator=(CallOperands &&) = default;

  /// Return a keyword argument value if present, or null otherwise.
  const OperandValue *findKwArg(StringAttr keyword) const {
    assert(keyword && "cannot look up null keyword");
    for (auto &elt : values) {
      if (elt.keyword == keyword)
        return &elt;
    }
    return nullptr;
  }

  /// Return the number of positional operands.
  size_t getNumPositional() const {
    size_t result = 0;
    for (auto &value : values)
      if (!value.keyword)
        ++result;
    return result;
  }

  /// Return the number of keyword operands.
  size_t getNumKwOperands() const { return values.size() - getNumPositional(); }

  /// This is the syntax the operand list is being used for.
  CallSyntax syntax;

  /// This is the expression representing the overall call.
  const ExprNode *callExpr;

  const ExprNode *getExpr() const { return callExpr; }
  llvm::SMLoc getExprLoc() const;

  /// This is the location the call is going to be emitted into.  This can
  /// include information about the expected result type, the origin of the
  /// destination etc.
  ExprDest dest;

  /// The values passed in.  The keyword field will be null for positional
  /// arguments and present for keyword operands.
  OperandValueList values;

  /// Indicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

  //===--------------------------------------------------------------------===//
  // Element Accessors
  //===--------------------------------------------------------------------===//

  bool empty() const { return values.empty(); }
  size_t size() const { return values.size(); }

  const OperandValue &operator[](size_t index) const { return values[index]; }
  OperandValue &operator[](size_t index) { return values[index]; }

  //===--------------------------------------------------------------------===//
  // Manipulators
  //===--------------------------------------------------------------------===//

  /// Add a positional argument to the list.
  void add(ASTExprAnd<AnyValue> value,
           ArgUnpackStyle unpackStyle = ArgUnpackStyle::kPositional) {
    values.emplace_back(StringAttr(), std::move(value), unpackStyle);
  }

  /// Add a keyword argument, there can never be conflicts here because keyword
  /// argument conflicts should be checked in the parser before any semantic
  /// analysis is attempted.
  void add(StringAttr name, ASTExprAnd<AnyValue> value,
           ArgUnpackStyle unpackStyle) {
    values.push_back({name, std::move(value), unpackStyle});
  }

  /// This adds a "self" argument to the start of the positional argument list.
  void addSelf(ASTExprAnd<AnyValue> value) {
    assert(!hasSelfOperand && "Cannot add a self when one is already present");
    values.insert(values.begin(),
                  {StringAttr(), value, ArgUnpackStyle::kPositional});
    hasSelfOperand = true;
  }

  //===--------------------------------------------------------------------===//
  // Diagnostic helpers.
  //===--------------------------------------------------------------------===//

  void dump() const;

  struct PogAssignment {
    enum {
      /// This POG is unspecified because it is implicit (eg a result slot) or
      /// in a parameter list where parameters can get inferred.
      kPA_Unspecified = -1,
      /// This POG takes its default value.
      kPA_Default = -2,
      /// This POG is a variadic argument, whose elements are specified by the
      /// lists below.
      kPA_Variadic = -3,
    };

    /// This array contains one entry for every POG in the signature.  It
    /// indicates the operand index that is assigned to that POG.  Parameter
    /// lists may have unbound parameters and call operand list may have result
    /// slots.  These will generally be labeled as kPA_Unspecified.  Variadic
    /// lists have out-of-line representation and are marked by kPA_Variadic.
    SmallVector<ssize_t> operandIdxs;

    /// This is a list of operand indexes that are assigned to a PosVarArg or
    /// PackVarArg (only one of which may be present). This is only non-empty in
    /// the presence of a variadic keyword argument, but may be empty in the
    /// case that no keyword arguments are passed.
    SmallVector<size_t> posVariadicIdxs;

    /// This is a list of operand indexes that are assigned to a KWArg POG. This
    /// is only non-empty in the presence of a variadic keyword argument, but
    /// may be empty in the case that no keyword arguments are passed.
    SmallVector<size_t> kwVariadicIdxs;
  };

  /// Validate the operand list against the signature indicated by
  /// pogListAttr, emitting an error with "getDiag" if invalid.
  ///
  /// This populates "pogAssignment" with information about the mapping of
  /// operands to POG entries.
  LogicalResult assignToPogs(
      PogListAttr pogListAttr, bool isParameterList,
      PogAssignment &pogAssignment,
      llvm::function_ref<MojoInflightDiag &(llvm::SMLoc)> getDiag) const;
};

raw_ostream &operator<<(raw_ostream &os, const CallOperands &value);

//===----------------------------------------------------------------------===//
// OperandsNeedingOriginsList
//===----------------------------------------------------------------------===//

/// Parameter inference is used to evaluate whether a set of operands can work
/// for a callee, and determine a set of parameter bindings to use for it.
///
/// In that process, it may find that it could select the candidate if a
/// non-memory operand(eg a PValue or SRValue) were to be dumped into memory.
/// This list keeps track of those cases.
struct OperandNeedingOrigin {
  size_t operandIdx; // The index of the operand in the call operands list.
  size_t argIdx;     // The index of the argument in `signature`.
  ASTType expectedArgType; // The expected RValue type of the argument.

  // The signature that `argIdx` indexes into.  This is not necessarily the
  // signature of the call being evaluated: when the operand only fits after an
  // implicit conversion, this is the signature of that conversion's
  // constructor.  Consider:
  //
  //   struct RW[o: ImmOrigin](Copyable, Movable):
  //
  //     @implicit
  //     def __init__[vo: ImmOrigin, //](ref[vo] value: Int, out self: RW[vo]):
  //         self.n = value
  //
  // def grw(w: RW) -> Int:
  //     pass
  //
  // def main():
  //     var a, b = 1, 2
  //     _ = grw(a + b)
  //
  // `a + b` will need to be spilled to memory to get an origin to construct a
  // `RW`, so the spill has to use the implicit ctor's convention for `value`
  // (ref) rather than `grw`'s convention for `w` (readMem), since we want to
  // eventually emit `grw(RW(a + b))`.  Emission derives the convention from
  // this signature, which also tells it whether `argIdx` is variadic.
  FnTypeGeneratorType signature;

  // TODO: figure out the subtleties related to variadic convention etc, then we
  // can collapse signature and argIdx into a single ArgConvention field.
  ArgConvention getArgConvention() const {
    FnTypeGeneratorType sig = signature;
    if (sig.isPosVarArg(argIdx) || sig.isPack(argIdx))
      return sig.getVariadicConvention(argIdx);
    return sig.getArgConvention(argIdx);
  }

  enum {
    /// This is a sentinel representing the the "operand" that needs spilling is
    /// actually the ExprDest of the call, not an actual operand.
    kExprDestOperandIdx = ~1ULL,
  };
};

using OperandsNeedingOriginsList = std::vector<OperandNeedingOrigin>;

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CALLOPERANDS_H
