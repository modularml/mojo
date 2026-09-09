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

#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Casting.h"
#include <cstddef>

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Support/DebugInfoDialect/IR/DebugInfo.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

/// Return the scope from a ValueOp's location, recursively walking up through a
/// chain of inlined locations if needed.
static ErrorOr<DIScopeAttr> getValueOpLocationScope(Location loc) {
  DIScopeAttr scope;
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    // Since ValueOp belongs to a single variable declaration, nothing should
    // ever give it a fused location.
    ArrayRef<Location> locations = fusedLoc.getLocations();
    if (size_t numLocs = locations.size(); numLocs != 1) {
      return Error(
          "with fused location must reference a single location, got " +
          Twine(numLocs));
    }

    // FusedLoc _may_ contain the scope.
    scope = dyn_cast_or_null<DIScopeAttr>(fusedLoc.getMetadata());
    loc = locations[0];
  }

  // If not dealing with an inlined location, we return a scope (if found).
  auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc);
  if (!callSiteLoc)
    return scope;

  // Otherwise, we walk up the inlining chain.
  return getValueOpLocationScope(callSiteLoc.getCallee());
}

/// Returns whether the child scope is nested inside the ancestor scope.
static bool IsSubScope(DIScopeAttr child, DIScopeAttr ancestor) {
  if (child == ancestor) // short-circuit for common case.
    return true;

  return child
      .walk([&](DIScopeAttr scope) {
        if (scope == ancestor)
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

/// The local variable type must match the value type. Compare the types while
/// unwrapping debuginfo types.
static LogicalResult verifyValueOpType(ValueOp op) {
  Type inputType = op.getValue().getType();

  // All occurrences of debuginfo.expr.irvalue in the location conversion
  // expression must have types that match the ir (input) type.
  auto conversionExpr = op.getConversionExprAttr();
  auto walkResult = conversionExpr.walk([&](DIIRValueExprAttr irValue) {
    // We can only compare types if the irValue type is not yet lowered into a
    // DIType.
    Type leafType = irValue.getType();
    if (!isa<DIType>(leafType)) {
      if (leafType != inputType) {
        op->emitOpError("conversion expression leaf type ")
            << leafType << " does not match actual IR Value type " << inputType;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  // Consider providing hooks for user dialects to check the output type of the
  // DI expression against the DIType of the variable.
  return success();
}

static ParseResult parseValueOpOperand(OpAsmParser &p,
                                       DIExprAttr &conversionExpr,
                                       OpAsmParser::UnresolvedOperand &operand,
                                       Type &operandType) {
  // Parse as generic Attribute and cast, because DIExprAttr is an interface
  // (not a concrete attr) and lacks the static `name` field that the templated
  // parseOptionalAttribute now requires.
  Attribute genericAttr;
  auto parseResult = p.parseOptionalAttribute(genericAttr);
  bool hasExplicitDIExpression = parseResult.has_value();
  if (hasExplicitDIExpression) {
    conversionExpr = dyn_cast<DIExprAttr>(genericAttr);
    if (!conversionExpr)
      return p.emitError(p.getCurrentLocation(),
                         "expected a DIExprAttr interface attribute");
  }

  if (p.parseEqual() || p.parseOperand(operand) || p.parseColon() ||
      p.parseType(operandType))
    return failure();

  // If no explicit DI expression, create an "identity" conversion.
  if (!hasExplicitDIExpression)
    conversionExpr = DIIRValueExprAttr::get(operandType);
  return success();
}

static void printValueOpOperand(OpAsmPrinter &p, ValueOp value,
                                DIExprAttr conversionExpr, Value operand,
                                Type operandType) {
  // Omit identity conversion.
  bool isIdentityConversion = false;
  if (auto irValue = llvm::dyn_cast<DIIRValueExprAttr>(conversionExpr))
    if (irValue.getType() == operandType)
      isIdentityConversion = true;

  if (!isIdentityConversion) {
    p.printAttribute(conversionExpr);
    p << ' ';
  }
  p << "= ";
  p.printOperand(operand);
  p << " : ";
  p.printType(operandType);
}

LogicalResult ValueOp::verify() {
  if (failed(verifyValueOpType(*this)))
    return failure();

  ErrorOr<DIScopeAttr> locationScopeOr = getValueOpLocationScope(getLoc());
  if (locationScopeOr.isError())
    return emitOpError(locationScopeOr.getError());

  DILocalVariableAttr varAttr = getValueInfo();
  if (DIScopeAttr locationScope = *locationScopeOr) {
    if (!IsSubScope(locationScope, varAttr.getScope())) {
      return emitOpError("location scope must be a child scope of the variable "
                         "scope:\n")
             << locationScope << "\n vs. \n"
             << varAttr.getScope();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfo.cpp.inc"
