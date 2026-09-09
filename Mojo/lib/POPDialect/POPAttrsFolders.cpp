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
// This file contains evaluation/folding implementations for POP attributes.
// These methods implement
// ContextuallyEvaluatedAttrInterface::evaluateWithContext.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPUtils.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// SIMDAbsAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDAbsAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getOperand()};
  return foldAttrWithTarget(context, operands, foldSIMDAbs);
}

//===----------------------------------------------------------------------===//
// SIMDShlAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDShlAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getVal(), getShft()}; // spellchecker:disable-line
  return foldAttrWithTarget(context, operands, foldSIMDShl);
}

//===----------------------------------------------------------------------===//
// SIMDShrAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDShrAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getVal(), getShft()}; // spellchecker:disable-line
  return foldAttrWithTarget(context, operands, foldSIMDShr);
}
