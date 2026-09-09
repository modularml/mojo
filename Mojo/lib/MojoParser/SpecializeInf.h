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
// Specialization inference helpers for closure conformance.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_SPECIALIZEINFERENCE_H
#define KGEN_MOJOPARSER_SPECIALIZEINFERENCE_H

#include "InferenceState.h"
#include "OverloadSet.h"

namespace M::KGEN::LIT {

class ExprNode;

class SpecializeInf : public InferenceState {
public:
  SpecializeInf(ASTDecl &declScope, const ExprNode *expr,
                ArrayRef<Type> declaredParamTypes,
                PogListAttr declaredParamPogs, SMLoc defaultLoc,
                bool discardError);

  void setInitialInferredValue(size_t paramIdx, TypedAttr paramVal) {
    setInferredValue(paramIdx, paramVal);
  }

  FailureOr<SmallVector<TypedAttr>>
  inferSpecialization(FnTypeGeneratorType target, FnOp actualFn);

  /// Like the `FnOp` overload, but with an already-substituted actual
  /// signature (e.g. leading `__call__` aux rebound to storage aliases).
  FailureOr<SmallVector<TypedAttr>>
  inferSpecialization(FnTypeGeneratorType target, FnTypeGeneratorType actualSig,
                      ArrayRef<ParamDeclAttr> actualParams);

private:
  bool isExplicitlyUnbound(size_t) const override { return false; }

  LogicalResult matchArgument(Type actualType, ArgConvention actualConvention,
                              size_t argIdx, ASTType expectedType,
                              ArgConvention expectedConvention,
                              PogListAttr argPogs);
  LogicalResult matchValueType(ASTType actualType, size_t argIdx,
                               ASTType expectedType, PogListAttr argPogs);

  const ExprNode *expr;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_SPECIALIZEINFERENCE_H
