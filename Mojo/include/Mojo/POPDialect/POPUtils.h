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
// This file declares utility functions primarily for parsing, printing and
// verifying POP related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPUTILS_H
#define KGEN_POPDIALECT_POPUTILS_H

#include "Mojo/KGENDialect/FoldUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::POP {

/// Get the value of a scalar index-like parameter value.
/// This is a temporary helper utility during the Int->SIMD unification project.
/// After it's done, we should remove the IntegerAttr case.
ErrorOr<int64_t> getScalarIndexValue(TypedAttr value);

/// Fold a cast between two SIMD types.
OpFoldResult foldCast(TypedAttr operand, SIMDType resultType,
                      SIMDType inputType, SIMDType outputType,
                      std::optional<int64_t> indexBitWidth = std::nullopt);

/// Fold a SIMD Or-reduction operation.
OpFoldResult foldSIMDReduceOr(Value vectorVal, Attribute vectorAttr,
                              SIMDType vectorType);
/// Fold a SIMD And-reduction operation.
OpFoldResult foldSIMDReduceAnd(Value vectorVal, Attribute vectorAttr,
                               SIMDType vectorType);

/// Fold a SIMD left-shift operation.
FoldValue foldSIMDShl(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD right-shift operation.
FoldValue foldSIMDShr(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD abs operation.
FoldValue foldSIMDAbs(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD round operation.
OpFoldResult foldSIMDRound(Attribute val, TargetInfoAttr targetInfo);

/// Fold a SIMD div operation.
FoldValue foldSIMDDiv(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD floordiv operation.
FoldValue foldSIMDFloorDiv(FoldValues operands, TargetInfoAttr target = {});

/// Interpret a memcpy operation.
ErrorTreeOrSuccess interpretMemcpy(Attribute dst, Attribute src, Attribute len,
                                   Location loc, InterpreterState &state);
/// Interpret a memcpy operation.
ErrorTreeOrSuccess interpretMemcpy(Attribute dst, Attribute src, Attribute len,
                                   Location loc,
                                   ParametricInterpreterState &state);

} // namespace M::KGEN::POP

#endif // KGEN_POPDIALECT_POPUTILS_H
