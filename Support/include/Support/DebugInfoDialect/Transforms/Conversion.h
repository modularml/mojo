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
// This file provides support for converting debug info constructs.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_TRANSFORMS_CONVERSION_H
#define SUPPORT_DEBUGINFODIALECT_TRANSFORMS_CONVERSION_H

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <type_traits>

namespace M::DebugInfo {
/// This class enables the conversion of DebugInfo constructs in the presence of
/// type conversions. It describes how to convert the debug type information,
/// and how to update location expressions to account for type changes.
template <bool isCyclic>
class DebugInfoTypeConverterBase {
public:
  DebugInfoTypeConverterBase();

  /// Convert the given type to a debug info type. Returns a null type in the
  /// case of failure.
  DIType convertDebugType(Type type);

  /// Convert all types that occur inside `attr` into a debug info type.
  Attribute convertDebugTypesIn(Attribute attr);

  /// Attach a conversion the produces a new debug type for the given MLIR type.
  template <typename FnT,
            typename T = typename llvm::function_traits<FnT>::template arg_t<0>>
  std::enable_if_t<std::is_same_v<T, Type>> addConversion(FnT &&conversion) {
    // Wrap conversions that don't use a derived type to remove the need to
    // explicitly skip DITypes. These are handled automatically.
    replacer.addReplacement([conversion = std::forward<FnT>(conversion)](
                                Type type) -> std::optional<Type> {
      if (isa<DIType>(type))
        return std::nullopt;
      return conversion(type);
    });
  }
  template <typename FnT,
            typename T = typename llvm::function_traits<FnT>::template arg_t<0>>
  std::enable_if_t<!std::is_same_v<T, Type>> addConversion(FnT &&conversion) {
    replacer.addReplacement(std::forward<FnT>(conversion));
  }

  /// Attach a type converter, which is used to convert unresolved MLIR types as
  /// necessary.
  void addUnresolvedConverter(TypeConverter &converter);

  /// Apply the converter recursively to the given operation.
  void applyRecursively(Operation *op);

  /// Add cycle breaker in the case of known recursive replacements.
  template <typename FnT, bool C = isCyclic>
  std::enable_if_t<C> addCycleBreaker(FnT &&callback) {
    replacer.addCycleBreaker(callback);
  }

private:
  /// The underlying attr/type replacer, used to perform the actual
  /// conversions.
  std::conditional_t<isCyclic, mlir::CyclicAttrTypeReplacer,
                     mlir::AttrTypeReplacer>
      replacer;
};

using DebugInfoTypeConverter = DebugInfoTypeConverterBase<true>;
using DebugInfoNonCyclicTypeConverter = DebugInfoTypeConverterBase<false>;

/// Populate conversion patterns for transforming debug info dialect operations
/// in the presence of type conversions.
void populateTypeConversionPatterns(RewritePatternSet &patterns,
                                    DebugInfoTypeConverter &diConverter,
                                    TypeConverter &converter);

} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_TRANSFORMS_CONVERSION_H
