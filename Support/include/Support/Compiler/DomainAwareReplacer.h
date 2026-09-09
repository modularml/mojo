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

#ifndef SUPPORT_COMPILER_DOMAINAWAREREPLACER_H
#define SUPPORT_COMPILER_DOMAINAWAREREPLACER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/CyclicReplacerCache.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace M {

/// A CyclicAttrTypeReplacer that allows different replacing logic under
/// different domains. A domain is a user-defined integer.
///
/// When registering a replacer, a domain can be associated with it so that
/// the replacer is only triggered under that domain. When invoking "replace"
/// on an attr/type, a domain is specified.
///
/// Replacer cache is predicated on the domain to ensure correctness.
///
/// When user calls `replace`, a FailureOr<T> is returned to propagate the
/// error, it is guaranteed that when `succeeded(FailureOr<T>) == true`, T !=
/// nullptr.
///
/// Inside the replacer, a nullptr is used to indicate an error has occurred to
/// be compatible with MLIR APIs.
class DomainAwareReplacer {
public:
  using DomainId = size_t;

  DomainAwareReplacer();

  //===--------------------------------------------------------------------===//
  // Application
  //===--------------------------------------------------------------------===//

  mlir::FailureOr<Attribute> replace(Attribute element, DomainId domain) {
    Attribute ret = cachedReplaceImpl(element, domain);
    if (!ret)
      return llvm::failure();
    return ret;
  }

  mlir::FailureOr<Type> replace(Type element, DomainId domain) {
    Type ret = cachedReplaceImpl(element, domain);
    if (!ret)
      return llvm::failure();
    return ret;
  }

  mlir::LogicalResult replaceElementsIn(Operation *op, DomainId domain,
                                        bool replaceAttrs = true,
                                        bool replaceLocs = false,
                                        bool replaceTypes = false);

  //===--------------------------------------------------------------------===//
  // Registration - Replacers
  //===--------------------------------------------------------------------===//

  template <typename T>
  using ReplaceFnResult = std::optional<std::pair<T, WalkResult>>;
  template <typename T>
  using ReplaceFn = std::function<ReplaceFnResult<T>(T)>;

  void addReplacement(ReplaceFn<Attribute> fn, DomainId domain);
  void addReplacement(ReplaceFn<Type> fn, DomainId domain);

  /// Register a replacement function that doesn't match the default signature,
  /// either because it uses a derived parameter type, or it uses a simplified
  /// result type.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>,
            typename ResultT = std::invoke_result_t<FnT, T>>
  std::enable_if_t<!std::is_same_v<T, BaseT> ||
                   !std::is_convertible_v<ResultT, ReplaceFnResult<BaseT>>>
  addReplacement(FnT &&callback, DomainId domain) {
    addReplacement(
        [callback = std::forward<FnT>(callback)](
            BaseT base) -> ReplaceFnResult<BaseT> {
          if (auto derived = dyn_cast<T>(base)) {
            if constexpr (std::is_convertible_v<ResultT,
                                                std::optional<BaseT>>) {
              std::optional<BaseT> result = callback(derived);
              return result ? std::make_pair(*result, WalkResult::advance())
                            : ReplaceFnResult<BaseT>();
            } else {
              return callback(derived);
            }
          }
          return ReplaceFnResult<BaseT>();
        },
        domain);
  }

  //===--------------------------------------------------------------------===//
  // Registration - CycleBreakers
  //===--------------------------------------------------------------------===//

  template <typename T>
  using CycleBreakerFn = std::function<std::optional<T>(T)>;

  void addCycleBreaker(CycleBreakerFn<Attribute> fn, DomainId domain);
  void addCycleBreaker(CycleBreakerFn<Type> fn, DomainId domain);

  /// Register a cycle-breaking function that doesn't match the default
  /// signature.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>>
  std::enable_if_t<!std::is_same_v<T, BaseT>> addCycleBreaker(FnT &&callback,
                                                              DomainId domain) {
    addCycleBreaker(
        [callback =
             std::forward<FnT>(callback)](BaseT base) -> std::optional<BaseT> {
          if (auto derived = dyn_cast<T>(base))
            return callback(derived);
          return std::nullopt;
        },
        domain);
  }

private:
  using AttrOrType = PointerUnion<Attribute, Type>;
  using CacheKey = std::pair<AttrOrType, DomainId>;

  /// Invokes the registered cycle-breaker functions from most recently
  /// registered to least recently registered until a successful result is
  /// returned.
  std::optional<const void *> breakCycleImpl(CacheKey element);

  /// Shared concrete implementation of the public `replace` functions.
  /// Could return nullptr upon failure.
  template <typename T>
  T cachedReplaceImpl(T element, DomainId domain);

  /// The set of replacement functions that map sub elements.
  DenseMap<DomainId, SmallVector<ReplaceFn<Attribute>>> attrReplacementFns;
  DenseMap<DomainId, SmallVector<ReplaceFn<Type>>> typeReplacementFns;

  /// The set of registered cycle-breaker functions.
  DenseMap<DomainId, SmallVector<CycleBreakerFn<Attribute>>>
      attrCycleBreakerFns;
  DenseMap<DomainId, SmallVector<CycleBreakerFn<Type>>> typeCycleBreakerFns;

  mlir::CyclicReplacerCache<CacheKey, const void *> cache;
};

} // namespace M

#endif // SUPPORT_COMPILER_DOMAINAWAREREPLACER_H
