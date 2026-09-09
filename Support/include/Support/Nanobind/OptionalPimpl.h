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

#ifndef SUPPORT_NANOBIND_OPTIONALPIMPL_H
#define SUPPORT_NANOBIND_OPTIONALPIMPL_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#include <optional>
#include <utility>

namespace M::Graph::Python {

/// Specialize `is_optional_pimpl` for a type to mark it as an pimpl type
/// allowing a null implementation
template <typename T, typename SFINAE = void>
struct is_optional_pimpl {
  constexpr static bool value = false;
};

template <>
struct is_optional_pimpl<mlir::Attribute> {
  constexpr static bool value = true;
};

template <>
struct is_optional_pimpl<mlir::Type> {
  constexpr static bool value = true;
};

template <typename T>
inline constexpr bool is_optional_pimpl_v = is_optional_pimpl<T>::value;

/// Decorator around a function which may return a nullable PImpl type, such as
/// mlir::Attribute or mlir::Type.
/// - For an optional PImpl type, wrap the type as `std::optional<T>` and return
///     an empty optional. Nanobind will know to return a `None` in this case.
/// - For any other type, no-op.
/// Overload for function pointers
template <typename Return, typename... Args>
auto optional_pimpl(Return (*func)(Args...)) {
  if constexpr (is_optional_pimpl_v<Return>) {
    return [=](Args... args) {
      if (auto ret = std::invoke(func, std::forward<Args>(args)...)) {
        return std::optional<Return>{std::move(ret)};
      }
      return std::optional<Return>{};
    };
  } else {
    return func;
  }
}

/// Overload for const pointer to member functions
template <typename Class, typename Return, typename... Args>
auto optional_pimpl(Return (Class::*func)(Args...) const) {
  if constexpr (is_optional_pimpl_v<Return>) {
    return [=](Class *obj, Args... args) {
      if (auto ret = std::invoke(func, obj, std::forward<Args>(args)...)) {
        return std::optional<Return>{std::move(ret)};
      }
      return std::optional<Return>{};
    };
  } else {
    return func;
  }
}

/// Overload for non-const pointer to member functions
template <typename Class, typename Return, typename... Args>
auto optional_pimpl(Return (Class::*func)(Args...)) {
  if constexpr (is_optional_pimpl_v<Return>) {
    return [=](Class *obj, Args... args) {
      if (auto ret = std::invoke(func, obj, std::forward<Args>(args)...)) {
        return std::optional<Return>{std::move(ret)};
      }
      return std::optional<Return>{};
    };
  } else {
    return func;
  }
}

} // namespace M::Graph::Python

#endif // SUPPORT_NANOBIND_OPTIONALPIMPL_H
