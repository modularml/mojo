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

#ifndef SUPPORT_FUNCTION_EXTRAS_H
#define SUPPORT_FUNCTION_EXTRAS_H

#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include <functional>
#include <type_traits>
#include <utility>

namespace M {
/// This wraps a callable that may return a type or may return void, and allows
/// you to define custom return behavior in the case that the wrapped callable
/// returns void. The way you customize the return behavior is to provide a type
/// `Default` that has a static `get` method, and the result of that is what
/// will be returned. `Default` may be void, in which case this returns void.
template <typename Default, typename F, typename... Args,
          typename Result = std::invoke_result_t<F, Args...>>
static std::conditional_t<!std::is_void_v<Result>, Result, Default>
invokeWithDefaultResultType(F &&f, Args &&...args) {
  if constexpr (std::is_void_v<Result>) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    if constexpr (std::is_void_v<Default>)
      return;
    else
      return Default::get();
  } else {
    return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }
}

/// Provides a default type that can be used with `invokeWithDefaultResultType`.
/// This struct is fairly common, and so provided here. Its usage would be:
///
///  template<typename T>
///  ErrorOr<T> someFunc() {
///    ...
///    return invokeWithDefaultResultType<DefaultSuccess>(func, args...);
///  }
///
/// Where `someFunc` could be called with `T = void`.
struct DefaultSuccess {
  operator ErrorOrSuccess() { return success(); }
  static DefaultSuccess get() { return DefaultSuccess{}; }
};

/// Generate a forwarding call wrapper for the function `f`. Calling this
/// wrapper is equivalent to invoking f with its last `sizeof...(Args)`
/// parameters bound to args. I.e. `bind_back(f, bound_args...)(call_args...)`
/// is equivalent to `std::invoke(f, call_args..., bound_args...)`.
/// This function is meant to emulate the std::bind_back in C++23.
template <class F, class... TBoundArgs>
[[nodiscard]] constexpr auto bind_back(F &&fn, TBoundArgs &&...back_args) {
  static_assert((std::is_move_constructible_v<TBoundArgs> && ...));
  return [=, f = std::forward<F>(fn)](auto &&...front_args) {
    return std::invoke(f, front_args..., back_args...);
  };
}

} // namespace M

#endif // SUPPORT_ADT_FUNCTIONEXTRAS_H
