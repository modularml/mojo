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

#ifndef SUPPORT_ADT_SMARTVARIANT_H
#define SUPPORT_ADT_SMARTVARIANT_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <type_traits>
#include <variant>

namespace M {

/// A generic, discriminated union type, where, if all of the types are pointer
/// types, the discriminator is stored in the low bit of the pointer.
/// Otherwise, for non-pointer types, there is no notion of a discriminator to
/// use generally speaking.  Extend the set of pointer types by supporting the
/// concept of `PointerLike` types in `llvm`.
///
/// This implementation is extremely efficient in space due to leveraging the
/// low bits of the pointer, while exposing a natural and type-safe API.
///
/// `llvm::PointerUnion<Ts...>` can not be used generically by callers since it
/// only works when all of the types are pointer types. `SmartVariant<Ts...>`
/// allows for generically working with a variant-like type. Specifically, if
/// all of the types are pointers (i.e. there are bits we can steal to represent
/// the nullable value), the underlying storage is `PointerUnion`. Otherwise,
/// we fall back to the less space-efficient `std::variant`. `SmartVariant`
/// provides a uniform API for working with the set of types by providing the
/// LLVM casting infra: `isa`, `get`, `dyn_cast`, etc. along with a similar API
/// of checking for nullability via `isNull()` on the `SmartVariant` itself.

template <class... Ts>
class SmartVariant {
public:
  static constexpr bool CanStealBits =
      (llvm::detail::IsPointerLike<Ts>::value && ...);

private:
  using UnderlyingStorage =
      std::conditional_t<CanStealBits, llvm::PointerUnion<Ts...>,
                         std::variant<Ts...>>;

public:
  SmartVariant() = default;

  template <class... TArgs>
  SmartVariant(TArgs &&...args) : storage(std::forward<TArgs...>(args...)) {}

  // This implicit conversion operator allows for free copy-assignment,
  // move-assignment, and other operators from the underlying storage type that
  // is already implemented. We don't want to implement those ourselves that
  // just delegate, and we can't easily opt into them since the underlying
  // storage type isn't a base class, i.e. we can't write
  //
  // ```
  // using UnderlyingStorage::operator=;
  // ```
  //
  // since it's not a base class. We don't immediately want to make it a base
  // class either since there isn't a common "accessor" interface for the
  // underlying storage to use in the explicit case. We could use design by
  // introspection likely to enable that if we care, but this is easy and
  // sufficient for today's use cases.
  constexpr operator UnderlyingStorage() const { return storage; }

  UnderlyingStorage &getUnderlyingStorage() { return storage; }

  const UnderlyingStorage &getUnderlyingStorage() const { return storage; }

private:
  UnderlyingStorage storage;
};

template <class... Ts>
SmartVariant(Ts &&...) -> SmartVariant<Ts...>;

template <class... Ts>
bool operator==(const SmartVariant<Ts...> &lhs,
                const SmartVariant<Ts...> &rhs) {
  return lhs.getUnderlyingStorage() == rhs.getUnderlyingStorage();
}

template <class... Ts>
bool operator!=(const SmartVariant<Ts...> &lhs,
                const SmartVariant<Ts...> &rhs) {
  return lhs.getUnderlyingStorage() != rhs.getUnderlyingStorage();
}

template <class... Ts>
bool operator<(const SmartVariant<Ts...> &lhs, const SmartVariant<Ts...> &rhs) {
  return lhs.getUnderlyingStorage() < rhs.getUnderlyingStorage();
}

/// A helper struct to create an overloaded function object.
/// Useful when visiting a variant type with std::visit.
template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

} // namespace M

// Specialization of CastInfo for SmartVariant
namespace llvm {

template <class... Ts>
inline constexpr bool IsNullable<M::SmartVariant<Ts...>> =
    M::SmartVariant<Ts...>::CanStealBits;

template <class... Ts>
struct ValueIsPresent<M::SmartVariant<Ts...>,
                      std::enable_if_t<!M::SmartVariant<Ts...>::CanStealBits>> {
  static bool isPresent(const M::SmartVariant<Ts...> &t) { return true; }
  static decltype(auto) unwrapValue(M::SmartVariant<Ts...> &t) { return t; }
};

template <class To, class... Ts>
struct CastInfo<To, M::SmartVariant<Ts...>> {
  using From = M::SmartVariant<Ts...>;

  static bool isPossible(From &f) {
    if constexpr (From::CanStealBits)
      return isa<To>(f.getUnderlyingStorage());
    else
      return std::holds_alternative<To>(f.getUnderlyingStorage());
  }

  /// Perform the cast from `From` to `To`. If the underlying storage is a
  /// pointer, return a value. Otherwise, return a reference.
  static std::conditional_t<From::CanStealBits, To, To &> doCast(From &f) {
    if constexpr (From::CanStealBits)
      return cast<To>(f.getUnderlyingStorage());
    else
      return std::get<To>(f.getUnderlyingStorage());
  }

  /// Perform the cast from `From` to `To`. If the underlying storage is a
  /// pointer, return a value. Otherwise, return a reference.
  static std::conditional_t<From::CanStealBits, To, const To &>
  doCast(const From &f) {
    if constexpr (From::CanStealBits)
      return cast<To>(f.getUnderlyingStorage());
    else
      return std::get<To>(f.getUnderlyingStorage());
  }

  static To doCastIfPossible(From &f) {
    if (!isPossible(f))
      return castFailed();
    return doCast(f);
  }

  static To castFailed() { return {}; }
};

template <class To, class... Ts>
struct CastInfo<To, const M::SmartVariant<Ts...>>
    : public ConstStrippingForwardingCast<
          To, const M::SmartVariant<Ts...>,
          CastInfo<To, M::SmartVariant<Ts...>>> {};

} // namespace llvm

#endif // SUPPORT_ADT_SMARTVARIANT_H
