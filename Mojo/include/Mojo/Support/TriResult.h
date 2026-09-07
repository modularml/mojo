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
// This file defines TriResult, a three-valued result type modeled on TriBool
// where each state can carry its own payload. It is intended for APIs that
// answer yes / no / unknown, where each state may carry a payload.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_TRIRESULT_H
#define KGEN_SUPPORT_TRIRESULT_H

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

namespace M::KGEN {

/// A three-valued result: yes, no, or unknown. Each state can carry a
/// payload; a `void` payload means the state carries nothing, and only the
/// requested payloads occupy storage.
///
/// This mirrors `TriBool` but allows per-state payloads, similar to how
/// `FailureOr<T>` carries a value on success. `TriResult<T>` (a single
/// template argument) is the common case of a payload on `yes` only.
///
/// A payload is the whole point: use `TriBool` when no state carries one.
/// There is deliberately no conversion between the two, and no TriBool
/// algebra here: Combining two answers has to decide what happens to their
/// payloads, which is the caller's call, not this type's.
template <typename Yes = void, typename No = void, typename Unknown = void>
class [[nodiscard]] TriResult {
  struct Unit {};

  // A payload slot that is empty for `void` payloads, so states without a
  // payload cost nothing to store.
  template <typename T>
  using Slot = std::conditional_t<std::is_void_v<T>, Unit, T>;

  // Distinct per-state storage types, so the states stay distinguishable
  // even when several payloads are `void`.
  struct YesState {
    Slot<Yes> value;
  };
  struct NoState {
    Slot<No> value;
  };
  struct UnknownState {
    Slot<Unknown> value;
  };

public:
  /// Construct a `yes` result carrying `value`.
  static TriResult yes(Slot<Yes> value)
    requires(!std::is_void_v<Yes>)
  {
    return TriResult(YesState{std::move(value)});
  }
  /// Construct a `yes` result with no payload.
  static TriResult yes()
    requires std::is_void_v<Yes>
  {
    return TriResult(YesState{});
  }

  /// Construct a `no` result carrying `value`.
  static TriResult no(Slot<No> value)
    requires(!std::is_void_v<No>)
  {
    return TriResult(NoState{std::move(value)});
  }
  /// Construct a `no` result with no payload.
  static TriResult no()
    requires std::is_void_v<No>
  {
    return TriResult(NoState{});
  }

  /// Construct an `unknown` result carrying `value`.
  static TriResult unknown(Slot<Unknown> value)
    requires(!std::is_void_v<Unknown>)
  {
    return TriResult(UnknownState{std::move(value)});
  }
  /// Construct an `unknown` result with no payload.
  static TriResult unknown()
    requires std::is_void_v<Unknown>
  {
    return TriResult(UnknownState{});
  }

  /// True when the result is `yes`.
  bool isYes() const { return std::holds_alternative<YesState>(storage); }

  /// True when the result is `no`.
  bool isNo() const { return std::holds_alternative<NoState>(storage); }

  /// True when the result is `unknown`.
  bool isUnknown() const {
    return std::holds_alternative<UnknownState>(storage);
  }

  /// Access the `yes` payload. Only valid when `isYes()`.
  Slot<Yes> &getYes() &
    requires(!std::is_void_v<Yes>)
  {
    assert(isYes() && "getYes() called on a non-yes result");
    return std::get<YesState>(storage).value;
  }
  const Slot<Yes> &getYes() const &
    requires(!std::is_void_v<Yes>)
  {
    assert(isYes() && "getYes() called on a non-yes result");
    return std::get<YesState>(storage).value;
  }
  Slot<Yes> getYes() &&
    requires(!std::is_void_v<Yes>)
  {
    assert(isYes() && "getYes() called on a non-yes result");
    return std::get<YesState>(std::move(storage)).value;
  }

  /// Access the `no` payload. Only valid when `isNo()`.
  Slot<No> &getNo() &
    requires(!std::is_void_v<No>)
  {
    assert(isNo() && "getNo() called on a non-no result");
    return std::get<NoState>(storage).value;
  }
  const Slot<No> &getNo() const &
    requires(!std::is_void_v<No>)
  {
    assert(isNo() && "getNo() called on a non-no result");
    return std::get<NoState>(storage).value;
  }
  Slot<No> getNo() &&
    requires(!std::is_void_v<No>)
  {
    assert(isNo() && "getNo() called on a non-no result");
    return std::get<NoState>(std::move(storage)).value;
  }

  /// Access the `unknown` payload. Only valid when `isUnknown()`.
  Slot<Unknown> &getUnknown() &
    requires(!std::is_void_v<Unknown>)
  {
    assert(isUnknown() && "getUnknown() called on a non-unknown result");
    return std::get<UnknownState>(storage).value;
  }
  const Slot<Unknown> &getUnknown() const &
    requires(!std::is_void_v<Unknown>)
  {
    assert(isUnknown() && "getUnknown() called on a non-unknown result");
    return std::get<UnknownState>(storage).value;
  }
  Slot<Unknown> getUnknown() &&
    requires(!std::is_void_v<Unknown>)
  {
    assert(isUnknown() && "getUnknown() called on a non-unknown result");
    return std::get<UnknownState>(std::move(storage)).value;
  }

  /// Lift a boolean into a payload-free `yes`/`no` result.
  static TriResult fromBool(bool b)
    requires(std::is_void_v<Yes> && std::is_void_v<No>)
  {
    return b ? yes() : no();
  }

private:
  explicit TriResult(YesState state) : storage(std::move(state)) {}
  explicit TriResult(NoState state) : storage(std::move(state)) {}
  explicit TriResult(UnknownState state) : storage(std::move(state)) {}

  // A SmartVariant would fall back to this anyway: no state is pointer-like.
  std::variant<YesState, NoState, UnknownState> storage;
};

} // namespace M::KGEN

#endif // KGEN_SUPPORT_TRIRESULT_H
