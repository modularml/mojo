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
// This file defines TriBool, a three-valued (Kleene) logic result: provably
// true, provably false, or not statically decidable. It provides the small
// three-valued algebra (Kleene AND/OR and an AND-fold) in one place.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_TRIBOOL_H
#define KGEN_SUPPORT_TRIBOOL_H

#include <cstdint>
#include <optional>

namespace M::KGEN {

/// A three-valued (Kleene) logic result: a predicate is provably true (`yes`),
/// provably false (`no`), or not statically decidable (`unknown`).
class [[nodiscard]] TriBool {
  enum class State : uint8_t { False, Unknown, True } state;
  constexpr explicit TriBool(State s) : state(s) {}

public:
  static constexpr TriBool yes() { return TriBool(State::True); }
  static constexpr TriBool no() { return TriBool(State::False); }
  static constexpr TriBool unknown() { return TriBool(State::Unknown); }
  static constexpr TriBool fromBool(bool b) { return b ? yes() : no(); }

  constexpr bool isTrue() const { return state == State::True; }
  constexpr bool isFalse() const { return state == State::False; }
  constexpr bool isUnknown() const { return state == State::Unknown; }
  /// True when the result is definitely known.
  constexpr bool isDefinite() const { return state != State::Unknown; }

  /// Returns std::nullopt when unknown, otherwise the definite boolean value.
  constexpr std::optional<bool> toOptionalBool() const {
    return isUnknown() ? std::nullopt : std::optional<bool>(isTrue());
  }

  /// Kleene AND: any `no` -> `no`; else any `unknown` -> `unknown`; else `yes`.
  constexpr TriBool operator&(TriBool o) const {
    if (isFalse() || o.isFalse())
      return no();
    if (isUnknown() || o.isUnknown())
      return unknown();
    return yes();
  }
  constexpr TriBool &operator&=(TriBool o) { return *this = *this & o; }

  /// Kleene OR: any `yes` -> `yes`; else any `unknown` -> `unknown`; else `no`.
  constexpr TriBool operator|(TriBool o) const {
    if (isTrue() || o.isTrue())
      return yes();
    if (isUnknown() || o.isUnknown())
      return unknown();
    return no();
  }
  constexpr TriBool &operator|=(TriBool o) { return *this = *this | o; }

  friend constexpr bool operator==(TriBool a, TriBool b) {
    return a.state == b.state;
  }
  friend constexpr bool operator!=(TriBool a, TriBool b) { return !(a == b); }
};

/// Fold a range of TriBool values under Kleene AND ("all must hold"). An empty
/// range yields `yes()` (vacuous truth).
template <class Range>
constexpr TriBool allTrue(Range &&r) {
  TriBool acc = TriBool::yes();
  for (TriBool x : r) {
    acc &= x;
    // Short-circuit: once a definite `no` is seen the result can't change.
    if (acc.isFalse())
      break;
  }
  return acc;
}

} // namespace M::KGEN

#endif // KGEN_SUPPORT_TRIBOOL_H
