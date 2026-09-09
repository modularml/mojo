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
// This file defines an IPInt class, which is a wrapper around APInt to
// represent (memory-bounded) infinite precision integers.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_IPINT_H
#define SUPPORT_IPINT_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace M {

/// IPInt is a wrapper around APInt to represent infinite precision integers.
/// The main motivation to write this was that APInts can't be compared for
/// equality when they have different bit widths (it instead raises an assertion
/// error).  But additionally this defines standard arithmetic operations on
/// infinite precision integers.
///
/// The underlying APInt can be extracted from an IPInt, and IPInt ensures that
/// the APInt always contains exactly the minimum bit width to represent the
/// number as a signed 2's compliment integer.
class IPInt {
public:
  /// The wrapped APInt is normalized to use the minimum number of bits so that
  /// equality testing works.
  IPInt(const llvm::APInt val) : val(val.trunc(val.getSignificantBits())) {}
  IPInt(const IPInt &val) : val(val.val) {}
  IPInt(const uint64_t uintV) {
    llvm::APInt v = llvm::APInt(64, uintV, false);
    val = v.trunc(v.getSignificantBits());
  }
  IPInt() : val() {}

  const llvm::APInt &getAPInt() const { return val; }

  IPInt &operator=(const IPInt &rhs) {
    val = rhs.getAPInt();
    return *this;
  }

  bool operator==(const IPInt &rhs) const;
  bool operator!=(const IPInt &rhs) const;
  bool operator<(const IPInt &rhs) const;
  bool operator<=(const IPInt &rhs) const;
  bool operator>(const IPInt &rhs) const;
  bool operator>=(const IPInt &rhs) const;
  IPInt operator+(const IPInt &rhs) const;
  IPInt operator-(const IPInt &rhs) const;
  IPInt operator-() const;
  IPInt operator*(const IPInt &rhs) const;
  IPInt operator/(const IPInt &rhs) const;
  IPInt operator%(const IPInt &rhs) const;
  IPInt operator<<(const IPInt &rhs) const;
  IPInt operator>>(const IPInt &rhs) const;
  IPInt operator&(const IPInt &rhs) const;
  IPInt operator|(const IPInt &rhs) const;
  IPInt operator^(const IPInt &rhs) const;
  IPInt abs() const;
  IPInt exponentiate(const IPInt &rhs) const;
  IPInt gcd(const IPInt &rhs) const;

  friend llvm::hash_code hash_value(const IPInt &arg) {
    return llvm::hash_value(arg.val);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const IPInt &arg) {
    return os << arg.getAPInt();
  }

private:
  enum class BinOp {
    kAdd,
    kSub,
    kMul,
    kDiv,
    kMod,
    kLshift,
    kRshift,
    kAnd,
    kOr,
    kXor,
  };
  enum class CmpOp {
    kSgt,
    kSge,
    kSlt,
    kSle,
  };

  IPInt binop(const IPInt &rhs, IPInt::BinOp whichOp) const;
  bool cmp(const IPInt &rhs, IPInt::CmpOp whichOp) const;

  llvm::APInt val;
};

} // namespace M

#endif // SUPPORT_IPINT_H
