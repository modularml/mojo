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
// This file defines an IPRational class, which uses two IPInts to
// represent arbitrary precision rational numbers.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_IPRATIONAL_H
#define SUPPORT_IPRATIONAL_H

#include "Support/IPInt.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace M {

/// IPRational class for arbitrary precision rational numbers.  Represented as
/// two IPInts, one for each of numerator and denominator.
class IPRational {
public:
  /// Constructs a rational with the given numerator and denominator.  The user
  /// is responsible for ensuring that the denominator is never zero.
  IPRational(const IPInt n, const IPInt d);
  /// Constructs a rational with the given numerator and denominator of 1.
  IPRational(const IPInt &numerator);
  /// Constructs a rational with same numerator and denominator as the given
  /// rational.
  IPRational(const IPRational &val);
  /// Constructs a rational with denominator 1 and numerator from the given
  /// APInt.
  IPRational(const llvm::APInt &val);
  /// Constructs a rational with denominator 1 and numerator from the given
  /// uint.
  IPRational(uint64_t val);
  IPRational();

  const IPInt &getNumerator() const;
  const IPInt &getDenominator() const;

  IPRational &operator=(const IPRational &rhs);

  bool operator==(const IPRational &rhs) const;
  bool operator!=(const IPRational &rhs) const;
  bool operator<(const IPRational &rhs) const;
  bool operator<=(const IPRational &rhs) const;
  bool operator>(const IPRational &rhs) const;
  bool operator>=(const IPRational &rhs) const;
  IPRational operator+(const IPRational &rhs) const;
  IPRational operator-(const IPRational &rhs) const;
  IPRational operator-() const;
  IPRational operator*(const IPRational &rhs) const;
  IPRational operator/(const IPRational &rhs) const;
  IPRational abs() const;

  friend llvm::hash_code hash_value(const IPRational &arg);

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const IPRational &arg);

private:
  IPInt numerator;
  IPInt denominator;
};

} // namespace M

#endif // SUPPORT_IPRATIONAL_H
