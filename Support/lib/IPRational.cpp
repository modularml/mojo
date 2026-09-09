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

#include "Support/IPRational.h"
#include "Support/IPInt.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace M;

enum class BinOp : uint8_t {
  kAdd,
  kSub,
  kMul,
  kDiv,
};
enum class CmpOp : uint8_t {
  kSgt,
  kSge,
  kSlt,
  kSle,
};

IPRational::IPRational(IPInt n, IPInt d) {
  // When constructing, we always canonicalize by using the greatest
  // common denominator of the numerator and denominator.
  assert(d != 0 && "denominator should never be zero");
  IPInt gcd = n.gcd(d);
  numerator = n / gcd;
  denominator = d / gcd;
  if (denominator < 0) {
    numerator = numerator * -1;
    denominator = denominator * -1;
  }
}
IPRational::IPRational(const IPInt &numerator)
    : numerator(numerator), denominator(IPInt(1)) {}
IPRational::IPRational(const IPRational &val)
    : numerator(val.numerator), denominator(val.denominator) {}
IPRational::IPRational(const llvm::APInt &val)
    : numerator(IPInt(val)), denominator(IPInt(1)) {}
IPRational::IPRational(uint64_t val)
    : numerator(IPInt(val)), denominator(IPInt(1)) {}
IPRational::IPRational() : numerator(IPInt(0)), denominator(IPInt(1)) {}

const IPInt &IPRational::getNumerator() const { return numerator; }
const IPInt &IPRational::getDenominator() const { return denominator; }

IPRational &IPRational::operator=(const IPRational &rhs) {
  numerator = rhs.getNumerator();
  denominator = rhs.getDenominator();
  return *this;
}

/// Perform comparison operation.
static bool cmp(const IPRational &lhs, const IPRational &rhs, CmpOp whichOp) {
  switch (whichOp) {
  case CmpOp::kSgt:
    return lhs.getNumerator() * rhs.getDenominator() >
           lhs.getDenominator() * rhs.getNumerator();
  case CmpOp::kSge:
    return lhs == rhs || cmp(lhs, rhs, CmpOp::kSgt);
  case CmpOp::kSlt:
    return !cmp(lhs, rhs, CmpOp::kSgt) && lhs != rhs;
  case CmpOp::kSle:
    return lhs == rhs || cmp(lhs, rhs, CmpOp::kSlt);
  }
  llvm_unreachable("Unknown compare operation.");
}

/// Perform binary arithmetic operation.
static IPRational binop(const IPRational &lhs, const IPRational &rhs,
                        BinOp whichOp) {
  const IPInt &lN = lhs.getNumerator();
  const IPInt &lD = lhs.getDenominator();
  const IPInt &rN = rhs.getNumerator();
  const IPInt &rD = rhs.getDenominator();

  switch (whichOp) {
  case BinOp::kAdd:
    return IPRational(lN * rD + rN * lD, lD * rD);
  case BinOp::kSub:
    return IPRational(lN * rD - rN * lD, lD * rD);
  case BinOp::kMul:
    return IPRational(lN * rN, lD * rD);
  case BinOp::kDiv:
    assert(rN != 0 && "can't divide by zero");
    return IPRational(lN * rD, lD * rN);
  }
  llvm_unreachable("Unknown compare operation.");
}

bool IPRational::operator==(const IPRational &rhs) const {
  return (getNumerator() == rhs.getNumerator()) &&
         (getDenominator() == rhs.getDenominator());
}
bool IPRational::operator!=(const IPRational &rhs) const {
  return !(*this == rhs);
}
bool IPRational::operator<(const IPRational &rhs) const {
  return cmp(*this, rhs, CmpOp::kSlt);
}
bool IPRational::operator<=(const IPRational &rhs) const {
  return cmp(*this, rhs, CmpOp::kSle);
}
bool IPRational::operator>(const IPRational &rhs) const {
  return cmp(*this, rhs, CmpOp::kSgt);
}
bool IPRational::operator>=(const IPRational &rhs) const {
  return cmp(*this, rhs, CmpOp::kSge);
}
IPRational IPRational::operator+(const IPRational &rhs) const {
  return binop(*this, rhs, BinOp::kAdd);
}
IPRational IPRational::operator-(const IPRational &rhs) const {
  return binop(*this, rhs, BinOp::kSub);
}
IPRational IPRational::operator-() const { return IPRational(0) - *this; }
IPRational IPRational::operator*(const IPRational &rhs) const {
  return binop(*this, rhs, BinOp::kMul);
}
IPRational IPRational::operator/(const IPRational &rhs) const {
  return binop(*this, rhs, BinOp::kDiv);
}
IPRational IPRational::abs() const {
  if (getNumerator() < 0)
    return IPRational(-1) * (*this);
  return IPRational(*this);
}

llvm::hash_code M::hash_value(const IPRational &arg) {
  return llvm::hash_combine(arg.getNumerator(), arg.getDenominator());
}

llvm::raw_ostream &M::operator<<(llvm::raw_ostream &os, const IPRational &arg) {
  // TODO(#23387): MLIR's AsmParser doesn't have `parseSlash` or a more generic
  // way to parse literal strings/characters, so we will use the pipe "|"
  // character instead.
  return os << arg.getNumerator() << "|" << arg.getDenominator();
}

// namespace M
