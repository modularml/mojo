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

#include "Support/IPInt.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

using namespace M;

bool IPInt::cmp(const IPInt &rhs, IPInt::CmpOp whichOp) const {
  const llvm::APInt &lOrig = getAPInt();
  const llvm::APInt &rOrig = rhs.getAPInt();
  bool useLWidth = lOrig.getBitWidth() > rOrig.getBitWidth();
  bool useRWidth = lOrig.getBitWidth() < rOrig.getBitWidth();
  llvm::APInt lExt = useLWidth ? lOrig : lOrig.sext(rOrig.getBitWidth());
  llvm::APInt rExt = useRWidth ? rOrig : rOrig.sext(lOrig.getBitWidth());

  switch (whichOp) {
  case IPInt::CmpOp::kSgt:
    return lExt.sgt(rExt);
  case IPInt::CmpOp::kSge:
    return lExt.sge(rExt);
  case IPInt::CmpOp::kSlt:
    return lExt.slt(rExt);
  case IPInt::CmpOp::kSle:
    return lExt.sle(rExt);
  }
  llvm_unreachable("Unknown compare operation.");
}

IPInt IPInt::binop(const IPInt &rhs, IPInt::BinOp whichOp) const {
  const llvm::APInt &lOrig = getAPInt();
  const llvm::APInt &rOrig = rhs.getAPInt();

  // Get the width to use for the operation.
  unsigned widthUse = lOrig.getBitWidth();
  switch (whichOp) {
  case IPInt::BinOp::kAdd:
  case IPInt::BinOp::kSub:
  case IPInt::BinOp::kMul:
  case IPInt::BinOp::kDiv:
    widthUse = lOrig.getBitWidth() + rOrig.getBitWidth();
    break;
  case IPInt::BinOp::kLshift:
  case IPInt::BinOp::kRshift: {
    llvm::APInt rAbs = rOrig.abs();
    uint64_t rTrunc = rAbs.getZExtValue();
    widthUse = lOrig.getBitWidth() + rTrunc;
  } break;
  case IPInt::BinOp::kAnd:
  case IPInt::BinOp::kOr:
  case IPInt::BinOp::kXor:
  case IPInt::BinOp::kMod:
    widthUse = lOrig.getBitWidth() > rOrig.getBitWidth() ? lOrig.getBitWidth()
                                                         : rOrig.getBitWidth();
    break;
  }

  llvm::APInt lExt = lOrig.sextOrTrunc(widthUse);
  llvm::APInt rExt = rOrig.sextOrTrunc(widthUse);
  llvm::APInt result;

  switch (whichOp) {
  case IPInt::BinOp::kAdd:
    result = lExt.sadd_sat(rExt);
    break;
  case IPInt::BinOp::kSub:
    result = lExt.ssub_sat(rExt);
    break;
  case IPInt::BinOp::kMul:
    result = lExt.smul_sat(rExt);
    break;
  case IPInt::BinOp::kDiv:
    result = lExt.sdiv(rExt);
    break;
  case IPInt::BinOp::kMod:
    result = lExt.srem(rExt);
    break;
  case IPInt::BinOp::kLshift:
    result = lExt.shl(rExt);
    break;
  case IPInt::BinOp::kRshift:
    result = lExt.ashr(rExt);
    break;
  case IPInt::BinOp::kAnd:
    result = lExt & rExt;
    break;
  case IPInt::BinOp::kOr:
    result = lExt | rExt;
    break;
  case IPInt::BinOp::kXor:
    result = lExt ^ rExt;
    break;
  }

  return IPInt(result);
}

bool IPInt::operator==(const IPInt &rhs) const {
  return (getAPInt().getBitWidth() == rhs.getAPInt().getBitWidth()) &&
         (getAPInt() == rhs.getAPInt());
}
bool IPInt::operator!=(const IPInt &rhs) const { return !(*this == rhs); }
bool IPInt::operator<(const IPInt &rhs) const {
  return cmp(rhs, IPInt::CmpOp::kSlt);
}
bool IPInt::operator<=(const IPInt &rhs) const {
  return cmp(rhs, IPInt::CmpOp::kSle);
}
bool IPInt::operator>(const IPInt &rhs) const {
  return cmp(rhs, IPInt::CmpOp::kSgt);
}
bool IPInt::operator>=(const IPInt &rhs) const {
  return cmp(rhs, IPInt::CmpOp::kSge);
}
IPInt IPInt::operator+(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kAdd);
}
IPInt IPInt::operator-(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kSub);
}
IPInt IPInt::operator-() const { return IPInt(0) - (*this); }
IPInt IPInt::operator*(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kMul);
}
IPInt IPInt::operator/(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kDiv);
}
IPInt IPInt::operator%(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kMod);
}
IPInt IPInt::operator<<(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kLshift);
}
IPInt IPInt::operator>>(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kRshift);
}
IPInt IPInt::operator&(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kAnd);
}
IPInt IPInt::operator|(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kOr);
}
IPInt IPInt::operator^(const IPInt &rhs) const {
  return binop(rhs, IPInt::BinOp::kXor);
}
IPInt IPInt::abs() const {
  if (*this < IPInt(0))
    return IPInt(0) - *this;
  else
    return IPInt(*this);
}
IPInt IPInt::gcd(const IPInt &rhs) const {
  // Euclid's algorithm for GCD:
  // https://en.wikipedia.org/wiki/Euclidean_algorithm
  IPInt l(*this);
  IPInt r(rhs);
  while (r != 0) {
    IPInt tmp(r);
    r = l % r;
    l = tmp;
  }
  return l;
}

/// This is limited to exponentiating with non-negative RHS when the number to
/// exponentiate is 0.
IPInt IPInt::exponentiate(const IPInt &rhs) const {
  IPInt base(*this);
  assert((base != 0 || rhs >= 0) && "'0 ** n' is undefined for negative n");
  if (rhs < 0) {
    if (base == 1)
      return IPInt(1);
    // (-1) ** n is 1 for even n, -1 for odd n
    if (base == -1)
      return IPInt((rhs.abs() & 1) == 0 ? 1 : -1);
    return IPInt(0);
  }
  // I looked up and found this fast exponentiation algorithm here:
  // https://mathstats.uncg.edu/sites/pauli/112/HTML/secfastexp.html#algfastexp
  IPInt result = 1;
  IPInt exp = rhs;
  while (exp != 0) {
    if (exp % 2 == 1)
      result = result * base;
    exp = exp / 2;
    base = base * base;
  }
  return result;
}

// namespace M
