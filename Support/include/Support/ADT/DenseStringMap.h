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
// Provides llvm::DenseMapInfo<std::string> so that std::string can be used as
// a key in llvm::DenseMap and llvm::DenseSet.
//
// Include this header instead of defining the specialization locally.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ADT_DENSESTRINGMAP_H
#define SUPPORT_ADT_DENSESTRINGMAP_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

#include <string>

namespace llvm {
template <>
struct DenseMapInfo<std::string> {
  // We want to be able to hash empty strings, so use `\x01` and `\x02`
  // as empty key and tombstone key.
  static std::string getEmptyKey() { return std::string(1, '\x01'); }
  static std::string getTombstoneKey() { return std::string(1, '\x02'); }
  static unsigned getHashValue(const std::string &str) {
    return llvm::hash_value(str);
  }
  static bool isEqual(const std::string &LHS, const std::string &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // SUPPORT_ADT_DENSESTRINGMAP_H
