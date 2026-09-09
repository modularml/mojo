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
// This file contains a set of commonly used keys and generic type
// infra to create custom keys by composition.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_SUPPORT_KEYS_H
#define CACHE_SUPPORT_KEYS_H

#include "Support/Buffer.h"
#include "Support/HashUtils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BLAKE3.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

namespace mlir {
class Operation;
} // namespace mlir

namespace M::Cache::Keys {
template <typename T>
struct TypeKey : std::false_type {};

/// A simple key that takes a StringRef and returns the string from it
/// without any hashing.
template <>
struct TypeKey<llvm::StringRef> {
  using KeyTy = llvm::StringRef;
  static std::string hashKey(KeyTy key) { return key.str(); }
};
using StringKey = TypeKey<llvm::StringRef>;

template <>
struct TypeKey<llvm::ArrayRef<uint8_t>> {
  using KeyTy = llvm::ArrayRef<uint8_t>;
  static std::string hashKey(KeyTy key) {

    llvm::BLAKE3 hashState{};
    hashState.update(key);

    auto hash = hashState.final();
    return {hash.begin(), hash.end()};
  }
};
using ArrayKey = TypeKey<llvm::ArrayRef<uint8_t>>;
template <>
struct TypeKey<M::BufferRef> {
  using KeyTy = M::BufferRef;
  static std::string hashKey(KeyTy key) {
    llvm::BLAKE3 hashState{};
    hashState.update(key->getBuffer());

    auto hash = hashState.final();
    return {hash.begin(), hash.end()};
  }
};
using BufferKey = TypeKey<M::BufferRef>;

template <>
struct TypeKey<mlir::Operation *> {
  using KeyTy = mlir::Operation *;
  static std::string hashKey(KeyTy key) { return *M::getBytecodeHash(key); }
};
using OperationKey = TypeKey<mlir::Operation *>;

template <typename... Ts>
struct VariantTypeKey {
  using KeyTy = std::variant<Ts...>;

  static std::string hashKey(KeyTy key) {
    std::string hashedKey;

    // Go through the types and if any of them belongs to the variant
    // get key for it.
    (getUnderlyingHash<Ts>(std::forward<KeyTy>(key), hashedKey) || ...);
    return hashedKey;
  }

private:
  template <typename T>
  static bool getUnderlyingHash(KeyTy key, std::string &out) {
    if (auto val = std::get_if<T>(&key)) {
      out = TypeKey<T>::hashKey(*val);
      return true;
    }
    return false;
  }
};

/// Wrap a given key generator with one or more wrappers. Wrappers need to
/// implement a static function wrapKey which takes a string and returns a
/// string back. Wrapping works like this.
///        1. Generate key with key generator
///        2. Call Wrappers::wrapKey(...) on generated key.
///        3. Repeat for all wrappers with previous result.
///        4. Once again hash the final accumulated string.
/// Wrapping takes place in the order in which it is defined
template <typename KeyGen, typename... Wrappers>
class WrappedKey {
public:
  using KeyTy = typename KeyGen::KeyTy;

  static std::string hashKey(KeyTy key) {
    std::string hashedKey = KeyGen::hashKey(std::forward<KeyTy>(key));
    // Apply all the wrappers.
    ([&]() mutable {
      hashedKey = Wrappers::wrapKey(hashedKey);
      return true;
    }() &&
     ...);
    llvm::BLAKE3 hashState{};
    hashState.update(hashedKey);
    auto hash = hashState.final();
    return {hash.begin(), hash.end()};
  }
};

/// Provide a key that doesn't do any hashing - we only want to read things from
/// keys provided to this.
using ReadOnlyKey = TypeKey<llvm::StringRef>;
} // namespace M::Cache::Keys

#endif // CACHE_SUPPORT_KEYS_H
