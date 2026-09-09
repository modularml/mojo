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

#ifndef SUPPORT_HASHUTILS_H
#define SUPPORT_HASHUTILS_H

#include "Support/LogicalResult.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

// Pull in xxHash *without* XXH_INLINE_ALL so the streaming XXH3 functions
// resolve to external symbols provided by @xxhash. XXH_INLINE_ALL would make
// XXH_PUBLIC_API mean `static`, which is incompatible with the x86 runtime
// dispatcher (xxh_x86dispatch) wired up in HashUtils.cpp, whose *_dispatch
// symbols have external linkage. XXH_STATIC_LINKING_ONLY exposes the full
// XXH3_state_t definition that the streams below store by value.
//
// NOTE: this header must not be included in the same translation unit as the
// deprecated "xxh3.h" (which forces XXH_INLINE_ALL) — the two are mutually
// exclusive. The x86 dispatcher itself is included only in HashUtils.cpp, so
// its XXH3_*_update macro rewrites do not leak into consumers of this header.
#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace M {

/// A raw_ostream that hashes content using the xxhash 128-bit algorithm.
class raw_xxhash128_stream : public llvm::raw_ostream {
  XXH3_state_t State;

  /// See raw_ostream::write_impl. Defined out-of-line in HashUtils.cpp so the
  /// x86 dispatcher's XXH3 update macro rewrites stay confined to that file.
  void write_impl(const char *Ptr, size_t Size) override;

public:
  raw_xxhash128_stream();

  raw_xxhash128_stream(raw_xxhash128_stream &&other)
      : State(std::move(other.State)) {}

  std::array<uint8_t, 16> hash();
  std::string hashString();

  uint64_t current_pos() const override { return 0; }
};

/// A raw_ostream that hashes content using the xxhash 64-bit algorithm.
class raw_xxhash64_stream : public llvm::raw_ostream {
  XXH3_state_t State;

  /// See raw_ostream::write_impl. Defined out-of-line in HashUtils.cpp so the
  /// x86 dispatcher's XXH3 update macro rewrites stay confined to that file.
  void write_impl(const char *Ptr, size_t Size) override;

public:
  raw_xxhash64_stream();

  raw_xxhash64_stream(raw_xxhash64_stream &&other)
      : State(std::move(other.State)) {}

  uint64_t hash();

  uint64_t current_pos() const override { return 0; }
};

/// Compute the xxhash for the given MLIR operation.
/// The hash ignores source location information.
FailureOr<std::string> getBytecodeHash(mlir::Operation *op);

/// This function computes the bytecode hash similar to getBytecodeHash, but
/// does so by hashing each individual operation in the module in parallel
/// (via BytecodeHasher) and then combining the results into a single hash.
FailureOr<std::string> getModuleBytecodeHash(mlir::ModuleOp module);

} // namespace M
#endif // SUPPORT_HASHUTILS_H
