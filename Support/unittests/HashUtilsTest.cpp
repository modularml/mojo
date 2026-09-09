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

#include "Support/HashUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/xxhash.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>

using namespace mlir;
using namespace M;

TEST(HashUtils, GetBytecodeHashBasic) {
  MLIRContext context;
  OpBuilder builder(&context);

  // Create a simple operation
  auto loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp = mlir::ModuleOp::create(loc);

  FailureOr<std::string> result = getBytecodeHash(*moduleOp);
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result->empty());

  // Hash should be 32 chars (128 bits = 16 bytes = 32 hex chars)
  // NB: As of 2025-02-22, the hash is actually 31 chars, but the implementation
  // zero pads to 32.
  ASSERT_EQ(result->size(), 32);
}

TEST(HashUtils, GetBytecodeHashEquivalentOps) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // Create two equivalent operations
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp1 = mlir::ModuleOp::create(loc);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp2 = mlir::ModuleOp::create(loc);

  FailureOr<std::string> hash1 = getBytecodeHash(*moduleOp1);
  FailureOr<std::string> hash2 = getBytecodeHash(*moduleOp2);

  EXPECT_TRUE(succeeded(hash1));
  EXPECT_TRUE(succeeded(hash2));

  // Equivalent ops should have same hash
  ASSERT_EQ(*hash1, *hash2);
}

TEST(HashUtils, GetBytecodeHashDifferentOps) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // Create two different operations
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp1 = mlir::ModuleOp::create(loc);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp2 = mlir::ModuleOp::create(loc);

  // Add something to moduleOp2 to make it different
  builder.setInsertionPointToStart(moduleOp2->getBody());
  ModuleOp::create(builder, loc);

  FailureOr<std::string> hash1 = getBytecodeHash(*moduleOp1);
  FailureOr<std::string> hash2 = getBytecodeHash(*moduleOp2);

  EXPECT_TRUE(succeeded(hash1));
  EXPECT_TRUE(succeeded(hash2));

  // Different ops should have different hashes
  ASSERT_NE(*hash1, *hash2);
}

TEST(HashUtils, GetBytecodeHashIgnoresLocation) {
  MLIRContext context;
  OpBuilder builder(&context);

  // Create two ops with different locations
  auto loc1 = builder.getUnknownLoc();
  auto loc2 = FileLineColLoc::get(builder.getStringAttr("test.mlir"), 1, 1);

  mlir::OwningOpRef<mlir::ModuleOp> moduleOp1 = mlir::ModuleOp::create(loc1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp2 = mlir::ModuleOp::create(loc2);

  FailureOr<std::string> hash1 = getBytecodeHash(*moduleOp1);
  FailureOr<std::string> hash2 = getBytecodeHash(*moduleOp2);

  EXPECT_TRUE(succeeded(hash1));
  EXPECT_TRUE(succeeded(hash2));

  // Hashes should be equal despite different locations
  ASSERT_EQ(*hash1, *hash2);
}

// Guards the x86 runtime dispatcher wiring (xxh_x86dispatch): the streaming
// XXH3 update must produce bit-identical output regardless of which SIMD
// backend (scalar/SSE2/AVX2/AVX-512) the dispatcher selects at runtime.
TEST(HashUtils, StreamingHashMatchesReference) {
  // Large enough to exercise the XXH3 long/streaming SIMD path (well past the
  // 240-byte short-input cutoff and spanning multiple internal stripe blocks),
  // which is where the dispatcher's AVX2/AVX-512 accumulate kicks in.
  std::vector<uint8_t> data(8192);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<uint8_t>((i * 131u + 7u) & 0xFFu);

  raw_xxhash64_stream stream64;
  raw_xxhash128_stream stream128;

  // Feed the bytes through the streams in irregular chunks (cycling chunk
  // sizes until the buffer is consumed) so the streaming update crosses
  // internal buffer and stripe boundaries.
  const size_t chunkSizes[] = {1, 7, 64, 257, 1024, 4096};
  size_t offset = 0;
  for (size_t i = 0; offset < data.size(); ++i) {
    size_t n =
        std::min(chunkSizes[i % std::size(chunkSizes)], data.size() - offset);
    stream64.write(reinterpret_cast<const char *>(data.data() + offset), n);
    stream128.write(reinterpret_cast<const char *>(data.data() + offset), n);
    offset += n;
  }

  // Anchor both dispatched streaming results to LLVM's independent,
  // non-dispatched XXH3 implementation (llvm::xxh3_* never goes through the
  // x86 dispatcher). Equality proves the dispatcher emits the canonical XXH3
  // value on this host, rather than just agreeing with itself.
  uint64_t streamed64 = stream64.hash();
  uint64_t reference64 = llvm::xxh3_64bits(llvm::ArrayRef<uint8_t>(data));
  EXPECT_EQ(streamed64, reference64);

  std::array<uint8_t, 16> streamed128 = stream128.hash();
  llvm::XXH128_hash_t reference = llvm::xxh3_128bits(data.data(), data.size());
  std::array<uint8_t, 16> reference128;
  std::memcpy(&reference128[0], &reference.low64, 8);
  std::memcpy(&reference128[8], &reference.high64, 8);
  EXPECT_EQ(streamed128, reference128);
}
