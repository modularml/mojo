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

#include "Cache/CachedTransform.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Support/FileSystemExtras.h"
#include "Support/Preprocessor.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "gtest/gtest.h"

using namespace M;
using namespace Cache;
using namespace AsyncRT;
using namespace mlir;

namespace {
static TempDir createTempDir() {
  auto tempDirOr = TempDir::create("cache-transform-test.%%%%%%");
  assert(!tempDirOr.isError());
  return tempDirOr.takeValue();
}

class TestPass
    : public mlir::PassWrapper<TestPass, OperationPass<mlir::func::FuncOp>> {
public:
  TestPass(bool *actuallyRun) : actuallyRun(actuallyRun) {}
  using PassWrapper::PassWrapper;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPass)

  /// A diagnostic message emitted when the pass is run.
  static constexpr StringLiteral kDiagnosticMessage =
      "TestPass was run on the module.";

  void runOnOperation() override {
    if (!*actuallyRun)
      assert(false && "should not run the pass!");
    // Get the return and put a specific attribute on it.
    func::FuncOp func = getOperation();
    auto returnOp =
        cast<func::ReturnOp>(func.getFunctionBody().front().getTerminator());
    // Remove the attr if it's already there.
    if (returnOp->hasAttr("hello"))
      returnOp->removeAttr("hello");
    else
      returnOp->setAttr("hello", StringAttr::get(&getContext(), "world"));

    // Emit a diagnostic message.
    func.emitRemark(kDiagnosticMessage);
  }

  bool *actuallyRun;
};

struct TestPassDiagnosticValidator : public mlir::ScopedDiagnosticHandler {
  TestPassDiagnosticValidator(MLIRContext *ctx) : ScopedDiagnosticHandler(ctx) {
    setHandler([&](Diagnostic &diag) {
      EXPECT_TRUE(StringRef(diag.str()).contains(TestPass::kDiagnosticMessage));
      foundExpectedDiagnostic = true;
    });
  }

  /// Flag indicating if the expected diagnostic was emitted.
  bool foundExpectedDiagnostic = false;
};
} // namespace

// We can have transform functions that write to a buffer and return a buffer.
// Here we are testing the transform return value is the value that's returned
// on a cache miss and output of cacheHitFn on cache hit.
TEST(CachedTransformTest, BufferReturn) {
  TempDir tempDir = createTempDir();
  CPUDeviceRef cpuDevice = getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
                                                CPUDeviceOptions().forDebug());
  auto transformBackendChainOr =
      getLocalDefaultBackendChain(tempDir.getPath() / "xform");
  EXPECT_FALSE(failed(transformBackendChainOr));
  auto transformCache = RCRef<BlobCache<TransformCacheKey>>::create(
      transformBackendChainOr.takeValue());

  static constexpr StringLiteral world = "world";
  int runCount = 0;
  auto transform = [&](AnyAsyncValueRef inputChain) mutable {
    ++runCount;
    auto outputBuffer =
        AsyncValueRef<BufferRef>::allocate(cpuDevice->getCompactPtr());
    auto inner = [&, output = outputBuffer.copy()]() mutable {
      BufferRef outputBuffer = Buffer::get(world);
      return std::move(output).emplace(std::move(outputBuffer));
    };
    std::move(inputChain).andThenSync(std::move(inner));
    return outputBuffer;
  };
  int hitCount = 0;
  auto hitFn = [&](BufferRef buf) {
    ++hitCount;
    return buf.copy();
  };

  std::string hashKey;
  const AsyncValueRef<Chain> &inputChain = cpuDevice->getReadyChain();
  constexpr StringLiteral keyStr = "hello";
  WriteableBufferRef key = WriteableBuffer::get(0, {}, keyStr.size());
  key->write(keyStr.data(), keyStr.size());
  EncodedLocation loc = AsyncRT::UnknownLocationDecoder::getEncodedLocation();

  // First call should generate new hash
  AnyAsyncValueRef output =
      cachedTransform(loc.copy(), transformCache.copy(), inputChain.copy(),
                      key.copy(), transform, hitFn, true, &hashKey);
  await(output);

  ASSERT_TRUE(output.isType<BufferRef>());
  auto &outputBuffer = output.get<BufferRef>();
  EXPECT_EQ(outputBuffer->getBuffer(), world);
  EXPECT_EQ(runCount, 1);
  EXPECT_FALSE(hashKey.empty()) << "Hash key should be populated on first run";
  std::string initialHash = hashKey;

  // Second call should reuse same hash
  hashKey.clear();
  const AsyncValueRef<Chain> &inputChain2 = cpuDevice->getReadyChain();
  AnyAsyncValueRef output2 =
      cachedTransform(loc.copy(), transformCache.copy(), inputChain2.copy(),
                      key.copy(), transform, hitFn, true, &hashKey);
  await(output2);

  ASSERT_TRUE(output2.isType<BufferRef>());
  auto &outputBuffer2 = output2.get<BufferRef>();
  EXPECT_EQ(outputBuffer2->getBuffer(), world);
  EXPECT_EQ(runCount, 1);
  EXPECT_EQ(hitCount, 1);
  EXPECT_EQ(hashKey, initialHash) << "Hash should match initial value";

  // Third call with same key
  hashKey.clear();
  const AsyncValueRef<Chain> &inputChain3 = cpuDevice->getReadyChain();
  AnyAsyncValueRef output3 =
      cachedTransform(loc.copy(), transformCache.copy(), inputChain3.copy(),
                      key.copy(), transform, hitFn, true, &hashKey);
  await(output3);

  ASSERT_TRUE(output3.isType<BufferRef>());
  auto &outputBuffer3 = output3.get<BufferRef>();
  EXPECT_EQ(outputBuffer3->getBuffer(), world);
  EXPECT_EQ(runCount, 1);
  EXPECT_EQ(hitCount, 2);
  EXPECT_EQ(hashKey, initialHash) << "Hash should remain consistent";

  // Fourth call with modified hit function but same key
  hashKey.clear();
  constexpr llvm::StringLiteral prependStr = " again ";
  auto anotherHitFn = [&](BufferRef buf) {
    ++hitCount;

    WriteableBufferRef output = WriteableBuffer::get();
    output->write(prependStr.data(), prependStr.size());
    output->write(buf->getBufferStart(), buf->getBufferSize());
    BufferRef outputBuffer = std::move(output);
    return outputBuffer;
  };

  const AsyncValueRef<Chain> &inputChain4 = cpuDevice->getReadyChain();
  AnyAsyncValueRef output4 =
      cachedTransform(loc.copy(), transformCache.copy(), inputChain4.copy(),
                      key.copy(), transform, anotherHitFn, true, &hashKey);
  await(output4);

  ASSERT_TRUE(output4.isType<BufferRef>());
  auto &outputBuffer4 = output4.get<BufferRef>();
  EXPECT_EQ(outputBuffer4->getBuffer(), (prependStr + world).str());
  EXPECT_EQ(runCount, 1);
  EXPECT_EQ(hitCount, 3);
  EXPECT_EQ(hashKey, initialHash)
      << "Hash should persist through hit function changes";
}

// The cache key must be relocatable: two operations that are identical except
// for the absolute file paths in their source locations must produce byte-for-
// byte identical key buffers. Otherwise a cache warmed under one install path
// (e.g. one venv's site-packages) misses under a different install path even
// for byte-identical IR, forcing a full recompile of every kernel.
TEST(CachedTransformTest, KeyIgnoresSourceLocations) {
  MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();

  // Same IR, parsed identically, then re-located against two different
  // absolute paths to simulate two different package install locations.
  constexpr StringLiteral kIR = "func.func @k() { return }";

  auto buildKey = [&](StringRef path) -> std::string {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kIR, &ctx);
    EXPECT_TRUE(module) << "failed to parse test IR";
    // Stamp every op with a FileLineColLoc rooted at `path`, mirroring how
    // kernel `.mojo` files are parsed with their on-disk absolute paths.
    auto fileLoc = FileLineColLoc::get(&ctx, path, /*line=*/1, /*column=*/1);
    module->walk([&](Operation *op) { op->setLoc(fileLoc); });

    WriteableBufferRef key = WriteableBuffer::get();
    EXPECT_FALSE(failed(writeOperationToCacheKey(module.get(), key)));
    auto buf = key->getBuffer();
    return std::string(buf.data(), buf.size());
  };

  std::string keyA = buildKey("/install/a/lib/python3.14/site-packages/k.mojo");
  std::string keyB = buildKey("/install/b/lib/python3.14/site-packages/k.mojo");

  EXPECT_FALSE(keyA.empty());
  EXPECT_EQ(keyA, keyB)
      << "Cache key must not depend on absolute source-location paths";
}
