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

#include "Cache/BlobCache.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Support/FileSystemExtras.h"
#include "Support/Preprocessor.h"
#include "Support/RCRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"
#include <thread>

using namespace M;
using namespace Cache;
using namespace AsyncRT;

namespace {

/// Basic string key info.
struct StringKeyInfo {
  using KeyTy = StringRef;

  static std::string hashKey(KeyTy key) {
    ArrayRef<uint8_t> bytes((const uint8_t *)key.data(), key.size());
    return llvm::toHex(llvm::SHA256::hash(bytes), true);
  }
};

static TempDir createTempDir() {
  auto tempDirOr = TempDir::create("cache-test.%%%%%%");
  assert(!tempDirOr.isError());
  return tempDirOr.takeValue();
}

class BlobCacheTest : public testing::Test {
protected:
  TempDir tempDir;
  AsyncRT::CPUDeviceRef cpuDevice;
  RCRef<BlobCache<StringKeyInfo>> cache;

  BlobCacheTest()
      : tempDir(createTempDir()),
        cpuDevice(getOrCreateCPUDevice(
            AsyncRT::CPUDeviceSource::Test,
            AsyncRT::CPUDeviceOptions().withLeakCheckedAllocator())),
        cache(RCRef<BlobCache<StringKeyInfo>>::create(
            getLocalDefaultBackendChain(tempDir.getPath()).takeValue())) {}
};

} // namespace

TEST_F(BlobCacheTest, NotContainItemThatHasNotBeenInserted) {
  auto done = AsyncValueRef<Chain>::allocate(*cpuDevice);
  auto contains = cache->contains(*cpuDevice, "does not exist");
  std::move(contains).andThenSync(
      [done = done.copy()](AsyncValueRef<bool> &&contains) mutable {
        ASSERT_FALSE(contains.isError())
            << contains.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(*contains)
            << "expected not to have item named 'does not exist'\n";
        std::move(done).emplace();
      });
  await(done);
}

TEST_F(BlobCacheTest, FindShouldNotReturnErrorForNonexistentItem) {
  auto done = AsyncValueRef<Chain>::allocate(*cpuDevice);
  auto dneOr = cache->find(*cpuDevice, "does not exist");
  std::move(dneOr).andThenSync(
      [done =
           done.copy()](AsyncValueRef<std::optional<BufferRef>> dneOr) mutable {
        ASSERT_FALSE(dneOr.isError())
            << dneOr.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(dneOr->has_value())
            << "expected not to have item named 'does not exist'\n";
        std::move(done).emplace();
      });
  await(done);
}

TEST_F(BlobCacheTest, ContainItemWhenInserted) {
  auto done = AsyncValueRef<Chain>::allocate(*cpuDevice);

  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  AsyncValueRef<std::string> insertOr =
      cache->insert(*cpuDevice, "zeros", std::move(zerosBuf));
  std::move(insertOr).andThenSync(
      [this, cache = cache.copy(),
       done = done.copy()](AsyncValueRef<std::string> insertOr) mutable {
        ASSERT_FALSE(insertOr.isError())
            << insertOr.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

        auto contains = cache->contains(*cpuDevice, "zeros");
        std::move(contains).andThenSync(
            [done = std::move(done)](AsyncValueRef<bool> contains) mutable {
              ASSERT_FALSE(contains.isError())
                  << contains.getDiagnostic().getMessage() << '\n';
              EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
              std::move(done).emplace();
            });
      });
  await(done);
}

TEST_F(BlobCacheTest, FindItemThatExists) {

  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  auto inserted = AsyncValueRef<Chain>::allocate(*cpuDevice);
  AsyncValueRef<std::string> insertOr =
      cache->insert(*cpuDevice, "zeros", zerosBuf.copy());
  std::move(insertOr).andThenSync(
      [this, cache = cache.copy(), inserted = inserted.copy()](
          AsyncValueRef<std::string> insertOr) mutable {
        ASSERT_FALSE(insertOr.isError())
            << insertOr.getDiagnostic().getMessage() << '\n';
        EXPECT_FALSE(insertOr->empty()) << "expected to receive the hash key\v";

        auto contains = cache->contains(*cpuDevice, "zeros");
        std::move(contains).andThenSync(
            [inserted =
                 std::move(inserted)](AsyncValueRef<bool> contains) mutable {
              ASSERT_FALSE(contains.isError())
                  << contains.getDiagnostic().getMessage() << '\n';
              EXPECT_TRUE(*contains) << "expected to have item named 'zeros'\n";
              std::move(inserted).emplace();
            });
      });
  await(inserted);

  auto found = AsyncValueRef<Chain>::allocate(*cpuDevice);
  auto zerosOr = cache->find(*cpuDevice, "zeros");
  std::move(zerosOr).andThenSync(
      [zerosBuf = zerosBuf.copy(), found = found.copy()](
          AsyncValueRef<std::optional<BufferRef>> &&zerosOr) mutable {
        ASSERT_FALSE(zerosOr.isError())
            << zerosOr.getDiagnostic().getMessage() << '\n';
        ASSERT_TRUE(zerosOr->has_value());
        BufferRef outZeros = std::move(**zerosOr);
        ASSERT_EQ(outZeros->getBufferSize(), zerosBuf->getBufferSize())
            << "output buffer size did not match input buffer size\n";
        EXPECT_TRUE(
            outZeros->getBuffer() ==
            StringRef(zerosBuf->getBufferStart(), zerosBuf->getBufferSize()))
            << "buffer returned did not match the buffer inputted\n";
        std::move(found).emplace();
      });
  await(found);
}

TEST_F(BlobCacheTest, FileSystemFindItemThatExists) {
  // Get an uninitialized buffer. We don't care what's in this, as long as it
  // goes in and comes out the same.
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);
  BufferRef zerosBuf = std::move(zerosDataBuf);

  AsyncValueRef<std::string> err =
      cache->insert(*cpuDevice, "zeros", zerosBuf.copy());
  await(err);
  ASSERT_FALSE(err.isError()) << err.getDiagnostic().getMessage() << '\n';

  // Reset the cache so that we are forced to look it up from the file system.
  auto fsCache = RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(tempDir.getPath()).takeValue());

  // Check that the cache holds the new item, and it's the same data as before.
  auto zerosOr = fsCache->find(*cpuDevice, "zeros");
  await(zerosOr);
  ASSERT_FALSE(zerosOr.isError())
      << zerosOr.getDiagnostic().getMessage() << '\n';
  ASSERT_TRUE(zerosOr->has_value());

  BufferRef outZeros = std::move(**zerosOr);
  ASSERT_TRUE(outZeros->getBufferSize() == zerosBuf->getBufferSize())
      << "output buffer size did not match input buffer size\n";
  EXPECT_TRUE(outZeros->getBuffer() ==
              StringRef(zerosBuf->getBufferStart(), zerosBuf->getBufferSize()))
      << "buffer returned did not match the buffer inputted\n";
}

TEST_F(BlobCacheTest, FileSystemTestOldVersionDeletion) {
  // Mock the existence of an old version of the cache.
  // Specifically create the directory to have a trailing path separator to
  // test canonicalization of paths when figuring out deletion criteria.
  auto tempDirectory = tempDir.getPath() / "ModularOldVersionString";

  std::error_code ec;
  std::filesystem::create_directory(tempDirectory, ec);
  ASSERT_FALSE(ec) << "failed to create directory: " << ec.message() << "\n";

  // Upon creating a new cache, all of the old versions on the filesystem
  // should be deleted.
  auto fsCache = RCRef<BlobCache<StringKeyInfo>>::create(
      getLocalDefaultBackendChain(tempDir.getPath()).takeValue());
  ASSERT_TRUE(!std::filesystem::exists(tempDirectory, ec))
      << "expected the temp directory to be deleted by cacheDir creation\n";
}

TEST_F(BlobCacheTest, OutKeyHashPopulatedOnContains) {
  std::string outHash;
  auto contains =
      cache->contains(*cpuDevice, "containsHashTest", std::nullopt, &outHash);
  await(contains);

  ASSERT_FALSE(contains.isError()) << contains.getDiagnostic().getMessage();
  EXPECT_EQ(outHash, StringKeyInfo::hashKey("containsHashTest"));
}

TEST_F(BlobCacheTest, OutKeyHashPopulatedOnFind) {
  std::string outHash;
  auto findOr = cache->find(*cpuDevice, "findHashTest", std::nullopt, &outHash);
  await(findOr);

  ASSERT_FALSE(findOr.isError()) << findOr.getDiagnostic().getMessage();
  EXPECT_EQ(outHash, StringKeyInfo::hashKey("findHashTest"));
}

TEST_F(BlobCacheTest, OutKeyHashNullSafe) {
  // Verify null outKeyHash doesn't cause errors
  auto zerosDataBuf = WriteableBuffer::get();
  zerosDataBuf->write(0);

  auto contains = cache->contains(*cpuDevice, "nullHashTest");
  auto findOr = cache->find(*cpuDevice, "nullHashTest");

  await(contains);
  await(findOr);

  ASSERT_FALSE(contains.isError()) << contains.getDiagnostic().getMessage();
  ASSERT_FALSE(findOr.isError()) << findOr.getDiagnostic().getMessage();
}

TEST_F(BlobCacheTest, InsertKeyedAsyncWorks) {
  // Create a buffer with a known value
  auto onesDataBuf = WriteableBuffer::get();
  onesDataBuf->write(1);
  BufferRef onesBuf = std::move(onesDataBuf);

  // Use a plain key for retrieval
  std::string plainKey = "insertKeyedTest";
  // Compute the hashed key which will be used for insertion
  std::string hashedKey = cache->getHash(plainKey);

  // Insert using insertKeyed; note we pass the already hashed key
  auto chainAV = cache->insertKeyed(*cpuDevice, hashedKey, onesBuf.copy());
  await(chainAV);

  // Now find using the plain key, which will compute hash
  auto findAV = cache->find(*cpuDevice, plainKey);
  await(findAV);

  ASSERT_FALSE(findAV.isError()) << findAV.getDiagnostic().getMessage();
  ASSERT_TRUE(findAV->has_value())
      << "Expected to find item inserted with insertKeyed";
  BufferRef outBuf = std::move(**findAV);
  ASSERT_EQ(outBuf->getBufferSize(), onesBuf->getBufferSize())
      << "Output buffer size did not match input buffer size";
  EXPECT_TRUE(outBuf->getBuffer() ==
              StringRef(onesBuf->getBufferStart(), onesBuf->getBufferSize()))
      << "Buffer returned did not match the buffer inserted via insertKeyed";
}

TEST_F(BlobCacheTest, InsertKeyedSyncWorks) {
  // Create a buffer with a known value
  auto twosDataBuf = WriteableBuffer::get();
  twosDataBuf->write(2);
  BufferRef twosBuf = std::move(twosDataBuf);

  // Use a plain key for retrieval
  std::string plainKey = "insertKeyedSyncTest";
  // Compute the hashed key which will be used for synchronous insertion
  std::string hashedKey = cache->getHash(plainKey);

  // Insert using insertKeyedSync; pass the pre-hashed key
  auto err = cache->insertKeyedSync(hashedKey, twosBuf.copy());
  ASSERT_FALSE(err.isError())
      << "Synchronous insertKeyedSync failed: " << err.getError();

  // Retrieve using findSync with the plain key
  auto findOr = cache->findSync(plainKey);
  ASSERT_FALSE(findOr.isError()) << findOr.getError();
  ASSERT_TRUE(findOr->has_value())
      << "Expected to retrieve item inserted via insertKeyedSync";
  BufferRef outBuf = std::move(**findOr);
  ASSERT_EQ(outBuf->getBufferSize(), twosBuf->getBufferSize())
      << "Output buffer size did not match input buffer size";
  EXPECT_TRUE(outBuf->getBuffer() ==
              StringRef(twosBuf->getBufferStart(), twosBuf->getBufferSize()))
      << "Buffer returned did not match the buffer inserted via "
         "insertKeyedSync";
}

TEST_F(BlobCacheTest, ContainsKeyedWorks) {
  // Create buffer with unique value
  auto threesDataBuf = WriteableBuffer::get();
  threesDataBuf->write(3);
  BufferRef threesBuf = std::move(threesDataBuf);

  std::string plainKey = "containsKeyedTest";
  std::string hashedKey = cache->getHash(plainKey);

  // Insert using insertKeyed with pre-hashed key
  auto chainAV = cache->insertKeyed(*cpuDevice, hashedKey, threesBuf.copy());
  await(chainAV);

  // Verify containsKeyed with direct hash check
  auto containsAV = cache->containsKeyed(*cpuDevice, hashedKey);
  await(containsAV);

  ASSERT_FALSE(containsAV.isError()) << containsAV.getDiagnostic().getMessage();
  EXPECT_TRUE(*containsAV)
      << "containsKeyed should find item with pre-hashed key";
}

TEST_F(BlobCacheTest, ContainsKeyedSyncWorks) {
  // Create buffer with unique value
  auto foursDataBuf = WriteableBuffer::get();
  foursDataBuf->write(4);
  BufferRef foursBuf = std::move(foursDataBuf);

  std::string plainKey = "containsKeyedSyncTest";
  std::string hashedKey = cache->getHash(plainKey);

  auto err = cache->insertKeyedSync(hashedKey, foursBuf.copy());
  ASSERT_FALSE(err.isError()) << "Synchronous insertKeyedSync failed";

  auto containsOr = cache->containsKeyedSync(hashedKey);
  ASSERT_FALSE(containsOr.isError()) << containsOr.getError();
  EXPECT_TRUE(*containsOr)
      << "containsKeyedSync should find item with pre-hashed key";
}

TEST_F(BlobCacheTest, FindKeyedWorks) {
  // Create buffer with unique value
  auto fivesDataBuf = WriteableBuffer::get();
  fivesDataBuf->write(5);
  BufferRef fivesBuf = std::move(fivesDataBuf);

  std::string plainKey = "findKeyedTest";
  std::string hashedKey = cache->getHash(plainKey);

  auto chainAV = cache->insertKeyed(*cpuDevice, hashedKey, fivesBuf.copy());
  await(chainAV);

  auto findAV = cache->findKeyed(*cpuDevice, hashedKey);
  await(findAV);

  ASSERT_FALSE(findAV.isError()) << findAV.getDiagnostic().getMessage();
  ASSERT_TRUE(findAV->has_value())
      << "findKeyed should retrieve item with pre-hashed key";

  BufferRef outBuf = std::move(**findAV);
  EXPECT_TRUE(outBuf->getBuffer() ==
              StringRef(fivesBuf->getBufferStart(), fivesBuf->getBufferSize()))
      << "findKeyed returned different data than inserted";
}

TEST_F(BlobCacheTest, FindKeyedSyncWorks) {
  // Create buffer with unique value
  auto sixesDataBuf = WriteableBuffer::get();
  sixesDataBuf->write(6);
  BufferRef sixesBuf = std::move(sixesDataBuf);

  std::string plainKey = "findKeyedSyncTest";
  std::string hashedKey = cache->getHash(plainKey);

  auto err = cache->insertKeyedSync(hashedKey, sixesBuf.copy());
  ASSERT_FALSE(err.isError()) << "Synchronous insertKeyedSync failed";

  auto findOr = cache->findKeyedSync(hashedKey);
  ASSERT_FALSE(findOr.isError()) << findOr.getError();
  ASSERT_TRUE(findOr->has_value())
      << "findKeyedSync should retrieve item with pre-hashed key";

  BufferRef outBuf = std::move(**findOr);
  EXPECT_TRUE(outBuf->getBuffer() ==
              StringRef(sixesBuf->getBufferStart(), sixesBuf->getBufferSize()))
      << "findKeyedSync returned different data than inserted";
}

//===----------------------------------------------------------------------===//
// Specialized FilesystemBackend tests
//===----------------------------------------------------------------------===//

/// Returns key for given thread and run.
static std::string makeKeyStr(int thread, int run) {
  return "key[" + std::to_string(thread) + "," + std::to_string(run) + "]";
}

/// Returns key in buffer form for thread and run.
static BufferRef makeKey(int thread, int run) {
  std::string key = makeKeyStr(thread, run);
  auto writeableKeyBuffer = WriteableBuffer::get();
  writeableKeyBuffer->write(key.data(), key.size());
  return std::move(writeableKeyBuffer);
}

/// Returns buffer with distinguished byte value for thread and run.
static BufferRef makeValue(size_t size, int numThreads, int thread, int run) {
  uint8_t value = (thread * numThreads) + run;
  auto writeableValueBuffer = WriteableBuffer::get(size);
  memset(writeableValueBuffer->getBufferStart(), value, size);
  return std::move(writeableValueBuffer);
}

static AsyncRT::EncodedLocation unknownLoc() {
  return AsyncRT::UnknownLocationDecoder::getEncodedLocation();
}

static AsyncRT::CPUDeviceRef makeCPUDevice() {
  return AsyncRT::getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
                                       AsyncRT::CPUDeviceOptions()
                                           .withLeakCheckedAllocator()
                                           .withMainWillNotDonate());
}

TEST(FilesystemBackend, Hammer) {
  const size_t size = 8000;
  const int numThreads = 20;
  const int numKeys = 200;
  TempDir tempDir = createTempDir();

  std::vector<std::thread> threads;
  for (int thread = 0; thread < numThreads; ++thread) {
    threads.emplace_back([thread, &tempDir]() {
      auto cpuDevice = makeCPUDevice();
      auto backend = getFilesystemBackend(tempDir.getPath());
      auto threadDone = AsyncValueRef<Chain>::allocate(*cpuDevice);

      // Insert known values with known keys.
      std::vector<AnyAsyncValueRef> insertsDone;
      for (int run = 0; run < numKeys; ++run) {
        insertsDone.emplace_back(
            backend->insert(*cpuDevice, makeKey(thread, run),
                            makeValue(size, numThreads, thread, run)));
      }
      andThenSyncMoving(
          insertsDone,
          [thread, cpuDevice = cpuDevice.getPointer(), backend = backend.copy(),
           threadDone = threadDone.copy()](
              MutableArrayRef<AnyAsyncValueRef> insertsDone) mutable {
            for (auto &ref : insertsDone) {
              if (ref.isError())
                return std::move(threadDone).setToError(ref.takeDiagnostic());
            }

            // Retrieve those values and check they match.
            std::vector<AnyAsyncValueRef> findsDone;
            for (int run = 0; run < numKeys; ++run) {
              auto findDone = AsyncValueRef<Chain>::allocate(*cpuDevice);
              backend->find(*cpuDevice, makeKey(thread, run))
                  .andThenSync([thread, run, findDone = findDone.copy()](
                                   AsyncValueRef<std::optional<BufferRef>>
                                       optResult) mutable {
                    if (optResult.isError())
                      return std::move(findDone).setToError(
                          optResult.takeDiagnostic());
                    if (!optResult->has_value())
                      return std::move(findDone).setToError(
                          {Twine("no entry for ") + makeKeyStr(thread, run),
                           unknownLoc()});
                    if (optResult->value()->getBufferSize() != size)
                      return std::move(findDone).setToError(
                          {Twine("mismatched size for ") +
                               makeKeyStr(thread, run) + ": actual size is " +
                               Twine(optResult->value()->getBufferSize()),
                           unknownLoc()});
                    BufferRef expectedValue =
                        makeValue(size, numThreads, thread, run);
                    if (memcmp(optResult->value()->getBufferStart(),
                               expectedValue->getBufferStart(), size))
                      return std::move(findDone).setToError(
                          {Twine(
                               "retrieved value does not match expected value "
                               "for ") +
                               makeKeyStr(thread, run),
                           unknownLoc()});
                    std::move(findDone).emplace();
                  });
              findsDone.emplace_back(std::move(findDone));
            }
            andThenSyncMoving(findsDone, [threadDone = std::move(threadDone)](
                                             MutableArrayRef<AnyAsyncValueRef>
                                                 findsDone) mutable {
              for (auto &ref : findsDone) {
                if (ref.isError())
                  return std::move(threadDone).setToError(ref.takeDiagnostic());
              }
              std::move(threadDone).emplace();
            });
          });

      await(threadDone);
      EXPECT_FALSE(threadDone.isError())
          << threadDone.getDiagnostic().getMessage().get() << '\n';
    });
  }

  for (auto &thread : threads)
    thread.join();
}
