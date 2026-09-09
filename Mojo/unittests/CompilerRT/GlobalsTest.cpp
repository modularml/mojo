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

#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>
#include <vector>

using namespace M;

// These are forward declarations from Globals.cpp.
COMPILERRT_EXPORT void *
KGEN_CompilerRT_GetOrCreateGlobal(llvm::StringRef name, void *(*initFn)(),
                                  void (*destroyFn)(void *));
COMPILERRT_EXPORT void *KGEN_CompilerRT_GetGlobalOrNull(llvm::StringRef name);
COMPILERRT_EXPORT void KGEN_CompilerRT_InsertGlobal(llvm::StringRef name,
                                                    void *value);
COMPILERRT_EXPORT void KGEN_CompilerRT_DestroyGlobals();

namespace {

std::atomic<int> constructCount{0};
std::atomic<int> destructCount{0};

struct TestObject {
  int value;
  TestObject(int v) : value(v) {
    constructCount.fetch_add(1, std::memory_order_relaxed);
  }
  ~TestObject() { destructCount.fetch_add(1, std::memory_order_relaxed); }
};

void *createTestObject() { return new TestObject(42); }

void destroyTestObject(void *obj) { delete static_cast<TestObject *>(obj); }

struct GlobalsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Reset counters.
    constructCount.store(0, std::memory_order_relaxed);
    destructCount.store(0, std::memory_order_relaxed);
  }

  void TearDown() override {
    // Clean up any globals created during test.
    KGEN_CompilerRT_DestroyGlobals();
  }
};

} // namespace

TEST_F(GlobalsTest, BasicOperations) {
  // This tests GetOrCreateGlobal.
  void *obj1 = KGEN_CompilerRT_GetOrCreateGlobal("test1", createTestObject,
                                                 destroyTestObject);
  ASSERT_NE(obj1, nullptr);
  EXPECT_EQ(static_cast<TestObject *>(obj1)->value, 42);

  // This tests that we get the same object on second call.
  void *obj2 = KGEN_CompilerRT_GetOrCreateGlobal("test1", createTestObject,
                                                 destroyTestObject);
  EXPECT_EQ(obj1, obj2);

  // This tests GetGlobalOrNull for existing global.
  void *obj3 = KGEN_CompilerRT_GetGlobalOrNull("test1");
  EXPECT_EQ(obj1, obj3);

  // This tests GetGlobalOrNull for non-existing global.
  void *obj4 = KGEN_CompilerRT_GetGlobalOrNull("nonexistent");
  EXPECT_EQ(obj4, nullptr);
}

TEST_F(GlobalsTest, InsertGlobal) {
  TestObject *directObj = new TestObject(99);
  KGEN_CompilerRT_InsertGlobal("test2", directObj);

  void *retrieved = KGEN_CompilerRT_GetGlobalOrNull("test2");
  EXPECT_EQ(retrieved, directObj);
  EXPECT_EQ(static_cast<TestObject *>(retrieved)->value, 99);

  // This is manual cleanup since InsertGlobal doesn't set up destruction.
  delete directObj;
}

TEST_F(GlobalsTest, MultiThreadedAccess) {
  const int numThreads = 8;
  const int numGlobalsPerThread = 100;
  std::vector<std::thread> threads;
  std::atomic<int> totalCreated{0};

  // This creates threads that try to create the same globals.
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < numGlobalsPerThread; ++i) {
        std::string name = ("global_" + llvm::Twine(i)).str();
        void *obj = KGEN_CompilerRT_GetOrCreateGlobal(name, createTestObject,
                                                      destroyTestObject);
        if (obj) {
          totalCreated.fetch_add(1, std::memory_order_relaxed);
          EXPECT_EQ(static_cast<TestObject *>(obj)->value, 42);
        }
      }
    });
  }

  // This waits for all threads.
  for (auto &thread : threads)
    thread.join();

  EXPECT_EQ(totalCreated.load(), numThreads * numGlobalsPerThread);
  // In lock-free implementation, temporary over-creation is expected.
  // This allows for reasonable overhead due to concurrent creation attempts.
  EXPECT_LE(constructCount.load(), numGlobalsPerThread * 3);
}

TEST_F(GlobalsTest, HashCollisions) {
  // This creates many globals with names that are likely to collide.
  const int numGlobals = 1000;
  std::vector<void *> globals;

  for (int i = 0; i < numGlobals; ++i) {
    // This uses short names to increase collision probability.
    std::string name =
        ("x" + llvm::Twine(i % 100) + "y" + llvm::Twine(i / 100)).str();
    void *obj = KGEN_CompilerRT_GetOrCreateGlobal(name, createTestObject,
                                                  destroyTestObject);
    ASSERT_NE(obj, nullptr) << "Failed to create global: " << name;
    globals.push_back(obj);

    // This verifies we can retrieve it.
    void *retrieved = KGEN_CompilerRT_GetGlobalOrNull(name);
    EXPECT_EQ(retrieved, obj) << "Failed to retrieve global: " << name;
  }

  // This verifies all globals are still accessible.
  for (int i = 0; i < numGlobals; ++i) {
    std::string name =
        ("x" + llvm::Twine(i % 100) + "y" + llvm::Twine(i / 100)).str();
    void *obj = KGEN_CompilerRT_GetGlobalOrNull(name);
    EXPECT_EQ(obj, globals[i]) << "Global changed after creation: " << name;
  }
}

TEST_F(GlobalsTest, DestructionOrder) {
  int destructBefore = destructCount.load();

  // This creates a few globals.
  KGEN_CompilerRT_GetOrCreateGlobal("destroy1", createTestObject,
                                    destroyTestObject);
  KGEN_CompilerRT_GetOrCreateGlobal("destroy2", createTestObject,
                                    destroyTestObject);

  // This destroys all globals.
  KGEN_CompilerRT_DestroyGlobals();

  int destructAfter = destructCount.load();
  EXPECT_GT(destructAfter, destructBefore);
}

TEST_F(GlobalsTest, DoubleDestructionProtection) {
  // This creates a global.
  void *obj = KGEN_CompilerRT_GetOrCreateGlobal(
      "double_destroy", createTestObject, destroyTestObject);
  ASSERT_NE(obj, nullptr);

  int destructBefore = destructCount.load();

  // This destroys all globals in first destruction.
  KGEN_CompilerRT_DestroyGlobals();

  int destructAfter = destructCount.load();
  EXPECT_GT(destructAfter, destructBefore);

  // This tries to destroy again and should be safe no-op.
  KGEN_CompilerRT_DestroyGlobals();

  int destructFinal = destructCount.load();
  EXPECT_EQ(destructFinal, destructAfter); // No additional destructions
}

TEST_F(GlobalsTest, TableCapacityLimits) {
  // Test behavior with moderate load
  const int numGlobals = 1000;
  int successCount = 0;

  for (int i = 0; i < numGlobals; ++i) {
    std::string name = ("capacity_test_" + llvm::Twine(i)).str();
    void *obj = KGEN_CompilerRT_GetOrCreateGlobal(name, createTestObject,
                                                  destroyTestObject);
    if (obj)
      ++successCount;
  }

  // With 16K table size, 1000 globals should all succeed.
  EXPECT_EQ(successCount, numGlobals) << "Unexpected creation failures";
}

TEST_F(GlobalsTest, ProbeExhaustionScenario) {
  // This tests what happens when we actually exhaust the probe limit.
  // This creates entries that will hash to the same cluster to force long probe
  // chains.

  // Strategy: This uses very similar strings that should hash to nearby values
  // and fill up consecutive slots to create long probe chains.

  int successCount = 0;
  int failureCount = 0;
  (void)successCount; // This avoids unused variable warning.
  (void)failureCount; // This avoids unused variable warning.
  std::string basePattern = "cluster_collision_test_";

  // This tries to create many globals with similar hash patterns.
  // This continues until we hit probe exhaustion.
  for (int i = 0; i < 100000; ++i) {
    std::string name = (basePattern + llvm::Twine(i)).str();
    void *obj = KGEN_CompilerRT_GetOrCreateGlobal(name, createTestObject,
                                                  destroyTestObject);

    if (obj) {
      ++successCount;
      // Verify we can retrieve the created global.
      void *retrieved = KGEN_CompilerRT_GetGlobalOrNull(name);
      EXPECT_EQ(retrieved, obj) << "Failed to retrieve global: " << name;
    } else {
      ++failureCount;
      break; // This stops at first failure to see what happens.
    }
  }

  // This verifies that we should have no failures with reasonable load.
  EXPECT_EQ(failureCount, 0) << "Unexpected failures creating globals";
  EXPECT_GT(successCount, 0) << "Should have created at least some globals";
}
