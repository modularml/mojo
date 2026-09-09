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

#include "AsyncRT/Runtime/Allocator.h"

#include "gtest/gtest.h"

using namespace M::AsyncRT;

namespace {

#if HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
TEST(UseAfterFreeAllocator, Detects) {
  auto allocator = createUseAfterFreeAllocator();
  int *ptr1 = allocator->allocate<int>();
  int *ptr2 = allocator->allocate<int>();
  *ptr1 = 42;
  *ptr2 = 43;
  allocator->deallocate(ptr1);
  EXPECT_DEATH(*ptr1 = 44, ".*");
  allocator->deallocate(ptr2);
}

#endif

TEST(LeakCheckAllocator, ForwardsNumaPlacementFromBase) {
  // TCMalloc base constructed with a NUMA placement — wrapper should report
  // the same placement.
  auto wrapped =
      createLeakCheckAllocator(createTCMallocAllocator(/*numaPlacement=*/2));
  EXPECT_EQ(wrapped->getNumaPlacement(), 2);

  // Exercise an alloc/free so the leak-check bookkeeping stays balanced.
  int *ptr = wrapped->allocate<int>();
  *ptr = 1;
  wrapped->deallocate(ptr);
}

TEST(LeakCheckAllocator, ReportsAnyNumaNodeForPlainBase) {
  auto wrapped = createLeakCheckAllocator(createMallocAllocator());
  EXPECT_EQ(wrapped->getNumaPlacement(), M::kAnyNumaNode);
}

TEST(ProfilingAllocator, ForwardsNumaPlacementFromBase) {
  auto wrapped =
      createProfilingAllocator(createTCMallocAllocator(/*numaPlacement=*/5));
  EXPECT_EQ(wrapped->getNumaPlacement(), 5);

  int *ptr = wrapped->allocate<int>();
  *ptr = 1;
  wrapped->deallocate(ptr);
}

} // namespace
