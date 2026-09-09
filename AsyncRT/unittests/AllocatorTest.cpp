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

#include "Support/AlignedAlloc.h"
#include "Support/Threading/HWInfo.h"

#include "gtest/gtest.h"

#if HAVE_LINUX_X86_SYSTEM_INFO
#include <cstring>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#endif

using namespace M::AsyncRT;

namespace {

TEST(AllocatorTest, DefaultFactories_ReportNoNumaAffinity) {
  EXPECT_EQ(createMallocAllocator()->getNumaPlacement(), M::kAnyNumaNode);
  EXPECT_EQ(createTCMallocAllocator()->getNumaPlacement(), M::kAnyNumaNode);
}

TEST(AllocatorTest, TCMallocNumaFactory_StoresNumaPlacement) {
  auto allocator = createTCMallocAllocator(/*numaPlacement=*/3);
  EXPECT_EQ(allocator->getNumaPlacement(), 3);

  int *ptr = allocator->allocate<int>();
  *ptr = 7;
  EXPECT_EQ(*ptr, 7);
  allocator->deallocate(ptr);
}

TEST(AllocatorTest, TCMallocNumaFactory_AcceptsAnyNumaNodeSentinel) {
  auto allocator = createTCMallocAllocator(M::kAnyNumaNode);
  EXPECT_EQ(allocator->getNumaPlacement(), M::kAnyNumaNode);
}

TEST(AllocatorTest, Use_TCMalloc) {
  auto allocator = createTCMallocAllocator();
  int *ptr1 = allocator->allocate<int>();
  int *ptr2 = allocator->allocate<int>();

  // Expect that the pointers are aligned according to the default/preferred
  // alignment.
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) &
                (M::kPreferredMemoryAlignment - 1),
            0UL)
      << ptr1;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) &
                (M::kPreferredMemoryAlignment - 1),
            0UL)
      << ptr2;

  *ptr1 = 42;
  *ptr2 = 43;
  EXPECT_EQ(*ptr1, 42);
  allocator->deallocate(ptr1);
  allocator->deallocate(ptr2);

  // Use an alignment which is larger than the default.
  constexpr size_t largeAlignment = 512;
  ptr1 = reinterpret_cast<int *>(
      allocator->allocateBytes(sizeof(int), largeAlignment));
  ptr2 = reinterpret_cast<int *>(
      allocator->allocateBytes(sizeof(int), largeAlignment));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) & (largeAlignment - 1), 0UL)
      << ptr1;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) & (largeAlignment - 1), 0UL)
      << ptr2;
  allocator->deallocateBytes(ptr1, sizeof(int));
  allocator->deallocateBytes(ptr2, sizeof(int));
}

#if HAVE_LINUX_X86_SYSTEM_INFO

// Returns the physical NUMA node for the page containing ptr via move_pages(2)
// with nodes=nullptr (query only, no migration). ptr must be page-aligned and
// the page must already be faulted in. Returns a negative errno on failure.
static int queryPageNumaNode(void *ptr) {
  void *pages[1] = {ptr};
  int status[1] = {-1};
  ::syscall(SYS_move_pages, 0, 1, pages, nullptr, status, 0);
  return status[0];
}

// Verify that allocations from a NUMA-node-bound TCMallocAllocator land on a
// physical page in the correct TCMalloc partition. TCMalloc maps node N to
// partition N % 2 and mbinds virtual address regions to NUMA nodes within that
// partition.
TEST(AllocatorTest, TCMalloc_NumaAllocation_LandsOnCorrectNode) {
  const auto &topoOrErr = M::NUMATopology::get();
  if (topoOrErr.isError() || topoOrErr->numaNodes.size() < 2)
    GTEST_SKIP() << "System does not have multiple NUMA nodes";

  // Allocate two pages so at least one full page is resident after the write.
  constexpr size_t kPageSize = 4096;
  constexpr size_t kAllocSize = 2 * kPageSize;

  for (int node : topoOrErr->numaNodes) {
    SCOPED_TRACE("NUMA node " + std::to_string(node));

    auto allocator = createTCMallocAllocator(node);
    void *ptr = allocator->allocateBytes(kAllocSize, kPageSize);
    ASSERT_NE(ptr, nullptr);

    // Fault in all pages so move_pages can report a valid physical node.
    std::memset(ptr, 0, kAllocSize);

    const int physicalNode = queryPageNumaNode(ptr);
    allocator->deallocateBytes(ptr, kAllocSize);

    if (physicalNode < 0)
      continue; // Page not resident or syscall unsupported; skip this node.

    // The physical page must be in the same TCMalloc partition as the requested
    // node. On machines with > 2 NUMA nodes, multiple nodes share a partition.
    EXPECT_EQ(static_cast<size_t>(physicalNode) % 2,
              static_cast<size_t>(node) % 2)
        << "Allocation for NUMA node " << node << " landed on physical node "
        << physicalNode << " (wrong TCMalloc partition).";
  }
}

// Verify that memory allocated with a NUMA-bound allocator can be safely freed
// from a different thread. TCMalloc derives the partition from the pointer's
// virtual address tag, so deallocation is partition-correct regardless of
// which thread or NUMA node the caller is on.
TEST(AllocatorTest, TCMalloc_NumaAllocation_FreeFromAnyThread) {
  const auto &topoOrErr = M::NUMATopology::get();
  if (topoOrErr.isError() || topoOrErr->numaNodes.size() < 2)
    GTEST_SKIP() << "System does not have multiple NUMA nodes";

  auto allocator = createTCMallocAllocator(/*numaPlacement=*/0);
  constexpr size_t kPageSize = 4096;
  void *ptr = allocator->allocateBytes(kPageSize, kPageSize);
  ASSERT_NE(ptr, nullptr);
  *static_cast<volatile char *>(ptr) = 1; // fault in

  std::thread([&] { allocator->deallocateBytes(ptr, kPageSize); }).join();
}

#endif // HAVE_LINUX_X86_SYSTEM_INFO

} // namespace
