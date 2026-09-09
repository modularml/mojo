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

using namespace M;
using namespace M::AsyncRT;

namespace {
/// This is an implementation of the Allocator interface that just calls to
/// alignedAlloc/alignedFree, the system allocator implementations.
class MallocAllocator : public Allocator {
  // Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) override {
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.alloc", (uint64_t)size));

    return alignedAlloc(alignment, size);
  }

  // Deallocate the specified pointer that has the specified size.
  void deallocateBytes(void *ptr, size_t size) override {
    TimeTraceScope scope(
        MemAllocFreeProfilerEntry::create("mem.free", (uint64_t)size));
    alignedFree(ptr);
  }
};
} // namespace

std::unique_ptr<Allocator> M::AsyncRT::createMallocAllocator() {
  return std::make_unique<MallocAllocator>();
}

void M::AsyncRT::profiledMemcpy(void *dst, const void *src, size_t size) {
  // Since this profiling entry is on by default we don't include the size to
  // avoid string manipulation.
  TimeTraceScope scope(
      MemCopyProfilerEntry::create("mem.copy", (uint64_t)size));
  std::memcpy(dst, src, size);
}
