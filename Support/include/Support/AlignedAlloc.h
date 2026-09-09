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
// This file declares alignedAlloc() and alignedFree() for allocating dynamic
// buffers with explicit alignment.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ALIGNED_ALLOC_H
#define SUPPORT_ALIGNED_ALLOC_H

#include "Support/PlatformUtils.h"

#include <cstddef>
#include <cstdlib>
#include <memory>

namespace M {

#if MODULAR_X86_64
static constexpr size_t kPreferredMemoryAlignment = 64;
#elif MODULAR_ARM_NEON
static constexpr size_t kPreferredMemoryAlignment = 16;
#else
static constexpr size_t kPreferredMemoryAlignment = 16;
#endif

/// Allocate the a block of memory with the specified size and alignment.
///  NOTE: The returned pointer *must* be deallocated with alignedFree().
/// Deallocating with e.g. free() instead causes runtime issues on Windows that
/// are hard to debug.
void *alignedAlloc(size_t alignment, size_t size);

#ifndef _WIN32
/// alignedFree deallocates a pointer allocated with alignedAlloc.
inline void alignedFree(void *ptr) { std::free(ptr); }
#else
/// alignedFree deallocates a pointer allocated with alignedAlloc.
void alignedFree(void *ptr);
#endif

/// Helper alias template that fixes the deleter type to be `alignedFree`.
template <class T>
using unique_ptr_aligned = std::unique_ptr<T, decltype(&alignedFree)>;

/// Helper function template to simplify declarations of aligned unique
/// pointers.  The alignment and size are passed through to `alignedAlloc` and
/// `alignedFree` is always used as the deleter.  The aligned unique pointer
/// created is based on the template type `T`.
template <class T>
unique_ptr_aligned<T> makeAlignedUniquePtr(size_t alignment, size_t size) {
  return unique_ptr_aligned<T>(static_cast<T *>(alignedAlloc(alignment, size)),
                               &alignedFree);
}

} // namespace M

#endif // SUPPORT_ALIGNED_ALLOC_H
