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

#include "./Memory.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "Support/AlignedAlloc.h"
#include "Support/Log.h"
#include "Support/SymbolExport.h"
using namespace M;

namespace {
struct {
  void *(*alloc)(size_t alignment,
                 size_t size) = AsyncRT::TCMallocGlobals::tc_new;
  void (*free)(void *ptr) = AsyncRT::TCMallocGlobals::tc_delete;
} constinit static KGEN_Allocators{};
} // namespace

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_SetAsanAllocators() {
  KGEN_Allocators = {.alloc = M::alignedAlloc, .free = M::alignedFree};
}

/// Returns an alignment allocated memory. If the alignment value is not
/// positive, then the default alignment is used.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size) {
  if (alignment <= 0)
    alignment = kPreferredMemoryAlignment;
  void *ptr = KGEN_Allocators.alloc(alignment, size);
#if MODULAR_ALLOC_LOGGING
  MLOG_DEBUG("mojo alloc: ptr={} size={} alignment={}", ptr, size, alignment);
#endif
  return ptr;
}

/// Frees memory allocated via KGEN_CompilerRT_AlignedAlloc.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AlignedFree(void *ptr) {
#if MODULAR_ALLOC_LOGGING
  MLOG_DEBUG("mojo free: ptr={}", ptr);
#endif
  return KGEN_Allocators.free(ptr);
}
