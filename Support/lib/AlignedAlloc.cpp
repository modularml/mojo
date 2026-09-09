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

#include "Support/AlignedAlloc.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstddef>
#include <cstdlib>
using namespace M;

/// This is a helper to handle host-specific system alignment functions.
void *M::alignedAlloc(size_t alignment, size_t size) {
  assert(llvm::isPowerOf2_64(alignment) && "non-power-of-2 alignment!");

#ifdef _WIN32
  // MSVC runtime doesn't support aligned_alloc(). See
  // https://developercommunity.visualstudio.com/t/c17-stdaligned-alloc%E7%BC%BA%E5%A4%B1/468021#T-N473365
  if (size == 0)
    return nullptr;
  size = llvm::alignToPowerOf2(size, alignment);
  return _aligned_malloc(size, alignment);
#else  // _WIN32
  if (alignment <= 8)
    return std::malloc(size);
  assert(alignment >= sizeof(void *) && "caller already checked");

  // Returns the next integer (mod 2**64) that is greater than or equal to
  // size and is a multiple of alignment. We know that the alignment is a
  // multiple of 2.
  size = llvm::alignToPowerOf2(size, alignment);

  return std::aligned_alloc(alignment, size);
#endif // _WIN32
}

#ifdef _WIN32
void M::alignedFree(void *ptr) {
  // _aligned_alloc() must be paired with _aligned_free().
  //
  // Attempting to use free() with a pointer returned by _aligned_malloc()
  // results in runtime issues that are hard to debug.
  _aligned_free(ptr);
}
#endif
