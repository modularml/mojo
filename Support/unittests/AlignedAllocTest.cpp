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

#include "gtest/gtest.h"

TEST(AlignedAlloc, uniquePtrAlignedShouldUseAlignedFreeDeleter) {
  auto expectedDeleter = &M::alignedFree;
  auto size = 32;
  auto actual =
      M::makeAlignedUniquePtr<int>(M::kPreferredMemoryAlignment, size);
  EXPECT_EQ(expectedDeleter, actual.get_deleter());
}
