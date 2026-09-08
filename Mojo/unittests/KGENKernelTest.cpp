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

#include "test_kernels.h"
#include "gtest/gtest.h"

TEST(KGENKernelTest, testArrayArgument) {
  int32_t values[4] = {11, 22, 33, 44};
  int32_t result = array_index(values);
  EXPECT_EQ(result, values[2]);
}
