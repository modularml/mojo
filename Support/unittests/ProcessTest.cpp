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

#include "Support/Process.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using ::testing::HasSubstr;

TEST(ProcessTest, GetProcessExecutablePathWorks) {
  // The full path will vary based on the test sandbox, so only try to match
  // part of the path that we know will stay consistent.
  EXPECT_THAT(getProcessExecutablePath(),
              HasSubstr("Support/unittests/BaseTest"));
}
