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

#include "Support/Context.h"

#include "gtest/gtest.h"

using namespace M;

namespace {

struct ContextA {
  int i = 42;

  ContextA() = default;
  ContextA(int i) : i(i) {}
};

struct ContextB {
  bool b = true;
  char lots[26]; // Give this struct a large but unaligned size

  ContextB() = default;
};

struct ContextC {
  char c = 'a';
};

TEST(ContextTest, Basic) {
  auto ctx = ContextRef::create();

  ContextA &contextARef = ctx->emplace<ContextA>(5);
  ctx->emplace<ContextB>();

  ++contextARef.i;

  ContextA *contextAPtr = ctx->get<ContextA>();
  ContextB *contextBPtr = ctx->get<ContextB>();
  ContextC *contextCPtr = ctx->get<ContextC>();

  ASSERT_NE(contextAPtr, nullptr);
  EXPECT_EQ(contextAPtr->i, 6);
  ASSERT_NE(contextBPtr, nullptr);
  EXPECT_EQ(contextBPtr->b, true);
  EXPECT_EQ(contextCPtr, nullptr);

  bool created = false;
  ErrorOr<ContextC *> contextCOr = ctx->createIfMissing<ContextC>(
      [&created]() -> ErrorOr<std::unique_ptr<ContextC>> {
        created = true;
        return std::make_unique<ContextC>();
      });
  ASSERT_TRUE(created);
  ASSERT_FALSE(contextCOr.isError());
  ASSERT_EQ((*contextCOr)->c, 'a');

  created = false;
  ErrorOr<ContextC *> contextCAgainOr = ctx->createIfMissing<ContextC>(
      [&created]() -> ErrorOr<std::unique_ptr<ContextC>> {
        created = true;
        return std::make_unique<ContextC>();
      });
  ASSERT_FALSE(created);
  ASSERT_FALSE(contextCAgainOr.isError());
  ASSERT_EQ(*contextCAgainOr, *contextCOr);
}

TEST(ContextTest, DoubleEmplace) {
  auto ctx = ContextRef::create();
  ctx->emplace<ContextA>();
  ASSERT_DEATH_IF_SUPPORTED(ctx->emplace<ContextA>(),
                            "set already holds object of type");
}

} // namespace
