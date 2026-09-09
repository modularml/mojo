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

#include "Support/ADT/GenericUniquePtr.h"
#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"

using namespace M;

namespace {

struct TypeA {
  int i = 42;

  TypeA() = default;
  TypeA(int i) : i(i) {}
};

struct TypeB {
  int i = 50;
};

TEST(GenericUniquePtr, Lifecycle) {
  auto ptr1 = makeGenericUniquePtr<TypeA>(5);
  auto ptr2 = makeGenericUniquePtr<TypeA>();
  GenericUniquePtr ptr3;

  ASSERT_TRUE(ptr1);
  ASSERT_TRUE(ptr2);
  ASSERT_FALSE(ptr3);

  ASSERT_EQ(ptr1.get<TypeA>()->i, 5);
  ASSERT_EQ(ptr2.get<TypeA>()->i, 42);

  ptr3.reset(std::make_unique<TypeB>());
  ASSERT_TRUE(ptr3);
  ASSERT_EQ(ptr3.get<TypeB>()->i, 50);

  TypeB *released = ptr3.release<TypeB>();
  ASSERT_NE(released, nullptr);
  delete released;

  ptr1 = std::move(ptr2);
  ASSERT_FALSE(ptr2);
  ASSERT_TRUE(ptr1);
  ASSERT_EQ(ptr1.get<TypeA>()->i, 42);
}

#ifndef NDEBUG
TEST(GenericUniquePtr, IllTyped_ExpectDeath) {
  auto ptr = makeGenericUniquePtr<TypeA>(5);
  ASSERT_DEATH_IF_SUPPORTED(ptr.get<TypeB>(),
                            "mismatch between actual and expected type ids");
}
#endif

TEST(GenericUniquePtr, Casting) {
  auto ptr = makeGenericUniquePtr<TypeA>(32);
  ASSERT_TRUE(llvm::isa<TypeA>(ptr));
  ASSERT_TRUE(llvm::isa_and_present<TypeA>(ptr));
  ASSERT_EQ(llvm::dyn_cast<TypeA>(ptr)->i, 32);
}

} // namespace
