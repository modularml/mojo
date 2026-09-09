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

/// This test is still under AsyncRT because we want to do fine-grained thread
/// control to run this test with semaphores, which rightly lives in AsyncRT.
#include "Support/TypeID.h"

#include "AsyncRT/Support/Semaphore.h"

#include "gtest/gtest.h"

#include <string_view>
#include <thread>
#include <vector>

using namespace M;
using namespace AsyncRT;

template <typename T>
struct SingleClassTemplate {};

template <typename T, typename U>
struct DoubleClassTemplate {
  T foo1;
  U foo2;
};

using FooBar = DoubleClassTemplate<SingleClassTemplate<int>, bool>;

struct Baz {};
struct Foo {};

namespace NS1::NS2 {
struct bar;
}

namespace {

TEST(TypeID, typeName) {
  using namespace M::Detail;
  using namespace std::string_view_literals;
#if (defined(__clang__))
  static_assert("void"sv == typeNameFor<void>());
  static_assert("int"sv == typeNameFor<int>());
  static_assert("fwd"sv == typeNameFor<class fwd>());
  static_assert("Foo"sv == typeNameFor<Foo>());

  static_assert("const int *"sv == typeNameFor<const int *>());
  static_assert("const int &"sv == typeNameFor<const int &>());
  static_assert("int **"sv == typeNameFor<int **>());
  static_assert("int &&"sv == typeNameFor<int &&>());

  static_assert("NS1::NS2::bar"sv == typeNameFor<NS1::NS2::bar>());

  static_assert("SingleClassTemplate<void>"sv ==
                typeNameFor<SingleClassTemplate<void>>());
  static_assert("SingleClassTemplate<int>"sv ==
                typeNameFor<SingleClassTemplate<int>>());

  // Show how `preferred_name` attribute can come into play
#if defined(_LIBCPP_VERSION)
  static_assert("std::string"sv == typeNameFor<std::string>());
#else
  static_assert("std::basic_string<char>"sv == typeNameFor<std::string>());
#endif

#elif (defined(__GNUC__))
  static_assert("void"sv == typeNameFor<void>());
  static_assert("int"sv == typeNameFor<int>());
  static_assert("Foo"sv == typeNameFor<Foo>());

  static_assert("const int*"sv == typeNameFor<const int *>());
  static_assert("const int&"sv == typeNameFor<const int &>());
  static_assert("int**"sv == typeNameFor<int **>());
  static_assert("int&&"sv == typeNameFor<int &&>());

  static_assert("NS1::NS2::bar"sv == typeNameFor<NS1::NS2::bar>());

  static_assert("SingleClassTemplate<void>"sv ==
                typeNameFor<SingleClassTemplate<void>>());
  static_assert("SingleClassTemplate<int>"sv ==
                typeNameFor<SingleClassTemplate<int>>());
  static_assert("std::__cxx11::basic_string<char>"sv ==
                typeNameFor<std::string>());
#elif (defined(_MSC_VER))
  static_assert("void"sv == typeNameFor<void>());
  static_assert("int"sv == typeNameFor<int>());
#endif
}

TEST(TypeID, Smoke) {
  constexpr size_t numThreads = 10;

  std::vector<TypeID> typeIDsA, typeIDsB;
  typeIDsA.resize(numThreads, TypeID());
  typeIDsB.resize(numThreads, TypeID());

  {
    // Concurrently get the types.
    Semaphore getReady;
    auto getThreadWorkFn = [&getReady, &typeIDsA, &typeIDsB](size_t i) {
      getReady.wait();
      typeIDsA[i] = TypeID::get<FooBar>();
      typeIDsB[i] = TypeID::get<Baz>();
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; ++i)
      threads.emplace_back(getThreadWorkFn, i);
    for (size_t i = 0; i < numThreads; ++i)
      // Try to trigger a thundering hurd.
      getReady.post();
    for (auto &thread : threads)
      thread.join();
  }

#if (defined(__clang__))
  EXPECT_EQ(typeIDsA.front().getTypeName(),
            "DoubleClassTemplate<SingleClassTemplate<int>, bool>");
  EXPECT_EQ(typeIDsB.front().getTypeName(), "Baz");
#endif

  for (size_t i = 1; i < numThreads; ++i) {
    EXPECT_EQ(typeIDsA[i], typeIDsA.front());
    EXPECT_EQ(typeIDsB[i], typeIDsB.front());
  }
}

} // namespace
