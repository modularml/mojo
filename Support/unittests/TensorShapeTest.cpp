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

#include "Support/ML/TensorSpec.h"

#include "Support/ErrorOr.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/YAMLTraits.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;

namespace {

static_assert(std::is_same_v<std::decay_t<decltype(mlir::ShapedType::kDynamic)>,
                             std::int64_t>,
              "This file assumes kDynamic is an int64_t");
constexpr std::int64_t kDynamic = mlir::ShapedType::kDynamic;

TensorShape shape(const std::vector<std::int64_t> &vec) {
  return TensorShape(vec);
}

template <typename T>
ErrorOr<T> ok(T v) {
  return std::move(v);
}

bool tensorShapeRoundTrips(const std::vector<std::int64_t> &vec) {
  TensorShape shape(vec);
  std::vector<std::int64_t> roundTripped;
  for (auto dim : shape)
    roundTripped.push_back(dim);
  return vec == roundTripped;
}

struct ShapeHolder {
  TensorShape shape;
};

} // namespace

template <>
struct llvm::yaml::MappingTraits<ShapeHolder> {
  static void mapping(llvm::yaml::IO &io, ShapeHolder &holder) {
    io.mapRequired("shape", holder.shape);
  }
};

TEST(TensorShape, constructor) {
  {
    SmallVector<int64_t> shape{};
    EXPECT_EQ(TensorShape(shape).getAsString(), "");
  }

  {
    SmallVector<int64_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  {
    SmallVector<uint64_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  {
    SmallVector<int32_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  {
    SmallVector<size_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  {
    SmallVector<ssize_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  {
    SmallVector<ssize_t> shape{1, kDynamic, -1, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x?x?x4");
  }

  {
    SmallVector<size_t> shape{1, static_cast<size_t>(kDynamic), 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x?x3x4");
  }

  {
    SmallVector<uint64_t> shape{1, static_cast<uint64_t>(kDynamic), 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x?x3x4");
  }

  EXPECT_EQ(TensorShape(kDynamicallyRanked).getAsString(), "*");

  EXPECT_EQ(TensorShape({1, 2, 3, 4}).getAsString(), "1x2x3x4");

  {
    TensorShape shape = {1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape.getDimsCopy()), shape);
  }
}

TEST(TensorShape, hasRank) {
  ASSERT_TRUE(TensorShape({1, 2}).hasRank());
  ASSERT_FALSE(TensorShape(kDynamicallyRanked).hasRank());
}

TEST(TensorShape, isStatic) {
  {
    TensorShape shape({1, 2, 3, 4});
    ASSERT_EQ(shape.getKind(), 1 /* k32 */);
    EXPECT_TRUE(shape.isStatic());
  }
  {
    TensorShape shape({1, 2, -1, 4});
    ASSERT_EQ(shape.getKind(), 1 /* k32 */);
    EXPECT_FALSE(shape.isStatic());
  }
  {
    TensorShape shape({1, 2, 3, 4, 5, 6});
    ASSERT_EQ(shape.getKind(), 0 /* k16 */);
    EXPECT_TRUE(shape.isStatic());
  }
  {
    TensorShape shape({1, 2, -1, 4, 5, 6});
    ASSERT_EQ(shape.getKind(), 0 /* k16 */);
    EXPECT_FALSE(shape.isStatic());
  }
  {
    TensorShape shape({1, 2, 3, 4, 5, 6, 7, 8});
    ASSERT_EQ(shape.getKind(), 2 /* kOutOfLine */);
    EXPECT_TRUE(shape.isStatic());
  }
  {
    TensorShape shape({1, 2, -1, 4, 5, 6, 7, 8});
    ASSERT_EQ(shape.getKind(), 2 /* kOutOfLine */);
    EXPECT_FALSE(shape.isStatic());
  }
  EXPECT_FALSE(TensorShape(kDynamicallyRanked).isStatic());
}

TEST(TensorShape, representations) {
  EXPECT_TRUE(tensorShapeRoundTrips({}));
  EXPECT_TRUE(tensorShapeRoundTrips({1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, kDynamic}));

  EXPECT_EQ(TensorShape(kDynamicallyRanked), TensorShape(kDynamicallyRanked));
}

TEST(TensorShape, stringizing) {
  EXPECT_EQ(shape({}).getAsString(), "");
  EXPECT_EQ(shape({5}).getAsString(), "5");
  EXPECT_EQ(shape({5, 10}).getAsString(), "5x10");
  EXPECT_EQ(shape({5, 10, 20}).getAsString(), "5x10x20");
  EXPECT_EQ(shape({kDynamic, 10, 20}).getAsString(), "?x10x20");
  EXPECT_EQ(shape({5, kDynamic, 20}).getAsString(), "5x?x20");
  EXPECT_EQ(shape({5, 10, kDynamic}).getAsString(), "5x10x?");
  EXPECT_EQ(shape({kDynamic}).getAsString(), "?");
  EXPECT_EQ(TensorShape(kDynamicallyRanked).getAsString(), "*");
}

TEST(TensorShape, parsing) {
  EXPECT_EQ(TensorShape::parseFromString(""), ok(shape({})));
  EXPECT_EQ(TensorShape::parseFromString("5"), ok(shape({5})));
  EXPECT_EQ(TensorShape::parseFromString("5x10"), ok(shape({5, 10})));
  EXPECT_EQ(TensorShape::parseFromString("5x10x20"), ok(shape({5, 10, 20})));
  EXPECT_EQ(TensorShape::parseFromString("?x10x20"),
            ok(shape({kDynamic, 10, 20})));
  EXPECT_EQ(TensorShape::parseFromString("5x?x20"),
            ok(shape({5, kDynamic, 20})));
  EXPECT_EQ(TensorShape::parseFromString("5x10x?"),
            ok(shape({5, 10, kDynamic})));
  EXPECT_EQ(TensorShape::parseFromString("?"), ok(shape({kDynamic})));
  EXPECT_EQ(TensorShape::parseFromString("*"),
            ok(TensorShape(kDynamicallyRanked)));
  EXPECT_EQ(TensorShape::parseFromString("2x3.5"),
            ErrorOr<TensorShape>(
                Error("could not parse dimension integer from string: 2x3.5 "
                      "because 3.5 cannot be parsed as an integer")));
  EXPECT_EQ(TensorShape::parseFromString("2x-1"),
            ErrorOr<TensorShape>(Error("could not parse dimension integer from "
                                       "string: 2x-1 because -1 is negative")));
  EXPECT_EQ(TensorShape::parseFromString("1x2x3x4x5x6x7x8x9"),
            ErrorOr<TensorShape>(Error(
                "could not parse tensor shape from string: 1x2x3x4x5x6x7x8x9 "
                "because it is larger that the maximum supported rank 8")));
}

TEST(TensorShape, yamlInput) {
  llvm::yaml::Input input("shape: 5x?");
  ShapeHolder holder;
  input >> holder;
  EXPECT_EQ(shape({5, kDynamic}), holder.shape);
}

TEST(TensorShape, yamlOutput) {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::yaml::Output output(os);
  ShapeHolder holder{shape({5, kDynamic})};
  output << holder;
  EXPECT_EQ(str, "---\nshape:           5x?\n...\n");
}

// Regression test for #29927.
TEST(TensorShape, moveAssignmentDoesntLeak) {
  TensorShape a{1, 2, 3, 4, 5, 6, 7};
  a = TensorShape{1, 2, 3};
  // LSAN should now fail if this caused a leak.
}

// Regression test for #29927.
TEST(TensorShape, copyAssignmentDoesntLeak) {
  TensorShape a{1, 2, 3, 4, 5, 6, 7};
  TensorShape b{1, 2, 3};
  a = b;
  // LSAN should now fail if this caused a leak.
}

TEST(TensorShape, copyAssignSelf) {
  TensorShape a{1, 2, 3, 4, 5, 6, 7};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
  a = a;
#pragma GCC diagnostic pop
  EXPECT_EQ(a, TensorShape({1, 2, 3, 4, 5, 6, 7}));
}
