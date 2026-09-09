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

TensorSpec spec(const std::vector<std::int64_t> &vec, DType dtype) {
  return TensorSpec(TensorShape(vec), dtype);
}

template <typename T>
ErrorOr<T> ok(T v) {
  return std::move(v);
}

bool tensorSpecRoundTrips(const std::vector<std::int64_t> &vec, DType dtype) {
  TensorSpec spec(TensorShape(vec), dtype);
  std::vector<std::int64_t> roundTripped;
  for (auto dim : spec.getDimsCopy())
    roundTripped.push_back(dim);
  return vec == roundTripped && spec.getEltType() == dtype;
}

} // namespace

TEST(TensorSpec, constructor) {
  {
    SmallVector<int64_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorSpec(TensorShape(shape), DType::f32).getAsString(),
              "1x2x3x4xf32");
  }

  {
    SmallVector<size_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorSpec(TensorShape(shape), DType::ui8).getAsString(),
              "1x2x3x4xui8");
  }

  {
    SmallVector<ssize_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorSpec(TensorShape(shape), DType::si32).getAsString(),
              "1x2x3x4xsi32");
  }

  EXPECT_EQ(TensorSpec(TensorShape({1, 2, 3, 4}), DType::bf16).getAsString(),
            "1x2x3x4xbf16");
}

TEST(TensorSpec, representations) {
  EXPECT_TRUE(tensorSpecRoundTrips({}, DType::f16));
  EXPECT_TRUE(tensorSpecRoundTrips({1}, DType::bf16));
  EXPECT_TRUE(tensorSpecRoundTrips({2, 2, 2, 2, 2}, DType::si32));
  EXPECT_TRUE(tensorSpecRoundTrips({100000}, DType::si64));
  EXPECT_TRUE(tensorSpecRoundTrips({1, 2, 3, 4, 5}, DType::ui16));
  EXPECT_TRUE(tensorSpecRoundTrips({1, 2, 3, 4, 5, 6}, DType::f32));
  EXPECT_TRUE(tensorSpecRoundTrips({1, 1, 1, 1, kDynamic}, DType::ui8));
}

TEST(TensorSpec, stringizing) {
  EXPECT_EQ("f32", spec({}, DType::f32).getAsString());
  EXPECT_EQ("5xf16", spec({5}, DType::f16).getAsString());
  EXPECT_EQ("5x10xui8", spec({5, 10}, DType::ui8).getAsString());
  EXPECT_EQ("5x10x20xsi32", spec({5, 10, 20}, DType::si32).getAsString());
  EXPECT_EQ("?x10x20xsi64",
            spec({kDynamic, 10, 20}, DType::si64).getAsString());
  EXPECT_EQ("5x?x20xui32", spec({5, kDynamic, 20}, DType::ui32).getAsString());
  EXPECT_EQ("5x10x?xui64", spec({5, 10, kDynamic}, DType::ui64).getAsString());
  EXPECT_EQ("?xbf16", spec({kDynamic}, DType::bf16).getAsString());
  EXPECT_EQ("5xcomplex<f32>",
            spec({5}, DType::getComplex(DType::f32)).getAsString());
}

TEST(TensorSpec, parsing) {
  EXPECT_EQ(ok(spec({}, DType::f32)), TensorSpec::parseFromString("f32"));
  EXPECT_EQ(ok(spec({5}, DType::f16)), TensorSpec::parseFromString("5xf16"));
  EXPECT_EQ(ok(spec({5, 10}, DType::ui8)),
            TensorSpec::parseFromString("5x10xui8"));
  EXPECT_EQ(ok(spec({5, 10, 20}, DType::ui16)),
            TensorSpec::parseFromString("5x10x20xui16"));
  EXPECT_EQ(ok(spec({kDynamic, 10, 20}, DType::si32)),
            TensorSpec::parseFromString("?x10x20xsi32"));
  EXPECT_EQ(ok(spec({5, kDynamic, 20}, DType::f64)),
            TensorSpec::parseFromString("5x?x20xf64"));
  EXPECT_EQ(ok(spec({kDynamic}, DType::si64)),
            TensorSpec::parseFromString("?xsi64"));
  EXPECT_EQ(ok(spec({5}, DType::getComplex(DType::f32))),
            TensorSpec::parseFromString("5xcomplex<f32>"));
  EXPECT_EQ(ErrorOr<TensorSpec>(
                Error("could not parse shape from string: 2x3.5xf32: could not "
                      "parse dimension integer from string: 2x3.5 because 3.5 "
                      "cannot be parsed as an integer")),
            TensorSpec::parseFromString("2x3.5xf32"));
  EXPECT_EQ(ErrorOr<TensorSpec>(Error("could not parse dtype from string: 2x3 "
                                      "because 3 is not a valid DType")),
            TensorSpec::parseFromString("2x3"));
  EXPECT_EQ(
      ErrorOr<TensorSpec>(Error("could not parse dtype from string: 2x3xd32 "
                                "because d32 is not a valid DType")),
      TensorSpec::parseFromString("2x3xd32"));
}
