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

#include "Support/Compiler/DomainAwareReplacer.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APSInt.h"
#include "gtest/gtest.h"

using namespace M;
using namespace mlir;

TEST(DomainAwareReplacerTest, testNoRecursion) {
  // ===== Setup =====
  // A contrived replacer that convert integer attrs into either f32 or string
  // attrs.
  MLIRContext context;
  DomainAwareReplacer replacer;
  enum Domain : DomainAwareReplacer::DomainId { ToFloat32, ToString };
  Type float32Type = Float32Type::get(&context);
  replacer.addReplacement(
      [&](IntegerAttr attr) -> Attribute {
        return FloatAttr::get(float32Type, attr.getInt());
      },
      Domain::ToFloat32);
  replacer.addReplacement(
      [&](IntegerAttr attr) -> Attribute {
        return StringAttr::get(&context, Twine(attr.getInt()));
      },
      Domain::ToString);

  // ===== Tests =====
  auto int42 = IntegerAttr::get(IntegerType::get(&context, 32), 42);
  EXPECT_EQ(*replacer.replace(int42, Domain::ToFloat32),
            FloatAttr::get(float32Type, 42));
  EXPECT_EQ(*replacer.replace(int42, Domain::ToString),
            StringAttr::get(&context, "42"));

  SmallVector<int32_t> inputI32Array = {42, 31, 79};
  SmallVector<Attribute> inputAttrArray(
      llvm::map_range(inputI32Array, [&](int32_t num) {
        return IntegerAttr::get(IntegerType::get(&context, 32), num);
      }));
  Attribute inputAttr = ArrayAttr::get(&context, inputAttrArray);

  SmallVector<Attribute> f32AttrArray(
      llvm::map_range(inputI32Array, [&](int32_t num) {
        return FloatAttr::get(float32Type, num);
      }));
  EXPECT_EQ(*replacer.replace(inputAttr, Domain::ToFloat32),
            ArrayAttr::get(&context, f32AttrArray));

  SmallVector<Attribute> stringAttrArray(
      llvm::map_range(inputI32Array, [&](int32_t num) {
        return StringAttr::get(&context, Twine(num));
      }));
  EXPECT_EQ(*replacer.replace(inputAttr, Domain::ToString),
            ArrayAttr::get(&context, stringAttrArray));
}

TEST(DomainAwareReplacerTest, testMutualRecursion) {
  // ===== Setup =====
  // A contrived replacer that increments int attrs in one domain, and
  // decrements in the other domain, and switches between domains based on
  // even/odd numbers. Replacement results are chained in an ArrayAttr.
  MLIRContext context;
  DomainAwareReplacer replacer;
  enum Domain : DomainAwareReplacer::DomainId { Inc, Dec };
  replacer.addReplacement(
      [&](IntegerAttr attr) -> std::pair<Attribute, WalkResult> {
        int64_t newNum = attr.getInt() + 1;
        auto result = IntegerAttr::get(attr.getType(), newNum);
        // Switch to dec on even result.
        if (newNum % 2 == 0) {
          auto partial =
              cast<ArrayAttr>(*replacer.replace(result, Domain::Dec));
          SmallVector<Attribute> partialAttrs(partial.getValue());
          partialAttrs.push_back(result);
          return {ArrayAttr::get(&context, partialAttrs), WalkResult::skip()};
        }
        return {ArrayAttr::get(&context, {result}), WalkResult::skip()};
      },
      Domain::Inc);
  replacer.addReplacement(
      [&](IntegerAttr attr) -> std::pair<Attribute, WalkResult> {
        int64_t newNum = attr.getInt() - 1;
        auto result = IntegerAttr::get(attr.getType(), newNum);
        // Switch to inc on odd result.
        if (newNum % 2 == 1) {
          auto partial =
              cast<ArrayAttr>(*replacer.replace(result, Domain::Inc));
          SmallVector<Attribute> partialAttrs(partial.getValue());
          partialAttrs.push_back(result);
          return {ArrayAttr::get(&context, partialAttrs), WalkResult::skip()};
        }
        return {ArrayAttr::get(&context, {result}), WalkResult::skip()};
      },
      Domain::Dec);
  // On cycle, return a StringAttr for the domain it stopped in.
  replacer.addCycleBreaker(
      [&](IntegerAttr attr) {
        return ArrayAttr::get(&context, {StringAttr::get(&context, "Inc")});
      },
      Domain::Inc);
  replacer.addCycleBreaker(
      [&](IntegerAttr attr) {
        return ArrayAttr::get(&context, {StringAttr::get(&context, "Dec")});
      },
      Domain::Dec);

  // ===== Tests =====
  auto getInt = [&](int32_t num) {
    return IntegerAttr::get(IntegerType::get(&context, 32), num);
  };

  IntegerAttr int42 = getInt(42);
  // Inc becomes an odd number. Should not create cycles.
  EXPECT_EQ(*replacer.replace(int42, Domain::Inc),
            ArrayAttr::get(&context, {getInt(43)}));
  // Dec becomes on odd number. Will trigger cycle and get back to 42 again.
  EXPECT_EQ(*replacer.replace(int42, Domain::Dec),
            ArrayAttr::get(&context, {StringAttr::get(&context, "Dec"),
                                      getInt(42), getInt(41)}));

  // Inc results in the replacement of 42 under the Dec domain, which has
  // already been performed above and cached as an independent result. Should
  // just reuse the result.
  EXPECT_EQ(*replacer.replace(getInt(41), Domain::Inc),
            ArrayAttr::get(&context, {StringAttr::get(&context, "Dec"),
                                      getInt(42), getInt(41), getInt(42)}));
  // Dec becomes an even number. Should not create cycles.
  EXPECT_EQ(*replacer.replace(getInt(43), Domain::Dec),
            ArrayAttr::get(&context, {getInt(42)}));
}
