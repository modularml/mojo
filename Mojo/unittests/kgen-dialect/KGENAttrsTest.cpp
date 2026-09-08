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

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "llvm/ADT/APInt.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace testing;

namespace {
class SIMDAttrTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  SIMDAttrTest() { ctx.loadDialect<KGENDialect>(); }
};
} // namespace

// An `index` value has a target-dependent bit width (e.g. 64-bit host vs.
// 32-bit offload target), so the same logical value can reach attribute
// uniquing stored in APInts of different widths. `DTypeValue::operator==` uses
// `APInt::isSameValue` and treats those as equal, so the storage uniquer must
// hash them equally too. If it does not, the same value yields two distinct
// `SIMDAttr` instances, which silently breaks any pointer-identity-based dedup
// downstream (e.g. the elaborator's instantiation cache) and leads to duplicate
// symbol definitions.
TEST_F(SIMDAttrTest, IndexValueUniquesAcrossBitWidths) {
  auto indexDType = KGENDType(KGENDType::index);
  auto scalarIndexType = SIMDType::get(&ctx, /*size=*/1, indexDType);

  // The same logical value (64) stored in a 64-bit and a 32-bit APInt.
  auto wide = SIMDAttr::get(DTypeValue(llvm::APInt(64, 64), indexDType),
                            scalarIndexType);
  auto narrow = SIMDAttr::get(DTypeValue(llvm::APInt(32, 64), indexDType),
                              scalarIndexType);

  // Must unique to the exact same attribute instance.
  EXPECT_EQ(wide, narrow);
  EXPECT_EQ(wide.getAsOpaquePointer(), narrow.getAsOpaquePointer());

  // A genuinely different value must remain distinct.
  auto other = SIMDAttr::get(DTypeValue(llvm::APInt(64, 65), indexDType),
                             scalarIndexType);
  EXPECT_NE(wide, other);
}
