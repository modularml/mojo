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

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using namespace DebugInfo;
using namespace mlir;
using namespace testing;

//===----------------------------------------------------------------------===//
// SourceNameAttrTest
//===----------------------------------------------------------------------===//

namespace {
class SourceNameAttrTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  SourceNameAttrTest() { ctx.loadDialect<DebugInfoDialect>(); }
};
} // namespace

TEST_F(SourceNameAttrTest, TestEncodeDecode) {
  StringRef testStr = R"mlir(
    #builtin_name = #debuginfo.source_name<"builtin">
    #test_name = #debuginfo.source_name<"test">
    #int_name = #debuginfo.source_name<"int" from #builtin_name>
    #simd_name = #debuginfo.source_name<"simd" from #builtin_name>
    #Int_name = #debuginfo.source_name<"Int" from #int_name>
    #SIMD_name = #debuginfo.source_name<"SIMD"[#Int_name] from #simd_name>
    #func_name = #debuginfo.source_name<("fn")"func"(#SIMD_name)<"1"> from #test_name>

    #strange = #debuginfo.source_name<"strange*">
    #weird = #debuginfo.source_name<"weird&name"<":struct<index> { 1 }", "^&*"> from #strange>

    module attributes {
      kgen.test0 = #func_name,
      kgen.test1 = #weird
    } {}
  )mlir";

  OwningOpRef<ModuleOp> module = mlir::parseSourceString<ModuleOp>(
      testStr, mlir::ParserConfig(&ctx), "TestEncode_testStr");
  ASSERT_TRUE(module);

  {
    auto sourceName = (*module)->getAttrOfType<SourceNameAttr>("kgen.test0");
    ASSERT_TRUE(sourceName);

    StringRef expected =
        "test::fn func(builtin::simd::SIMD[builtin::int::Int])<1>";
    EXPECT_EQ(sourceName.encode().getValue(), expected);

    ErrorOr<SourceNameAttr> decoded = SourceNameAttr::decode(&ctx, expected);
    ASSERT_FALSE(decoded.isError());
    EXPECT_EQ(decoded.takeValue(), sourceName);
  }

  {
    auto sourceName = (*module)->getAttrOfType<SourceNameAttr>("kgen.test1");
    ASSERT_TRUE(sourceName);

    StringRef expected =
        "`strange*`::`weird&name`<`:struct<index> { 1 }`,`^&*`>";
    EXPECT_EQ(sourceName.encode().getValue(), expected);

    ErrorOr<SourceNameAttr> decoded = SourceNameAttr::decode(&ctx, expected);
    ASSERT_FALSE(decoded.isError());
    EXPECT_EQ(decoded.takeValue(), sourceName);
  }
}

//===----------------------------------------------------------------------===//
// DIScopeAttrUtilTest
//===----------------------------------------------------------------------===//

namespace {
class DIScopeAttrUtilTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  DIScopeAttrUtilTest() { ctx.loadDialect<DebugInfoDialect>(); }

  DISubprogramAttr getSimpleSubprogram(StringRef name) {
    return DISubprogramAttr::get(
        &ctx, {}, {}, SourceNameAttr::get(StringAttr::get(&ctx, name)), {}, {},
        {}, {}, {}, {});
  }
};
} // namespace

TEST_F(DIScopeAttrUtilTest, TestScopeWalk) {
  std::vector<DISubprogramAttr> sp;
  std::vector<Location> fused;
  for (int64_t i = 0; i < 3; ++i) {
    std::string suffix = std::to_string(i);
    sp.push_back(getSimpleSubprogram("func" + suffix));
    Location loc =
        FileLineColLoc::get(StringAttr::get(&ctx, "foo" + suffix), 1, 1);
    fused.emplace_back(
        FusedLocWith<DISubprogramAttr>::get(&ctx, {loc}, sp.back()));
  }

  auto getVisitedScopes = [](Location loc,
                             LocWalkPolicy policy) -> std::vector<DIScopeAttr> {
    std::vector<DIScopeAttr> visitedScopes;
    walkScope(loc, policy, [&](DIScopeAttr scope) {
      visitedScopes.push_back(scope);
      return WalkResult::advance();
    });
    return visitedScopes;
  };

  // Call tree:
  // 2 -> 1 -> 0
  Location innerCallsite = CallSiteLoc::get(fused[0], fused[1]);
  Location callsite = CallSiteLoc::get(innerCallsite, fused[2]);
  EXPECT_THAT(getVisitedScopes(callsite, LocWalkPolicy::CalleePriority),
              ElementsAre(sp[0], sp[1], sp[2]));
  EXPECT_THAT(getVisitedScopes(callsite, LocWalkPolicy::CallerPriority),
              ElementsAre(sp[2], sp[1], sp[0]));
}

TEST_F(DIScopeAttrUtilTest, TestExtractScopeFrom) {
  std::vector<DISubprogramAttr> sp;
  std::vector<DILexicalBlockAttr> block;
  std::vector<Location> fused;
  for (int64_t i = 0; i < 3; ++i) {
    std::string suffix = std::to_string(i);
    sp.push_back(getSimpleSubprogram("func" + suffix));
    block.push_back(DILexicalBlockAttr::get(&ctx, sp.back(), {}, {}, {}));
    Location loc =
        FileLineColLoc::get(StringAttr::get(&ctx, "foo" + suffix), 1, 1);
    fused.emplace_back(
        FusedLocWith<DISubprogramAttr>::get(&ctx, {loc}, block.back()));
  }

  // Call tree:
  // 2 -> 1 -> 0
  Location innerCallsite = CallSiteLoc::get(fused[0], fused[1]);
  Location callsite = CallSiteLoc::get(innerCallsite, fused[2]);

  // Find top scope.
  EXPECT_EQ(extractScopeFrom<DILexicalBlockAttr>(callsite,
                                                 LocWalkPolicy::CalleePriority),
            block[0]);
  EXPECT_EQ(extractScopeFrom<DILexicalBlockAttr>(callsite,
                                                 LocWalkPolicy::CallerPriority),
            block[2]);

  // Find nested scope.
  EXPECT_EQ(extractScopeFrom<DISubprogramAttr>(callsite,
                                               LocWalkPolicy::CalleePriority),
            sp[0]);
  EXPECT_EQ(extractScopeFrom<DISubprogramAttr>(callsite,
                                               LocWalkPolicy::CallerPriority),
            sp[2]);
}

TEST_F(DIScopeAttrUtilTest, TestExtractSourceLoc) {
  std::vector<DISubprogramAttr> sp;
  std::vector<FileLineColLoc> loc;
  std::vector<Location> fused;
  for (int64_t i = 0; i < 3; ++i) {
    std::string suffix = std::to_string(i);
    sp.push_back(getSimpleSubprogram("func" + suffix));
    loc.push_back(
        FileLineColLoc::get(StringAttr::get(&ctx, "foo" + suffix), 1, 1));
    fused.emplace_back(
        FusedLocWith<DISubprogramAttr>::get(&ctx, {loc.back()}, sp.back()));
  }

  {
    // Call tree:
    // 2 -> 1 -> 0
    Location innerCallsite = CallSiteLoc::get(fused[0], fused[1]);
    Location callsite = CallSiteLoc::get(innerCallsite, fused[2]);
    EXPECT_EQ(extractSourceLoc(callsite), loc[0]);
  }

  {
    // Call tree:
    // 2(raw) -> 1(raw) -> 0(raw)
    Location innerCallsite = CallSiteLoc::get(loc[0], loc[1]);
    Location callsite = CallSiteLoc::get(innerCallsite, loc[2]);
    EXPECT_EQ(extractSourceLoc(callsite), loc[0]);
  }

  {
    // Call tree:
    // 2 -> 1(raw) -> 0
    Location innerCallsite = CallSiteLoc::get(fused[0], loc[1]);
    Location callsite = CallSiteLoc::get(innerCallsite, fused[2]);
    EXPECT_EQ(extractSourceLoc(callsite), loc[0]);
  }
}
