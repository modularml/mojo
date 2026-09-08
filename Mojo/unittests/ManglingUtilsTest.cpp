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

#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {
class MangleParameterValuesTest : public ::testing::Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  MangleParameterValuesTest() { ctx.loadDialect<KGENDialect>(); }

  /// Parses a module holding one single-parameter generator named `name`.
  GeneratorOpInterface getGenerator(StringRef name = "gen") {
    std::string source =
        ("kgen.generator @\"" + name + "\"<p: string>() { kgen.return }").str();
    return getGeneratorFromSource(source);
  }

  /// Parses `source` as a module and returns its first generator.
  GeneratorOpInterface getGeneratorFromSource(StringRef source) {
    modules.push_back(parseSourceString<ModuleOp>(source, &ctx));
    ModuleOp module = *modules.back();
    assert(module && "failed to parse the test generator");
    return *module.getOps<GeneratorOpInterface>().begin();
  }

  TypedAttr stringParam(StringRef value) {
    return cast<TypedAttr>(StringAttr::get(value, StringType::get(&ctx)));
  }

  static std::string show(StringRef mangled) {
    std::string result;
    for (char c : mangled) {
      if (c == '\033')
        result += "<ESC>";
      else
        result.push_back(c);
    }
    return result;
  }

private:
  SmallVector<OwningOpRef<ModuleOp>> modules;
};
} // namespace

TEST_F(MangleParameterValuesTest, NoParameterValuesIsTheGeneratorName) {
  EXPECT_EQ(mangleParameterValues(getGenerator(), {}), "gen");
}

TEST_F(MangleParameterValuesTest, ParameterValuesAreNamedInTheMangling) {
  // String params print as `"..."`, then `"` is encoded to `~Q` on join so
  // NotEscaped and Escaped paths cannot disagree on quote spelling.
  EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam("42")})),
            "gen,p=~Q42~Q");
}

TEST_F(MangleParameterValuesTest, AtSignIsEncodedAsTildeA) {
  const std::pair<StringRef, StringRef> cases[] = {
      {"@", "gen,p=~Q~A~Q"},           {"a@b", "gen,p=~Qa~Ab~Q"},
      {"@@", "gen,p=~Q~A~A~Q"},        {"a@@b", "gen,p=~Qa~A~Ab~Q"},
      {"a@@@b", "gen,p=~Qa~A~A~Ab~Q"}, {"a@@@@b", "gen,p=~Qa~A~A~A~Ab~Q"},
  };
  for (auto [value, expected] : cases)
    EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam(value)})),
              expected)
        << "for parameter value " << value.str();
}

TEST_F(MangleParameterValuesTest, NoAtSignSurvivesEncoding) {
  for (StringRef value :
       {"@", "@@", "a@b", "a@@b", "@a@", "@@@", "a@@@@b", "@A"}) {
    std::string mangled =
        mangleParameterValues(getGenerator(), {stringParam(value)});
    EXPECT_EQ(mangled.find('@'), std::string::npos)
        << "parameter value " << value.str() << " mangled to " << show(mangled);
  }
}

TEST_F(MangleParameterValuesTest, TildeInNotEscapedTextIsDoubled) {
  EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam("a~b")})),
            "gen,p=~Qa~~b~Q");
}

// Escaped already-encoded generator names must not re-encode `~A` into `~~A`.
TEST_F(MangleParameterValuesTest, NestedMangledNameIsNotReEncoded) {
  TypedAttr param = stringParam("42");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen~A"), {param})),
            "gen~A,p=~Q42~Q");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen~~"), {param})),
            "gen~~,p=~Q42~Q");
}

TEST_F(MangleParameterValuesTest, AtInGeneratorNameIsSanitizedOnce) {
  TypedAttr param = stringParam("42");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen@"), {param})),
            "gen~A,p=~Q42~Q");
  EXPECT_EQ(show(mangleParameterValues(getGenerator("gen@A"), {param})),
            "gen~AA,p=~Q42~Q");
}

// NotEscaped path: `~` doubles, so an input that already looks encoded would
// be wrong to feed as NotEscaped — callers must mark Escaped for that. This
// pins the NotEscaped rule itself.
TEST_F(MangleParameterValuesTest, NotEscapedNestedTildeAIsEncodedOnceOnJoin) {
  EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam("~A")})),
            "gen,p=~Q~~A~Q");
}

// A parameter value cannot deliver a raw escape character: it is rendered
// through the MLIR asm printer, which writes non-printables as hex text.
TEST_F(MangleParameterValuesTest, EscapeCharacterInAValueArrivesAsHexText) {
  EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam("\033")})),
            "gen,p=~Q\\1B~Q");
}

// Quotes must become `~Q` on both fragment kinds so host stubs and offload
// kernelInfo names cannot disagree when the same `"` arrives via an Escaped
// generator name vs a NotEscaped param print.
TEST_F(MangleParameterValuesTest, QuoteInEscapedGeneratorNameIsSanitized) {
  TypedAttr param = stringParam("42");
  // `\22` is MLIR's hex escape for `"`, so the generator sym_name is `gen"`.
  EXPECT_EQ(show(mangleParameterValues(
                getGeneratorFromSource(
                    R"(kgen.generator @"gen\22"<p: string>() { kgen.return })"),
                {param})),
            "gen~Q,p=~Q42~Q");
  EXPECT_EQ(show(mangleParameterValues(
                getGeneratorFromSource(
                    R"(kgen.generator @"a\22b"<p: string>() { kgen.return })"),
                {param})),
            "a~Qb,p=~Q42~Q");
}

TEST_F(MangleParameterValuesTest, QuoteInNotEscapedStringParamIsSanitized) {
  // Asm printer writes an embedded quote as the hex text `\22` inside the
  // wrapping `"..."`, then join encodes those wrapping quotes to `~Q`.
  EXPECT_EQ(show(mangleParameterValues(getGenerator(), {stringParam("a\"b")})),
            "gen,p=~Qa\\22b~Q");
}
