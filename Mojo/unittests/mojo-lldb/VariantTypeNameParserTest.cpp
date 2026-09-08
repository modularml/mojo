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

#include "Mojo/lib/MojoLLDB/Language/Formatters/MojoVariantTypeNameParser.h"
#include "gtest/gtest.h"

using M::KGEN::Mojo::extractVariantTypeNames;

namespace {

// Representative DWARF-form `Variant[Int, String]` storage type name.
// `\22` is LLDB's octal encoding of `"`.
constexpr const char *kDWARFTwoArm =
    R"(!lit.struct<@std::@utils::@variant::@"_DefaultVariantStorage[:param_list<trait<@\22std::builtin::anytype::AnyType\22>> [@\22std::builtin::int::Int\22, @\22std::collections::string::string::String\22]]">)";

// Three-arm variant `Variant[Int, Bool, String]` — the case
// `testVariant` exercises for non-zero discriminants.
constexpr const char *kDWARFThreeArm =
    R"(!lit.struct<@std::@utils::@variant::@"_DefaultVariantStorage[:param_list<trait<@\22std::builtin::anytype::AnyType\22>> [@\22std::builtin::int::Int\22, @\22std::builtin::bool::Bool\22, @\22std::collections::string::string::String\22]]">)";

// aarch64-Linux DWARF form (MOCO-3787): the symbol carries a trailing
// `TypeList` parameter that re-embeds the same types in a second
// `[@"...", @"..."]`. The parser must stop at the FIRST closing `"]`
// after the opener to avoid merging the two lists into 3 entries with a
// garbled middle, which used to produce `summary=Int` for every
// non-zero discriminant.
constexpr const char *kDWARFAArch64WithTrailingTypeList =
    R"(!lit.struct<@std::@utils::@variant::@"Variant[:param_list<trait<@\22std::builtin::value::Movable\22>> [@\22std::builtin::int::Int\22, @\22std::collections::string::string::String\22], :!lit.struct<@\22std::builtin::variadics::TypeList\22<:!lit.anytrait<<@\22std::builtin::anytype::AnyType\22>> trait<@\22std::builtin::value::Movable\22>, :param_list<trait<@\22std::builtin::value::Movable\22>> [@\22std::builtin::int::Int\22, @\22std::collections::string::string::String\22]>> {}]">)";

// Representative REPL-form `Variant[Int, String]` type name: type
// references are `@`-delimited without quoting.
constexpr const char *kREPLTwoArm =
    "!lit.struct<@std::@utils::@variant::@_DefaultVariantStorage<:param_list<"
    "trait<@std::@builtin::@anytype::@AnyType>> [@std::@builtin::@int::@Int, "
    "@std::@collections::@string::@string::@String]>>";

} // namespace

TEST(VariantTypeNameParserTest, DWARFTwoArm) {
  auto names = extractVariantTypeNames(kDWARFTwoArm);
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0].fullName, "std::builtin::int::Int");
  EXPECT_EQ(names[0].displayName, "Int");
  EXPECT_EQ(names[1].fullName, "std::collections::string::string::String");
  EXPECT_EQ(names[1].displayName, "String");
}

TEST(VariantTypeNameParserTest, DWARFThreeArm) {
  auto names = extractVariantTypeNames(kDWARFThreeArm);
  ASSERT_EQ(names.size(), 3u);
  EXPECT_EQ(names[0].displayName, "Int");
  EXPECT_EQ(names[1].displayName, "Bool");
  EXPECT_EQ(names[2].displayName, "String");
}

// Regression test for the aarch64-Linux CI failure that kicked off
// MOCO-3787. Before the fix, the parser's `rfind(\22])` landed on the
// closing delimiter of the nested `TypeList`'s list, producing 3 entries
// instead of 2 and mis-indexing the String arm as "Int".
TEST(VariantTypeNameParserTest, DWARFTrailingTypeListDoesNotBleed) {
  auto names = extractVariantTypeNames(kDWARFAArch64WithTrailingTypeList);
  ASSERT_EQ(names.size(), 2u)
      << "Parser must stop at the first closing delimiter of the Ts pack, "
         "not swallow the trailing TypeList parameter.";
  EXPECT_EQ(names[0].displayName, "Int");
  EXPECT_EQ(names[1].displayName, "String");
  EXPECT_EQ(names[0].fullName, "std::builtin::int::Int");
  EXPECT_EQ(names[1].fullName, "std::collections::string::string::String");
}

TEST(VariantTypeNameParserTest, REPLTwoArm) {
  auto names = extractVariantTypeNames(kREPLTwoArm);
  ASSERT_EQ(names.size(), 2u);
  // REPL form uses `@`-separated components; the parser strips the `@`s
  // so the canonical FQN matches the DWARF form.
  EXPECT_EQ(names[0].fullName, "std::builtin::int::Int");
  EXPECT_EQ(names[0].displayName, "Int");
  EXPECT_EQ(names[1].fullName, "std::collections::string::string::String");
  EXPECT_EQ(names[1].displayName, "String");
}

TEST(VariantTypeNameParserTest, EmptyStringReturnsEmpty) {
  EXPECT_TRUE(extractVariantTypeNames("").empty());
}

TEST(VariantTypeNameParserTest, UnrecognizedFormatReturnsEmpty) {
  // No `[@"` opener and no ` [@` REPL opener.
  EXPECT_TRUE(extractVariantTypeNames("!lit.struct<@foo::@Bar>").empty());
}

TEST(VariantTypeNameParserTest, DWARFMissingCloseReturnsEmpty) {
  // Has `[@"` but no closing `"]`.
  EXPECT_TRUE(
      extractVariantTypeNames(R"(!lit.struct<...[@\22Int\22, @\22String\22)")
          .empty());
}

// If the DWARF opener is present, the parser must commit to DWARF and
// fail closed on a malformed close — it must NOT silently reinterpret the
// input as REPL form just because a ` [@...]` substring happens to match
// the REPL shape. Protects against the malformed-DWARF-as-REPL
// misidentification flagged in code review.
TEST(VariantTypeNameParserTest, DWARFMalformedDoesNotReparseAsREPL) {
  EXPECT_TRUE(
      extractVariantTypeNames(R"(!lit.struct<[@\22X\22 [@bar, @baz]>)").empty())
      << "DWARF opener present → parser must fail closed, not reparse the "
         "REPL-shaped substring ` [@bar, @baz]`.";
}
