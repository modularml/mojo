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

#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENCompilationContext.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace KGEN::LIT;
using namespace mlir;
using namespace testing;

//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//

namespace {
class FuncTypeGeneratorTypeTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  FuncTypeGeneratorTypeTest() { ctx.loadDialect<KGENDialect, LITDialect>(); }
};
class EnvAttrTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  EnvAttrTest() { ctx.loadDialect<KGENDialect>(); }
};
} // namespace

TEST_F(FuncTypeGeneratorTypeTest, TestSpecialization) {
  auto indexType = IndexType::get(&ctx);
  auto typeType = TypeType::get(&ctx);
  auto indexTypeAttr = TypeParamAttr::get(indexType, typeType);
  auto ref0Type = ParamType::get(ParamIndexRefAttr::get(0, typeType));
  FunctionType funcType = FunctionType::get(&ctx, {ref0Type}, {ref0Type});
  SmallVector<Type> inputParamTypes = {typeType};

  // Test bare KGEN Signature
  {
    FuncTypeGeneratorType sigGen =
        FuncTypeGeneratorType::get(inputParamTypes, funcType);
    FuncTypeGeneratorType concreteSigGen = sigGen.getSpecializedGenerator(
        {indexTypeAttr}, /*evaluationContext=*/nullptr);

    EXPECT_EQ(concreteSigGen,
              FuncTypeGeneratorType::get(
                  /*inputParamTypes=*/{},
                  FunctionType::get(&ctx, {indexType}, {indexType})));
  }

  // Test Signature with metadata
  {
    auto posOnly = PogMetadataAttr::get(
        StringAttr::get(&ctx), PassingKind::PosOnly, VariadicKind::None);
    PogListAttr pogs =
        PogListAttr::get(&ctx, SmallVector<PogMetadataAttr>{posOnly});
    FnMetaOriginDataAttr fnMetadata = FnMetaOriginDataAttr::get(
        &ctx,
        /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
        /*isNestedOriginsReadOnly=*/false, /*definesInteriorOrigins=*/false);
    FnMetaOriginDataAttr fnMetadataNoParams = FnMetaOriginDataAttr::get(
        &ctx,
        /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
        /*isNestedOriginsReadOnly=*/false, /*definesInteriorOrigins=*/false);
    FuncTypeGeneratorType sigGen =
        FuncTypeGeneratorType::get(inputParamTypes, funcType, /*argConvs=*/{},
                                   /*effects=*/{}, fnMetadata,
                                   /*genMetadata=*/pogs, /*argListAttrs=*/pogs);
    FuncTypeGeneratorType concreteSigGen = sigGen.getSpecializedGenerator(
        {indexTypeAttr}, /*evaluationContext=*/nullptr);

    EXPECT_EQ(concreteSigGen,
              FuncTypeGeneratorType::get(
                  /*inputParamTypes=*/{},
                  FunctionType::get(&ctx, {indexType}, {indexType}),
                  /*argConvs=*/{},
                  /*effects=*/{}, fnMetadataNoParams,
                  /*genMetadata=*/PogListAttr::get(&ctx),
                  /*argListAttrs=*/pogs));
  }
}

TEST_F(EnvAttrTest, testEnvAttr) {
  CompilationContext compileCtx;
  // The settings below mimic the code in M_setMojoDefineBool,
  // M_setMojoDefineInt, and M_setMojoDefineString

  compileCtx.mojoDefines["TEST_BOOL_TRUE"] = true;
  compileCtx.mojoDefines["TEST_BOOL_FALSE"] = false;
  compileCtx.mojoDefines["TEST_INT"] = 42;
  compileCtx.mojoDefines["TEST_STRING"] = std::string("test_value");

  auto envAttr = getModularEnvAttr(&ctx, &compileCtx);
  ASSERT_TRUE(envAttr);

  // Test the defines we added
  auto dict = envAttr.getValues();

  auto boolAttrTrue = dict.get("TEST_BOOL_TRUE");
  ASSERT_TRUE(boolAttrTrue);
  ASSERT_TRUE(isa<UnitAttr>(boolAttrTrue));

  auto boolAttrFalse = dict.get("TEST_BOOL_FALSE");
  // False attribute is not added at all.
  ASSERT_FALSE(boolAttrFalse);

  auto intAttr = dict.get("TEST_INT");
  ASSERT_TRUE(intAttr);
  EXPECT_EQ(cast<IntegerAttr>(intAttr).getInt(), 42);

  auto strAttr = dict.get("TEST_STRING");
  ASSERT_TRUE(strAttr);
  EXPECT_EQ(cast<StringAttr>(strAttr).getValue(), "test_value");

  // Check that the string attribute has the correct KGEN string type
  EXPECT_TRUE(isa<StringType>(cast<StringAttr>(strAttr).getType()));
}

TEST_F(EnvAttrTest, testParseDefines) {
  // Test mixed defines
  {
    std::vector<std::string> defines = {"TEST_BOOL_TRUE", "TEST_INT=100",
                                        "TEST_STRING=mytest"};
    auto result = EnvAttr::parseDefines(&ctx, defines);
    ASSERT_FALSE(result.isError());

    auto envAttr = result.get();
    auto dict = envAttr.getValues();

    auto boolAttr = dict.get("TEST_BOOL_TRUE");
    ASSERT_TRUE(boolAttr);
    EXPECT_TRUE(isa<UnitAttr>(boolAttr));

    auto intAttr = dict.get("TEST_INT");
    ASSERT_TRUE(intAttr);
    EXPECT_EQ(cast<IntegerAttr>(intAttr).getInt(), 100);

    auto stringAttr = dict.get("TEST_STRING");
    ASSERT_TRUE(stringAttr);
    EXPECT_EQ(cast<StringAttr>(stringAttr).getValue(), "mytest");
  }

  // Test duplicate defines (should fail)
  {
    std::vector<std::string> defines = {"FOO=1", "FOO=2"};
    auto result = EnvAttr::parseDefines(&ctx, defines);
    ASSERT_TRUE(result.isError());
    EXPECT_STREQ(result.getError(), "'FOO=2' was defined more than once");
  }

  // Test empty value (should be treated as empty string)
  {
    std::vector<std::string> defines = {"EMPTY="};
    auto result = EnvAttr::parseDefines(&ctx, defines);
    ASSERT_FALSE(result.isError());

    auto envAttr = result.get();
    auto dict = envAttr.getValues();

    auto emptyAttr = dict.get("EMPTY");
    ASSERT_TRUE(emptyAttr);
    ASSERT_TRUE(isa<StringAttr>(emptyAttr));
    EXPECT_EQ(cast<StringAttr>(emptyAttr).getValue(), "");
  }
}

//===----------------------------------------------------------------------===//
// StructType (KGEN)
//===----------------------------------------------------------------------===//

// Use explicit namespace to avoid ambiguity with LIT::StructType.
using KGENStructType = KGEN::StructType;

namespace {
class StructTypeTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  StructTypeTest() { ctx.loadDialect<KGENDialect, LITDialect>(); }
};
} // namespace

TEST_F(StructTypeTest, UniquingConsistency) {
  // Test that StructType created via different paths produces the same type.
  auto i32Type = IntegerType::get(&ctx, 32);
  auto i64Type = IntegerType::get(&ctx, 64);

  // Create via ArrayRef<Type>
  KGENStructType fromTypes = KGENStructType::get(&ctx, {i32Type, i64Type});

  // Create via ParamListAttr with TypeParamAttrs
  auto metatype = TypeType::get(&ctx);
  auto variadicType = ParamListType::get(metatype);
  SmallVector<TypedAttr> elements = {TypeParamAttr::get(i32Type, metatype),
                                     TypeParamAttr::get(i64Type, metatype)};
  ParamListAttr variadic = ParamListAttr::get(elements, variadicType);
  KGENStructType fromVariadic = KGENStructType::get(&ctx, variadic, false);

  // These should be the exact same type (pointer equality due to uniquing).
  EXPECT_EQ(fromTypes, fromVariadic);
}

TEST_F(StructTypeTest, UniquingWithParamTypes) {
  // Test that StructType with ParamTypes as elements unique correctly.
  auto metatype = TypeType::get(&ctx);

  // Create a ParamType wrapping a ParamDeclRefAttr.
  auto paramRef = ParamDeclRefAttr::get(StringAttr::get(&ctx, "T"), metatype);
  auto paramType = ParamType::get(paramRef);

  // Create struct via ArrayRef<Type>
  KGENStructType fromTypes = KGENStructType::get(&ctx, {paramType});

  // Create via ArrayRef<Type> again - should get the same type.
  KGENStructType fromTypes2 = KGENStructType::get(&ctx, {paramType});
  EXPECT_EQ(fromTypes, fromTypes2);

  // Create via ParamListAttr using TypeParamAttr::get - should get same type.
  auto variadicType = ParamListType::get(metatype);
  SmallVector<TypedAttr> elements = {
      cast<TypedAttr>(TypeParamAttr::get(paramType, metatype))};
  ParamListAttr variadic = ParamListAttr::get(elements, variadicType);
  KGENStructType fromVariadic = KGENStructType::get(&ctx, variadic, false);

  EXPECT_EQ(fromTypes, fromVariadic);
}

TEST_F(StructTypeTest, UniquingWithCanonicalizingTypeParamAttr) {
  // This test verifies that TypeParamAttr::get() canonicalization is
  // deterministic: the same input always produces the same output, ensuring
  // consistent type uniquing even when TypeParamAttr::get() returns a
  // ParamOperatorAttr instead of a TypeParamAttr.
  auto metatype = TypeType::get(&ctx);

  // Create a ParamType wrapping a ParamDeclRefAttr.
  auto paramRef = ParamDeclRefAttr::get(StringAttr::get(&ctx, "T"), metatype);
  auto paramType = ParamType::get(paramRef);

  // Create struct via ArrayRef<Type>.
  KGENStructType fromTypes = KGENStructType::get(&ctx, {paramType});

  // Create struct via ParamListAttr using TypeParamAttr::get() directly.
  // TypeParamAttr::get(paramType, metatype) may return a ParamOperatorAttr
  // instead of a TypeParamAttr due to canonicalization.
  auto variadicType = ParamListType::get(metatype);
  TypedAttr typeParamResult = TypeParamAttr::get(paramType, metatype);

  // Note: typeParamResult might be a ParamOperatorAttr, not a TypeParamAttr!
  // Since canonicalization is deterministic, both paths produce the same attr.
  SmallVector<TypedAttr> elements = {typeParamResult};
  ParamListAttr variadic = ParamListAttr::get(elements, variadicType);
  KGENStructType fromVariadic = KGENStructType::get(&ctx, variadic, false);

  // Since canonicalization is deterministic, these should be the same type.
  EXPECT_EQ(fromTypes, fromVariadic);

  // Verify we can get the element types back correctly.
  auto elementTypes = fromVariadic.getElementTypes();
  ASSERT_TRUE(elementTypes.has_value());
  EXPECT_EQ(elementTypes->size(), 1u);
  EXPECT_EQ((*elementTypes)[0], paramType);
}

TEST_F(StructTypeTest, UniquingEmptyStruct) {
  // Test that empty structs unique correctly.
  KGENStructType empty1 = KGENStructType::get(&ctx, ArrayRef<Type>{});
  KGENStructType empty2 = KGENStructType::get(&ctx, ArrayRef<Type>{});
  EXPECT_EQ(empty1, empty2);

  // Check isNoneOrEmpty works.
  EXPECT_TRUE(KGENStructType::isNoneOrEmpty(empty1));
}

TEST_F(StructTypeTest, GetElementTypes) {
  // Test that getElementTypes returns the correct types.
  auto i32Type = IntegerType::get(&ctx, 32);
  auto i64Type = IntegerType::get(&ctx, 64);

  KGENStructType structType = KGENStructType::get(&ctx, {i32Type, i64Type});
  auto elementTypes = structType.getElementTypes();

  ASSERT_TRUE(elementTypes.has_value());
  EXPECT_EQ(elementTypes->size(), 2u);
  EXPECT_EQ((*elementTypes)[0], i32Type);
  EXPECT_EQ((*elementTypes)[1], i64Type);
}

TEST_F(EnvAttrTest, testQueryValue) {
  // Create test EnvAttr with mixed values
  std::vector<std::string> defines = {"DEBUG", "COUNT=42", "NAME=test",
                                      "FEATURE="};
  auto parseResult = EnvAttr::parseDefines(&ctx, defines);
  ASSERT_FALSE(parseResult.isError());
  auto envAttr = parseResult.get();

  // Test querying index values
  {
    auto result = envAttr.queryValue("COUNT", IndexType::get(&ctx));
    ASSERT_FALSE(result.isError());
    auto intAttr = cast<IntegerAttr>(result.get());
    EXPECT_EQ(intAttr.getInt(), 42);
  }

  // Test querying string values
  {
    auto result = envAttr.queryValue("NAME", StringType::get(&ctx));
    ASSERT_FALSE(result.isError());
    auto strAttr = cast<StringAttr>(result.get());
    EXPECT_EQ(strAttr.getValue(), "test");
  }

  // Test querying empty string value
  {
    auto result = envAttr.queryValue("FEATURE", StringType::get(&ctx));
    ASSERT_FALSE(result.isError());
    auto strAttr = cast<StringAttr>(result.get());
    EXPECT_EQ(strAttr.getValue(), "");
  }

  // Test querying unit attribute as bool
  {
    auto result = envAttr.queryValue("DEBUG", IntegerType::get(&ctx, 1));
    ASSERT_FALSE(result.isError());
    auto boolAttr = cast<BoolAttr>(result.get());
    EXPECT_TRUE(boolAttr.getValue());
  }

  // Test querying non-existent unit attribute as bool (should return false)
  {
    auto result = envAttr.queryValue("NONEXISTENT", IntegerType::get(&ctx, 1));
    ASSERT_FALSE(result.isError());
    auto boolAttr = cast<BoolAttr>(result.get());
    EXPECT_FALSE(boolAttr.getValue());
  }

  // Test int to string conversion
  {
    auto result = envAttr.queryValue("COUNT", StringType::get(&ctx));
    ASSERT_FALSE(result.isError());
    auto strAttr = cast<StringAttr>(result.get());
    EXPECT_EQ(strAttr.getValue(), "42");
  }

  // Test implicit conversion from string to bool
  // The attribute exists, but is not a bool. This still works,
  // because the test is interpreted as querying for the existence of the
  // attribute.
  {
    auto result = envAttr.queryValue("NAME", IntegerType::get(&ctx, 1));
    ASSERT_FALSE(result.isError());
    auto boolAttr = cast<BoolAttr>(result.get());
    EXPECT_TRUE(boolAttr.getValue());
  }

  // Test error cases - missing required values
  {
    auto result = envAttr.queryValue("MISSING", IndexType::get(&ctx));
    ASSERT_TRUE(result.isError());
    EXPECT_EQ(std::string(result.getError()),
              "define 'MISSING' does not exist, please provide it via -D");
  }

  {
    auto result = envAttr.queryValue("MISSING", StringType::get(&ctx));
    ASSERT_TRUE(result.isError());
    EXPECT_EQ(std::string(result.getError()),
              "define 'MISSING' does not exist, please provide it via -D");
  }

  // Test error case - wrong type conversion
  {
    auto result = envAttr.queryValue("NAME", IndexType::get(&ctx));
    ASSERT_TRUE(result.isError());
    EXPECT_TRUE(std::string(result.getError()).find("is not an integer") !=
                std::string::npos);
  }
}
