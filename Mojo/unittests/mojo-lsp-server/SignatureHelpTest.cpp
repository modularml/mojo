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

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(SignatureHelpTest, testSignatureHelpOverload) {
  Document doc("test:///foo.mojo", R"(
def function(): # skip
    return
def function(arg: Int) -> Int: # skip
    return arg
def function(arg: Bool, arg2: Int) -> Int: # skip
    return arg2

def test():
    function()
    function(10)
    function(True, 10)
)");

  std::vector<lsp::Range> ranges = doc.findAllRanges("function(");
  ASSERT_EQ((int)ranges.size(), 3);

  createTestClient()
      .open(doc)
      .signatureHelp(doc, ranges[0].end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 3);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "def function()");
                       EXPECT_EQ(signatureHelp.signatures[1].label,
                                 "def function(arg: Int) -> Int");
                       EXPECT_EQ(signatureHelp.signatures[2].label,
                                 "def function(arg: Bool, arg2: Int) -> Int");
                     })
      .signatureHelp(doc, ranges[1].end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 5);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "def __init__() -> Self");
                       EXPECT_EQ(
                           signatureHelp.signatures[1].label,
                           "def __init__(out self, *, deinit move: Self)");
                       EXPECT_EQ(signatureHelp.signatures[2].label,
                                 "def __init__(out self, *, copy: Self)");
                       EXPECT_EQ(signatureHelp.signatures[3].label,
                                 "def function(arg: Int) -> Int");
                       EXPECT_EQ(signatureHelp.signatures[4].label,
                                 "def function(arg: Bool, arg2: Int) -> Int");
                     })
      .signatureHelp(doc, ranges[2].end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 1);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "def function(arg: Bool, arg2: Int) -> Int");
                     })
      .signatureHelp(doc, doc.findLastRange("True,")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 4);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 1);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "def __init__() -> Self");
                       EXPECT_EQ(
                           signatureHelp.signatures[1].label,
                           "def __init__(out self, *, deinit move: Self)");
                       EXPECT_EQ(signatureHelp.signatures[2].label,
                                 "def __init__(out self, *, copy: Self)");
                       EXPECT_EQ(signatureHelp.signatures[3].label,
                                 "def function(arg: Bool, arg2: Int) -> Int");
                     })
      .execute();
}

TEST(SignatureHelpTest, testSignatureHelpTypeCall) {
  Document doc("test:///foo.mojo", R"(
struct SomeStruct:
    var a_field: Int

    def __init__(out self):
        pass

    def __init__(out self, a_field: Int):
        pass

def test():
    SomeStruct()
)");

  createTestClient()
      .open(doc)
      .signatureHelp(doc, doc.findLastRange("SomeStruct(")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 3);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "def __init__(out self)");
                       EXPECT_EQ(signatureHelp.signatures[1].label,
                                 "def __init__(out self, a_field: Int)");
                       EXPECT_EQ(
                           signatureHelp.signatures[2].label,
                           "def __init__(out self, *, deinit move: Self)");
                     })
      .execute();
}

TEST(SignatureHelpTest, testSignatureOverloadParams) {
  Document doc("test:///foo.mojo", R"(
def function[type: DType](): # skip
    return
def function[type: DType, type2: DType](): # skip
    return

def test():
    function[DType.bool]()
    function[DType.bool, DType.bool]()
)");

  createTestClient()
      .open(doc)
      .signatureHelp(doc, doc.findFirstRange("function[")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 2);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "def function[type: DType]()");
                       EXPECT_EQ(signatureHelp.signatures[1].label,
                                 "def function[type: DType, type2: DType]()");
                     })
      .signatureHelp(doc, doc.findLastRange("DType.bool,")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 1);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 1);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "def function[type: DType, type2: DType]()");
                     })
      .execute();
}

TEST(SignatureHelpTest, testSignatureHelpParams) {
  Document doc("test:///foo.mojo", R"(
struct SomeStruct[dtype: DType]: # skip
    def __init__(out self):
        pass

def test():
    SomeStruct[DType.bool]()
)");

  createTestClient()
      .open(doc)
      .signatureHelp(doc, doc.findLastRange("SomeStruct[")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 1);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 R"(struct SomeStruct[dtype: DType]
# Traits: AnyType, Deinitable, Movable)");
                     })
      .execute();
}
