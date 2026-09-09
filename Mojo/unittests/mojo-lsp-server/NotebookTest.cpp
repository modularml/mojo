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

static lsp::DidChangeNotebookDocumentParams buildChangeParams(
    NotebookDocument &doc, const lsp::NotebookCellArrayChange &arrayChange,
    const std::vector<lsp::NotebookDocumentChangeEvent::CellsTextContent>
        &textContent = {}) {
  return lsp::DidChangeNotebookDocumentParams{
      lsp::VersionedNotebookDocumentIdentifier{doc.getURI(), /*version=*/0},
      lsp::NotebookDocumentChangeEvent{lsp::NotebookDocumentChangeEvent::Cells{
          lsp::NotebookDocumentChangeEvent::CellsStructure{
              arrayChange,
              /*didOpen=*/{},
              /*didClose=*/{},
          },
          /*data=*/{},
          /*textContent=*/textContent}}};
}

TEST(NotebookTest, TestRemoval) {
  NotebookDocument doc("test:///test_updates", {
                                                   R"(
def function() -> Int:
  return 10
)",
                                                   R"(
_ = function()
)"});

  createTestClient()
      .openNotebook(doc)
      .notebookDidChange(buildChangeParams(
          doc, lsp::NotebookCellArrayChange{
                   /*start=*/0,
                   /*deleteCount=*/1,
                   {lsp::NotebookCell{lsp::NotebookCellKind::Code,
                                      doc.getCells()[0].getURI()}}}))
      .onDiagnostics(doc.getCells()[1],
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "use of unknown declaration 'function'");
                     })
      .execute();
}

TEST(NotebookTest, testRemovalReadd) {
  NotebookDocument doc("test:///test_updates", {
                                                   R"(
def function() -> Int:
  return 10
)",
                                                   R"(
_ = function()
)"});

  createTestClient()
      .openNotebook(doc)
      .notebookDidChange(buildChangeParams(
          doc, lsp::NotebookCellArrayChange{
                   /*start=*/0,
                   /*deleteCount=*/1,
                   {lsp::NotebookCell{lsp::NotebookCellKind::Code,
                                      doc.getCells()[0].getURI()}}}))
      .notebookDidChange(buildChangeParams(
          doc,
          lsp::NotebookCellArrayChange{/*start=*/0,
                                       /*deleteCount=*/0,
                                       /*cells=*/{}},
          {lsp::NotebookDocumentChangeEvent::CellsTextContent{
               lsp::VersionedTextDocumentIdentifier{doc.getCells()[0].getURI(),
                                                    /*version=*/0},
               {lsp::TextDocumentContentChangeEvent{
                    /*range=*/std::nullopt, /*rangeLength=*/std::nullopt,
                    /*text=*/doc.getCells()[0].getContents().str()},
                lsp::TextDocumentContentChangeEvent{
                    /*range=*/doc.getCells()[0].findFirstRange("function"),
                    /*rangeLength=*/std::nullopt,
                    /*text=*/"renamed_function"}}},
           lsp::NotebookDocumentChangeEvent::CellsTextContent{
               lsp::VersionedTextDocumentIdentifier{doc.getCells()[1].getURI(),
                                                    /*version=*/0},
               {lsp::TextDocumentContentChangeEvent{
                   /*range=*/doc.getCells()[1].findFirstRange("function"),
                   /*rangeLength=*/std::nullopt,
                   /*text=*/"renamed_function"}}}}))
      .hover(doc.getCells()[1], lsp::Position(1, 4),
             [](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) def renamed_function() -> Int
```)");
               EXPECT_EQ(hover.range, lsp::Range({1, 4}, {1, 20}));
             })
      .execute();
}

TEST(NotebookTest, testSignatureHelp) {
  NotebookDocument doc("test:///test_signature_help", {
                                                          R"(
struct SomeStruct:
    var a_field: Int

    def __init__(out self):
        pass

    def __init__(out self, a_field: Int):
        pass
)",
                                                          R"(
SomeStruct()
)"});

  createTestClient()
      .openNotebook(doc)
      .signatureHelp(doc.getCells()[1],
                     doc.getCells()[1].findLastRange("SomeStruct(")->end,
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

TEST(NotebookTest, testPython) {
  NotebookDocument doc("test:///test_python", {
                                                  R"(%%python
def function():
  return
)",
                                                  R"(
function
)"});

  createTestClient()
      .openNotebook(doc)
      .hover(doc.getCells()[1], lsp::Position(1, 2),
             [](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) mut function: PythonObject
```)");
             })
      .execute();
}

TEST(NotebookTest, testCompletion) {
  NotebookDocument doc("test:///test_completion", {
                                                      R"(
def function() -> Int:
  return 10
)",
                                                      R"(
fu
)"});

  createTestClient()
      .openNotebook(doc)
      .completion(
          doc.getCells()[1], lsp::Position(1, 2),
          [](const lsp::CompletionList &completionList) {
            EXPECT_TRUE(llvm::any_of(
                completionList.items, [](const lsp::CompletionItem &item) {
                  return item.label == "function" &&
                         item.kind == lsp::CompletionItemKind::Function;
                }));
          })
      .execute();
}
