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

#include "Mojo/MojoTooling/REPLPythonExprUtils.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

/// Create the python script that extracts the top-level variables from the
/// provided python expression.
static std::string createSymbolExtractorPythonCode(StringRef pythonExpr) {
  const char *rawSymbolsExtractor = R"(
def __lldb_python_extract_symbols():
  symbols = []

  import ast

  # The following class visits only top level constructs of the given python
  # expression. It doesn't traverse recursively the AST.
  class AssignmentVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
      symbols.append(['declaration', node.name])

    def visit_Assign(self, node):
      for target in node.targets:
        if isinstance(target, ast.Name):
          symbols.append(['declaration', target.id])

    def visit_Import(self, node):
      for alias in node.names:
        asname = alias.asname if alias.asname else alias.name
        symbols.append(['import', asname, alias.name])

    # We remove the default implementation of the following node visitors to
    # prevent finding assignments recursively.
    def visit_AsyncFunctionDef(self, node):
      pass

    def visit_ClassDef(self, node):
      pass


  try:
    __lldb_python_ast = ast.parse("{0}")
    AssignmentVisitor().visit(__lldb_python_ast)
  except SyntaxError:
    import traceback
    traceback.print_exc()

  return symbols


# To simplify the parsing logic, we serialize the symbols as a multi-line
# string, where each line corresponds to a symbol, and each line is made of
# space-delimited tokens with the following format:
#
#   <symbol kind> <symbol name> [other tokens depending on the kind...]
def __lldb_python_serialize_symbols(symbols):
  serialized = []
  for symbol in symbols:
    serialized.append(' '.join(symbol))
  print('\n'.join(serialized))


__lldb_python_serialize_symbols(__lldb_python_extract_symbols())
  )";

  std::string escapedPythonExpr;
  llvm::raw_string_ostream escapedPythonExprOS(escapedPythonExpr);
  escapedPythonExprOS.write_escaped(pythonExpr);

  return llvm::formatv(rawSymbolsExtractor, escapedPythonExpr).str();
}

/// Try to get a python executable from the PATH.
static std::optional<std::string> getPythonExecutable() {
  if (auto pythonEnvVar = std::getenv("MODULAR_PYTHON_EXECUTABLE"))
    return std::string(pythonEnvVar);
  llvm::ErrorOr<std::string> pyOrErr = llvm::sys::findProgramByName("python3");
  if (!pyOrErr)
    pyOrErr = llvm::sys::findProgramByName("python");
  if (pyOrErr)
    return *pyOrErr;
  return std::nullopt;
}

/// Execute the script that extracts the top-level variables from the provided
/// python expression.
static ErrorOr<std::string>
executeVariableExtractorScript(StringRef pythonExpr) {
  // Create a temporary file to capture the output of the `python` invocation
  // and another one to store the extractor script.
  ErrorOr<TempFile> outOrErr =
      TempFile::create("mojo-repl-extractor-%%%%%%.out");
  if (failed(outOrErr))
    return Error("could not create the temporary file to capture the 'python "
                 "variable extractor' output");
  std::string pythonScriptPath = outOrErr->getPath().string();

  ErrorOr<TempFile> scriptOrErr =
      TempFile::create("mojo-repl-extractor-%%%%%%.py");
  if (failed(scriptOrErr))
    return Error("could not create the 'python variable extractor' script");
  std::string pythonScriptOutputPath = scriptOrErr->getPath().string();

  {
    std::error_code ec;
    llvm::raw_fd_stream fs(pythonScriptPath, ec);
    if (ec)
      return Error("could not set the contents of the 'python variable "
                   "extractor' script");
    fs << createSymbolExtractorPythonCode(pythonExpr);
  }

  std::optional<std::string> pythonExe = getPythonExecutable();
  if (!pythonExe)
    return Error("could not find 'python' executable to execute the 'python "
                 "variable extractor'");

  // Invoke `python`, directing its output to the file.
  const std::optional<StringRef> redirects[] = {
      /*stdin=*/"",
      /*stdout=*/pythonScriptOutputPath,
      /*stderr=*/"",
  };

  const StringRef args[] = {*pythonExe, pythonScriptPath};
  if (llvm::sys::ExecuteAndWait(*pythonExe, args, /*Env=*/std::nullopt,
                                redirects) != 0)
    return Error("could not execute the 'python variable extractor'");

  auto bufferOrErr = llvm::MemoryBuffer::getFile(pythonScriptOutputPath);
  if (!bufferOrErr)
    return Error(
        "could not parse the output of the 'python variable extractor'");

  // Here we access the result, which is a serialized description of each
  // symbol to extract.
  return bufferOrErr.get()->getBuffer().str();
}

ErrorOr<std::vector<std::unique_ptr<KGEN::Mojo::ExtractedPythonSymbol>>>
M::KGEN::Mojo::extractPythonSymbolsFromReplExpr(StringRef pythonExpr) {
  // We extract the necessary python symbols using the python ast module, which
  // requires us to invoke python.
  ErrorOr<std::string> extractorOutputOr =
      executeVariableExtractorScript(pythonExpr);
  if (failed(extractorOutputOr))
    return extractorOutputOr.takeError();

  // Here we access the result, which is a serialized description of each
  // symbol to extract.
  StringRef symbols = extractorOutputOr.get();

  SmallVector<StringRef> symbolLines;
  symbols.split(symbolLines, '\n', /*MaxSplit=*/-1, /*keepEmpty=*/false);

  // We process the symbols in reverse order so that we honor the last
  // occurrence of a given symbol name.
  llvm::StringSet<> seenVariables;
  std::vector<std::unique_ptr<ExtractedPythonSymbol>> extractedSymbols;
  for (StringRef symbolLine : llvm::reverse(symbolLines)) {
    SmallVector<StringRef> items;
    symbolLine.split(items, ' ');
    StringRef kind = items[0];
    StringRef name = items[1];
    if (!seenVariables.insert(name).second)
      continue;

    if (kind == "declaration") {
      extractedSymbols.emplace_back(
          std::make_unique<ExtractedPythonDecl>(name));
    } else if (kind == "import") {
      // Private import aliases (starting with a leading underscore) should not
      // be exposed to mojo.
      if (name.starts_with("_"))
        continue;

      StringRef module = items[2];
      extractedSymbols.emplace_back(
          std::make_unique<ExtractedPythonImport>(name, module));
    }
  }

  return extractedSymbols;
}
