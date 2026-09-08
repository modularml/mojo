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

#include "../mojo-lsp-test-client/LSPBatchClient.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace M;
namespace lsp = llvm::lsp;

static ErrorOr<StringRef>
extractMethodString(const llvm::json::Value *methodValue) {
  if (!methodValue)
    return Error(llvm::formatv("missing `method` value"));

  if (auto str = methodValue->getAsString()) {

    return *str;
  } else if (auto object = methodValue->getAsObject()) {
    if (auto method = object->getString("method")) {
      return *method;
    } else {
      return Error(llvm::formatv(
          "malformed `method` value: key `method` is not a string:\n{0}",
          methodValue));
    }
  } else {
    return Error(llvm::formatv(
        "malformed `method` value: expected string or method object:\n{0}",
        methodValue));
  }
}

static void dumpFile(StringRef path) {
  auto bufferOr = openInputFile(path);
  if (failed(bufferOr)) {
    llvm::errs() << llvm::formatv("Unable to read file {0}: {1}\n", path,
                                  bufferOr.getError());
    return;
  }

  llvm::errs() << bufferOr.get()->getBuffer() << "\n";
}

int main(int argc, char **argv) {
  llvm::InitLLVM il(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram x(argc, argv);

  llvm::cl::opt<bool> attachDebugger{
      "attach-debugger",
      llvm::cl::desc("Launch the LSP and start a debug session attached to "
                     "it on VS Code."),
      llvm::cl::init(false),
  };

  llvm::cl::opt<std::string> replayFile{
      "replayFile", llvm::cl::desc("The recorded jsonl file to re-play."),
      llvm::cl::Positional, llvm::cl::Required};

  llvm::cl::opt<std::string> openFile{
      "open",
      llvm::cl::desc(
          "An optional file to open before replaying the recording."),
      llvm::cl::Optional, llvm::cl::value_desc("OPEN")};

  llvm::cl::opt<bool> quiet{
      "quiet", llvm::cl::init(false),
      llvm::cl::desc("Silence output from the language server.")};

  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "This tool replays a recorded sequence of commands from a jsonl file. "
      "Recordings are created using the Mojo Nightly VS Code extension.");

  // We need to preserve the output files, in particular, for later inspection.
  setenv("PRESERVE_LSP_IO_FILES", "1", /*overwrite=*/true);
  // Disabling telemetry improves the language server startup/shutdown time in
  // some conditions.
  setenv("MODULAR_TELEMETRY_ENABLED", "0", /*overwrite=*/true);

  auto bufferOr = openInputFile(replayFile);
  if (failed(bufferOr))
    llvm::report_fatal_error(Twine("Error reading the file ") + replayFile +
                             ": " + bufferOr.getError());
  llvm::MemoryBuffer &buffer = *bufferOr->get();
  StringRef data = buffer.getBuffer();

  LSPBatchClient client(attachDebugger);

  std::unique_ptr<Document> doc;
  if (openFile != "") {
    auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(openFile));
    if (failed(bufferOr))
      llvm::report_fatal_error(Twine("Error reading the file ") + openFile +
                               ": " + bufferOr.getError());
    llvm::MemoryBuffer &buffer = *bufferOr->get();
    doc = std::make_unique<Document>("file://" + openFile, buffer.getBuffer());
    client.open(*doc);
  }

  while (true) {
    auto [line, remainder] = data.split('\n');
    if (line == data && remainder == "")
      break;

    data = remainder;

    if (auto value = llvm::json::parse(line)) {
      if (auto object = value->getAsObject()) {
        auto type = object->getString("type");
        auto method = extractMethodString(object->get("method"));

        if (failed(method)) {
          llvm::errs() << method.getError() << "\n";
        }
        auto param = object->get("param");

        if (!type)
          return 1;

        if (type == "request") {
          client.replayRequest(*method, *param);
        } else if (type == "notification") {
          client.replayNotification(*method, *param);
        }
      } else {
        llvm::errs() << "Malformed JSON: not an object:\n" << line << "\n";
        return 1;
      }
    } else {
      llvm::errs() << "Malformed JSON:\n" << line << "\n";
    }
  }

  auto result = client.execute();

  if (!quiet) {
    llvm::errs() << "Server stderr:\n";
    dumpFile(result.serverIOFiles->serverStderr);
  }

  if (failed(result.err)) {
    llvm::errs() << result.err.getError() << "\n";
    return 1;
  }
}
