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

#include "Support/CommonCLOptions.h"
#include "Support/Configuration.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

using namespace M;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MAX_CONFIG(path) _X_STRINGIFY(MAX_CONFIG_SECTION) path

#ifndef MAX_CONFIG_SECTION // NOLINT(ifdef), Wundef doesn't handle the define
#error "Expected MAX_CONFIG_SECTION to be set"
#endif

/// Helper function that creates a directory, finds a unique filename (adding
/// numerical suffix if needed), and opens the file for writing.
static std::unique_ptr<llvm::ToolOutputFile>
openUniqueFile(StringRef directory, StringRef baseName, StringRef extension) {
  // Create the temps directory if it doesn't exist.
  std::error_code errorCode;
  std::filesystem::create_directories(directory.str(), errorCode);
  if (errorCode)
    return nullptr;

  // Build the output file path.
  std::string outFile = (std::filesystem::path(directory.str()) /
                         (baseName.str() + extension.str()))
                            .string();

  auto absoluteOutputFile = std::filesystem::absolute(outFile, errorCode);
  if (errorCode)
    return nullptr;

  // Get a unique filename by adding numerical suffix if file exists.
  std::filesystem::path uniquePath = absoluteOutputFile;
  std::string stem = uniquePath.stem().string();
  std::string ext = uniquePath.extension().string();
  int suffix = 1;

  // Only try up to 999 suffixes before falling back to overwriting.
  while (std::filesystem::exists(uniquePath) && suffix <= 999) {
    uniquePath =
        uniquePath.parent_path() / (stem + "_" + std::to_string(suffix) + ext);
    ++suffix;
  }

  llvm::outs() << "Emitting intermediate file to '" << uniquePath.string()
               << "'.\n";

  std::string errorMessage;
  return mlir::openOutputFile(uniquePath.string(), &errorMessage);
}

std::unique_ptr<llvm::ToolOutputFile>
CommonOptions::getOutputFile(bool hasBinaryOutput,
                             StringRef fileExtension) const {
  // We generally listen to the `-o filename` command, unless we're being
  // asked to emit a binary file format to the console.  In that case, we
  // default to emitting a variant of the input filename.
  std::string outFile = outputFilename;
  if (hasBinaryOutput && inputFilename != "-" && outFile.empty()) {
    outFile = inputFilename + fileExtension.str();
    llvm::outs() << "Emitting binary file to " << outFile << ".\n";
  }

  // Create the output directory if the directory doesn't exist and the output
  // file *is* a file.
  std::error_code ec;
  if (outFile != "-" && !std::filesystem::exists(outFile, ec) && !ec) {
    auto outFilePath = std::filesystem::path(outFile);
    if (outFilePath.has_parent_path())
      std::filesystem::create_directories(outFilePath.parent_path(), ec);
  }
  // If anything failed, report the failure.
  if (ec)
    exit(reportError("std::filesystem: " + ec.message() + ": " + outFile));

  std::error_code error;
  auto result = std::make_unique<llvm::ToolOutputFile>(outFile, error,
                                                       llvm::sys::fs::OF_None);
  if (error)
    exit(reportError("Cannot open output file: '" + outFile +
                     "': " + error.message()));

  return result;
}

std::unique_ptr<llvm::ToolOutputFile>
CommonOptions::getIntermediateFile(StringRef inputName, StringRef ext) const {
  if (!saveTemps)
    return nullptr;

  std::filesystem::path inputPath(inputName.str());
  std::string directory;
  std::string baseName;

  if (!tempsDir.empty()) {
    // Use the provided temps directory with just the stem of the filename.
    directory = tempsDir;
    baseName = inputPath.stem().string();
  } else {
    // Use the input file's directory (or current dir) with the full filename.
    directory =
        inputPath.has_parent_path() ? inputPath.parent_path().string() : ".";
    baseName = inputPath.filename().string();
  }

  auto result = openUniqueFile(directory, baseName, ext);
  if (!result)
    exit(reportError("Failed to open intermediate file"));
  return result;
}

LogicalResult CommonOptions::emitArchive(StringRef object) const {
  std::unique_ptr<llvm::ToolOutputFile> outFile =
      getOutputFile(/*hasBinaryOutput=*/true);
  if (!outFile)
    return failure(reportError("failed to open the output file"));

  outFile->os().write(object.begin(), object.size());
  outFile->keep();

  return mlir::success();
}

std::unique_ptr<llvm::ToolOutputFile>
M::openIntermediateTextFile(StringRef baseName, StringRef extension) {
  auto configOr = Config::open();
  if (configOr.isError())
    return nullptr;

  // Prefer the user-facing debug option; fall back to the legacy save-temps
  // key (which is also where `MODULAR_MAX_TEMPS_DIR` lands). This mirrors the
  // precedence in `M::getMaxIRDumpDir()` so that `MODULAR_DEBUG=ir-output-dir`
  // emits intermediate text (e.g. the `.mojo` dump) alongside the MLIR files.
  StringRef dir = configOr->getValue("max-debug.ir-output-dir");
  if (dir.empty())
    dir = configOr->getValue(STRINGIFY_MAX_CONFIG(".temps_dir"));
  if (dir.empty())
    return nullptr;

  return openUniqueFile(dir, baseName, extension);
}
