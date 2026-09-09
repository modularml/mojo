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

#include "Support/Compiler/PassTimingUtils.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/LogicalResult.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <system_error>
#include <utility>

namespace {

//===----------------------------------------------------------------------===//
// OutputStrategy
//===----------------------------------------------------------------------===//

/// This class represents an OutputStrategy which we can use to structure the
/// JSON string that is produced when running MLIR passes with timing enabled.
class OutputJsonStrategy : public mlir::OutputStrategy {
private:
  std::unique_ptr<llvm::raw_fd_ostream> ostream;
  llvm::json::OStream jsonStream;
  llvm::StringRef pipelineName;
  uint32_t depth{0};

public:
  OutputJsonStrategy(std::unique_ptr<llvm::raw_fd_ostream> os,
                     llvm::StringRef pipelineName)
      : OutputStrategy(*os), ostream(std::move(os)), jsonStream(*ostream),
        pipelineName(pipelineName) {}

  void printHeader(const mlir::TimeRecord &total) override {
    jsonStream.objectBegin();
    jsonStream.attribute("pipeline_name", pipelineName);
    jsonStream.attributeBegin("data");
    jsonStream.arrayBegin();
  }

  void printFooter() override {
    jsonStream.arrayEnd();     // data
    jsonStream.attributeEnd(); // data
    jsonStream.objectEnd();
  }

  void printTime(const mlir::TimeRecord &time,
                 const mlir::TimeRecord &total) override {
    jsonStream.attributeBegin("wall");
    jsonStream.object([&] {
      jsonStream.attributeBegin("duration");
      jsonStream.rawValueBegin() << llvm::format("%.4f", time.wall);
      jsonStream.rawValueEnd();
      jsonStream.attributeEnd();

      jsonStream.attributeBegin("percentage");
      jsonStream.rawValueBegin()
          << llvm::format("%.1f", 100.0 * time.wall / total.wall);
      jsonStream.rawValueEnd();
      jsonStream.attributeEnd();
    });
    jsonStream.attributeEnd(); // wall
  }

  void printListEntry(llvm::StringRef name, const mlir::TimeRecord &time,
                      const mlir::TimeRecord &total, bool lastEntry) override {
    jsonStream.object([&] {
      jsonStream.attribute("name", name);
      printTime(time, total);
      jsonStream.attributeArray("passes", [] {});
    });
  }

  void printTreeEntry(unsigned indent, llvm::StringRef name,
                      const mlir::TimeRecord &time,
                      const mlir::TimeRecord &total) override {
    depth = indent;
    jsonStream.objectBegin();
    jsonStream.attribute("name", name);
    printTime(time, total);
    jsonStream.attributeBegin("passes");
    jsonStream.arrayBegin();
  }

  void printTreeEntryEnd(unsigned indent, bool lastEntry) override {
    depth = indent;
    jsonStream.arrayEnd();     // passes
    jsonStream.attributeEnd(); // passes
    jsonStream.objectEnd();
  }
};

} // namespace

M::ErrorOrSuccess
M::configureMLIRPassTimingJSONOutput(mlir::PassManager &pm,
                                     llvm::StringRef outDir,
                                     llvm::StringRef passPipelineName) {
  auto tm = std::make_unique<mlir::DefaultTimingManager>();
  tm->setEnabled(true);
  tm->setDisplayMode(mlir::DefaultTimingManager::DisplayMode::Tree);

  auto tempFileOr = TempFile::create(
      (std::filesystem::path(outDir.str()) / "pass-timing-%%%%%%.json")
          .string());
  if (tempFileOr.isError()) {
    return tempFileOr.takeError();
  }

  auto &tempFile = tempFileOr.get();
  tempFile.keep();

  std::error_code errorCode;
  std::unique_ptr<llvm::raw_fd_ostream> ostream =
      std::make_unique<llvm::raw_fd_ostream>(tempFile.getPath().string(),
                                             errorCode);
  if (errorCode) {
    return Error("Unable to open file: " + tempFile.getPath().string() +
                 ", reason: " + errorCode.message());
  }

  tm->setOutput(std::make_unique<OutputJsonStrategy>(std::move(ostream),
                                                     passPipelineName));
  pm.enableTiming(std::move(tm));
  return M::success();
}
