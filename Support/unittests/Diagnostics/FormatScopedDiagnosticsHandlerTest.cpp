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

#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"

#include "gtest/gtest.h"

using namespace M;
using namespace mlir;

TEST(FormatScopedDiagnosticHandler, FormatsErrorMessage) {
  MLIRContext context;
  FormatScopedDiagnosticHandler diagnostics(&context);

  mlir::emitError(mlir::FileLineColLoc::get(&context, "foo.cpp", 42, 12),
                  "It was a bright cold day in April");

  EXPECT_STREQ("foo.cpp:42:12: error: It was a bright cold day in April\n",
               diagnostics.formatMessage().c_str());
}

TEST(FormatScopedDiagnosticHandler, FormatsWarningMessage) {
  MLIRContext context;
  FormatScopedDiagnosticHandler diagnostics(&context);

  mlir::emitWarning(mlir::FileLineColLoc::get(&context, "foo.cpp", 42, 12),
                    "It was a bright cold day in April");

  EXPECT_STREQ("foo.cpp:42:12: warning: It was a bright cold day in April\n",
               diagnostics.formatMessage().c_str());
}

TEST(FormatScopedDiagnosticHandler, FormatsRemarkMessage) {
  MLIRContext context;
  FormatScopedDiagnosticHandler diagnostics(&context);

  mlir::emitRemark(mlir::FileLineColLoc::get(&context, "foo.cpp", 42, 12),
                   "It was a bright cold day in April");

  EXPECT_STREQ("foo.cpp:42:12: remark: It was a bright cold day in April\n",
               diagnostics.formatMessage().c_str());
}

TEST(FormatScopedDiagnosticHandler, IgnoresCallSiteLocation) {
  MLIRContext context;
  FormatScopedDiagnosticHandler diagnostics(&context);

  mlir::emitRemark(mlir::CallSiteLoc::get(
                       mlir::FileLineColLoc::get(&context, "foo.cpp", 42, 12),
                       mlir::FileLineColLoc::get(&context, "foo.cpp", 14, 8)),
                   "It was a bright cold day in April");

  EXPECT_STREQ("remark: It was a bright cold day in April\n",
               diagnostics.formatMessage().c_str());
}

TEST(FormatScopedDiagnosticHandler, FormatsAttachedNotes) {
  MLIRContext context;
  FormatScopedDiagnosticHandler diagnostics(&context);

  InFlightDiagnostic inFlightDiagnostic =
      mlir::emitRemark(mlir::FileLineColLoc::get(&context, "foo.cpp", 42, 12),
                       "It was a bright cold day in April");
  inFlightDiagnostic.attachNote() << "and the clocks were striking thirteen";
  context.getDiagEngine().emit(
      std::move(*inFlightDiagnostic.getUnderlyingDiagnostic()));

  EXPECT_STREQ("foo.cpp:42:12: remark: It was a bright cold day in April\n  "
               "foo.cpp:42:12: note: and the clocks were striking thirteen\n",
               diagnostics.formatMessage().c_str());
}

TEST(FormatScopedDiagnosticHandler, EmitsFirstLineToMinimalStream) {
  MLIRContext context;

  auto diagnostic =
      mlir::emitRemark(mlir::FileLineColLoc::get(&context, "foo.cpp", 42, 12),
                       "It was a bright cold day in April\nand the clocks were "
                       "striking thirteen");

  std::string minimalOutput;
  llvm::raw_string_ostream minimalStream(minimalOutput);
  llvm::raw_null_ostream nullSteam;
  FormatScopedDiagnosticHandler::emitDiagnosticToStream(
      minimalStream, nullSteam, *diagnostic.getUnderlyingDiagnostic());

  EXPECT_STREQ("foo.cpp:42:12: remark: It was a bright cold day in April "
               "(additional lines (37 bytes) elided)\n",
               minimalOutput.c_str());
}

TEST(FormatScopedDiagnosticHandler, EmitsFirstAndSecondLineToAdditionalStream) {
  MLIRContext context;

  auto diagnostic =
      mlir::emitRemark(mlir::FileLineColLoc::get(&context, "foo.cpp", 42, 12),
                       "It was a bright cold day in April\nand the clocks were "
                       "striking thirteen");

  llvm::raw_null_ostream nullSteam;
  std::string fullOutput;
  llvm::raw_string_ostream fullStream(fullOutput);
  FormatScopedDiagnosticHandler::emitDiagnosticToStream(
      nullSteam, fullStream, *diagnostic.getUnderlyingDiagnostic());

  EXPECT_STREQ("foo.cpp:42:12: remark: It was a bright cold day in April\n"
               "and the clocks were striking thirteen\n",
               fullOutput.c_str());
}
