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

#include "Support/ErrorOr.h"
#include "Support/Error.h"
#include "Support/LogicalResult.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

using namespace M;

// TODO(akirchhoff): These tests are fairly minimal.  More would be better.

TEST(ErrorOr, successful) {
  ErrorOr<int> eo(5);
  EXPECT_FALSE(eo.isError());
  EXPECT_FALSE(eo);
  EXPECT_TRUE(LogicalResult(eo).succeeded());
  EXPECT_EQ(5, eo.get());
  EXPECT_EQ(5, *eo);
  EXPECT_EQ(nullptr, eo.getError());
}

TEST(ErrorOr, erroneous) {
  ErrorOr<int> eo(Error("Toaster is broken"));
  EXPECT_TRUE(eo.isError());
  EXPECT_TRUE(eo);
  EXPECT_FALSE(LogicalResult(eo).succeeded());
  EXPECT_STREQ("Toaster is broken", eo.getError());
}

TEST(ErrorOr, erroneousTwine) {
  ErrorOr<int> eo(Error(llvm::Twine("Toaster is broken")));
  EXPECT_TRUE(eo.isError());
  EXPECT_TRUE(eo);
  EXPECT_FALSE(LogicalResult(eo).succeeded());
  EXPECT_STREQ("Toaster is broken", eo.getError());
}

TEST(ErrorOr, equality) {
  EXPECT_EQ(ErrorOr<int>(5), ErrorOr<int>(5));
  EXPECT_NE(ErrorOr<int>(5), ErrorOr<int>(10));
  EXPECT_EQ(ErrorOr<int>(Error("File not found")),
            ErrorOr<int>(Error("File not found")));
  EXPECT_NE(ErrorOr<int>(Error("File not found")),
            ErrorOr<int>(Error("Network connection lost")));
  EXPECT_NE(ErrorOr<int>(5), ErrorOr<int>(Error("File not found")));
}

TEST(ErrorOr, referenceType) {
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  ErrorOr<std::vector<int> &> v1OrErr = v1;
  EXPECT_EQ(v1OrErr, ErrorOr<std::vector<int> &>(v1));
  EXPECT_NE(v1OrErr, ErrorOr<std::vector<int> &>(v2));
  EXPECT_EQ(std::vector<int>({1, 2, 3}), v1OrErr.get());
  EXPECT_EQ(std::vector<int>({1, 2, 3}), *v1OrErr);
}

// TODO(akirchhoff): Test move semantics
// TODO(akirchhoff): Test copying

TEST(ErrorOr, fromLLVMError) {
  llvm::Error llvmError =
      llvm::createStringError(std::error_code(), "Toaster overheated");
  EXPECT_TRUE(!!llvmError);
  ErrorOrSuccess modularError = toModularErrorOr(std::move(llvmError));
  EXPECT_TRUE(modularError);
  EXPECT_STREQ("Toaster overheated", modularError.getError());
}

TEST(ErrorOr, fromLLVMErrorEmpty) {
  llvm::Error llvmError = llvm::Error::success();
  EXPECT_FALSE(!!llvmError);
  ErrorOrSuccess modularError = toModularErrorOr(std::move(llvmError));
  EXPECT_FALSE(modularError);
}

TEST(ErrorOr, fromLLVMExpectedOK) {
  llvm::Expected<int> llvmExpected(5);
  EXPECT_TRUE(!!llvmExpected);
  ErrorOr<int> modularErrorOr = toModularErrorOr(std::move(llvmExpected));
  EXPECT_FALSE(modularErrorOr);
  EXPECT_EQ(5, *modularErrorOr);
}

TEST(ErrorOr, fromLLVMExpectedError) {
  llvm::Expected<int> llvmExpected(
      llvm::createStringError(std::error_code(), "Toaster overheated"));
  EXPECT_FALSE(!!llvmExpected);
  ErrorOr<int> modularErrorOr = toModularErrorOr(std::move(llvmExpected));
  EXPECT_TRUE(modularErrorOr);
  EXPECT_STREQ("Toaster overheated", modularErrorOr.getError());
}

TEST(ErrorOr, fromLLVMErrorOrOK) {
  llvm::ErrorOr<int> llvmErrorOr(5);
  EXPECT_TRUE(!!llvmErrorOr);
  ErrorOr<int> modularErrorOr = toModularErrorOr(std::move(llvmErrorOr));
  EXPECT_FALSE(modularErrorOr);
  EXPECT_EQ(5, *modularErrorOr);
}

namespace {
class ToasterErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "Toaster"; }
  std::string message(int code) const override { return "Toaster overheated"; }
};
} // namespace

TEST(ErrorOr, fromLLVMErrorOrError) {
  static ToasterErrorCategory toasterCategory;
  llvm::ErrorOr<int> llvmErrorOr(std::error_code(0, toasterCategory));
  EXPECT_FALSE(!!llvmErrorOr);
  ErrorOr<int> modularErrorOr = toModularErrorOr(std::move(llvmErrorOr));
  EXPECT_TRUE(modularErrorOr);
  EXPECT_STREQ("Toaster overheated", modularErrorOr.getError());
}
