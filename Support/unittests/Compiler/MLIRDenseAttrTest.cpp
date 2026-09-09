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

#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace M;
using namespace mlir;

TEST(MLIRDenseAttr, createResourceAttr) {
  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  DenseResourceElementsAttr attr;
  {
    // The underlying string is released, but the resource attribute copies it.
    std::string data = "Please pretend this is MLIR bytecode.";

    // Add an additional byte for null terminator.
    attr = createResourceAttr(&ctx, ArrayRef(data.c_str(), data.size() + 1),
                              "This is the name.");
  }
  EXPECT_EQ(attr.getRawHandle().getKey(), "This is the name.");
  EXPECT_EQ(std::string(attr.getRawHandle().getBlob()->getData().data()),
            "Please pretend this is MLIR bytecode.");
}
