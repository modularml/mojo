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

#include "Mojo/LITDialect/LITInterfaces.h"
#include "Mojo/LITDialect/LITOps.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_STRIPPARSERMETADATA
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct StripParserMetadataPass
    : public impl::StripParserMetadataBase<StripParserMetadataPass> {
  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      // Strip doc strings from ASTDecl operations.
      if (auto astDecl = dyn_cast<LIT::ASTDeclInterface>(op))
        astDecl.removeDocStringAttr();
    });
  }
};
} // namespace
