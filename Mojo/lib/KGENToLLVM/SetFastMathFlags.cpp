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

#include "Mojo/ToolCommon/KGENPasses.h"

#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"

using namespace mlir;

namespace M::KGEN {
#define GEN_PASS_DEF_SETFASTMATHFLAGS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

struct SetFastMathFlagsPass
    : public M::KGEN::impl::SetFastMathFlagsBase<SetFastMathFlagsPass> {
  using SetFastMathFlagsBase::SetFastMathFlagsBase;

  void runOnOperation() override {
    using M::KGEN::POP::FastmathFlags;
    using M::KGEN::POP::FastmathFlagsInterface;

    // Only clearing `contract` changes anything.
    if (contract)
      return;

    getOperation()->walk([](FastmathFlagsInterface op) {
      op.setFastmathFlags(op.getFastmathFlags() & ~FastmathFlags::contract);
    });
  }
};

} // namespace
