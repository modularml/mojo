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

#ifndef KGEN_HLCFDIALECT_HLCFINTERFACES_H
#define KGEN_HLCFDIALECT_HLCFINTERFACES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// Interface Verifiers
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class FunctionLike;
} // namespace M::KGEN

namespace M::HLCF {
class ControlFlowNode;
class ControlFlowTerminator;

LogicalResult verifyControlFlowNode(ControlFlowNode op);
LogicalResult verifyControlFlowTerminator(ControlFlowTerminator op);
LogicalResult verifyControlFlow(KGEN::FunctionLike root);
} // namespace M::HLCF

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::HLCF {
struct ControlFlowTarget {
  ControlFlowTarget(std::optional<unsigned> index, ValueRange inputs = {})
      : index(index), inputs(inputs) {}

  std::optional<unsigned> index;
  ValueRange inputs;
};
} // namespace M::HLCF

#include "Mojo/HLCFDialect/HLCFInterfaces.h.inc"

#endif // KGEN_HLCFDIALECT_HLCFINTERFACES_H
