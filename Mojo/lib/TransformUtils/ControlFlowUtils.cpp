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

#include "Mojo/TransformUtils/ControlFlowUtils.h"
#include "Mojo/HLCFDialect/HLCFInterfaces.h"

using namespace M;
using namespace KGEN;

bool KGEN::userCrossesFunctionCFG(
    Operation *op, Operation *user,
    SmallVectorImpl<Operation *> *enclosingNodes) {
  for (Operation *cur = user->getParentOp(), *parent = op->getParentOp();
       cur != parent; cur = cur->getParentOp()) {
    // If there is any non-control-flow operation between the user and the
    // operation, then the user crosses an unknown region.
    if (!isa<HLCF::ControlFlowNode>(cur))
      return true;
    if (enclosingNodes)
      enclosingNodes->push_back(cur);
  }
  return false;
}
