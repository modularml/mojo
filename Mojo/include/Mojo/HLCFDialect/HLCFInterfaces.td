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

#ifndef HLCFINTERFACES_TD
#define HLCFINTERFACES_TD

include "mlir/IR/OpBase.td"

def ControlFlowNode : OpInterface<"ControlFlowNode"> {
  let description = [{
    This interface represents an operation whose regions form nodes in an HLCF
    control-flow tree.
  }];
  let cppNamespace = "::M::HLCF";

  let methods = [
    InterfaceMethod<[{
      Given potential constant values of the operands of this operation, return
      the indices of the entry region of the operation, which is the region to
      the beginning of which control-flow branches upon visiting the start of
      this operation, and the operands with which to branch to that region.
      Return `None` to indicate that control-flow branches directly to after the
      operation.
    }], "void", "getEntryTargets",
      (ins "ArrayRef<Attribute>":$operands,
           "SmallVectorImpl<HLCF::ControlFlowTarget> &":$targets)>,
    InterfaceMethod<[{
      Return the range of values that should be mapped onto incoming values. By
      default, this returns
    }], "ValueRange", "getEntryArguments",
      (ins "std::optional<unsigned>":$target)>,
    InterfaceMethod<[{
      Insert variants as operation operands.
    }], "void", "insertVariants",
      (ins "ValueRange":$newOperands), [{}], [{
        $_op.getOperation()->insertOperands(
            $_op.getOperation()->getNumOperands(), newOperands);
    }]>,
    InterfaceMethod<[{
       Add block arguments to the region.
    }], "BlockArgument", "insertArgumentToRegion",
      (ins "Location":$loc, "Type":$argType, "size_t":$argIdx,
           "Region &":$region), [{}], [{
        return region.addArgument(argType, loc);
    }]>
  ];

  let verify = [{ return verifyControlFlowNode(cast<ControlFlowNode>($_op)); }];
}

def ControlFlowTerminator : OpInterface<"ControlFlowTerminator"> {
  let description = [{
    This interface represents a terminator to an HLCF control-flow region. The
    terminator can indicate where it branches to.
  }];
  let cppNamespace = "::M::HLCF";

  let methods = [
    InterfaceMethod<[{
      This method is invoked on the proper ancestors of a control-flow
      terminator to determine the nearest valid parent operation. The method
      should return true if the provided operation is a valid parent operation
      to the terminator, and false to keep searching.
    }], "bool", "isParentNode", (ins "Operation *":$op)>,
    InterfaceMethod<[{
      Return the branch target of the terminator relative to its control-flow
      parent and the operands with which to branch to that region. For instance,
      to branch to the beginning of the first region, the method should return
      `0`. To branch to the after the parent operation, the method should return
      `None`.
    }], "void", "getBranchTargets",
      (ins "ArrayRef<Attribute>":$operands,
           "SmallVectorImpl<HLCF::ControlFlowTarget> &":$targets)>,
    InterfaceMethod<[{
      Insert variants as operation operands.
    }], "void", "insertVariants",
      (ins "ValueRange":$newOperands), [{}], [{
        $_op.getOperation()->insertOperands(
          $_op.getOperation()->getNumOperands(), newOperands);
    }]>
  ];

  let verify = [{
    return verifyControlFlowTerminator(cast<ControlFlowTerminator>($_op));
  }];
}

#endif // HLCFINTERFACES_TD
