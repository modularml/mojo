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

#ifndef POPINTERFACES_TD
#define POPINTERFACES_TD

include "mlir/IR/OpBase.td"

def POP_FastmathFlagsInterface : OpInterface<"FastmathFlagsInterface"> {
  let description = [{
    An operation whose floating-point behavior fastmath flags can relax.
  }];

  let cppNamespace = "::M::KGEN::POP";

  let methods = [
    InterfaceMethod<"The operation's fastmath flags.",
        "::M::KGEN::POP::FastmathFlags", "getFastmathFlags">,
    InterfaceMethod<"Sets the operation's fastmath flags.", "void",
        "setFastmathFlags", (ins "::M::KGEN::POP::FastmathFlags":$flags)>
  ];

  let extraClassDeclaration = [{
    /// Whether `fast` is set, allowing a cheaper result than the exact one.
    bool isFast() {
      return bitEnumContainsAll(getFastmathFlags(), FastmathFlags::fast);
    }
  }];
}

#endif // POPINTERFACES_TD
