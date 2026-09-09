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

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DebugInfoStrip
//===----------------------------------------------------------------------===//

def DebugInfoStrip : Pass<"debuginfo-strip"> {
  let summary = "Strip debug information for the current IR.";
  let description = [{
    This pass strips all debuginfo dialect information from the current IR, but
    leaves normal location information in place (to still allow proper
    verification and error reporting).
  }];
  let options = [
    Option<"preserveLineTables", "preserveLineTables", "bool",
           /*default=*/"false",
           "Don't strip information used for line table generation.">,
  ];
}
