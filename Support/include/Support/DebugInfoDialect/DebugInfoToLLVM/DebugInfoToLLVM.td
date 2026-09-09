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

def DebugInfoToLLVM : Pass<"convert-debuginfo-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the `debuginfo` dialect to the `llvm` dialect.";
  let description = [{
    This pass lowers Debug Info dialect attributes, operations, and types to
    the LLVM dialect.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"tradeoffPerfForVariableDI", "tradeoff-perf", "bool", "false",
           "If true, pass will attempt to maximally preserve debuginfo for "
           "variables at the expense of runtime performance.">,
  ];
}
