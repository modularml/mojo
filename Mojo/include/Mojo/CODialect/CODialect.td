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

#ifndef CODIALECT_TD
#define CODIALECT_TD

include "mlir/IR/DialectBase.td"

def CODialect : Dialect {
  let name = "co";

  let summary = "the coroutine dialect";
  let description = [{
    The `co` dialect defines a parametric opset for working with async functions
    implemented as coroutines with arbitrary suspension points.
  }];

  let cppNamespace = "::M::KGEN::CO";
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = ["KGENDialect"];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerOperations();
  }];
}

#endif // CODIALECT_TD
