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

#ifndef INTERPRETERDIALECT_TD
#define INTERPRETERDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// InterpreterDialect
//===----------------------------------------------------------------------===//

def InterpreterDialect : Dialect {
  let name = "interp";

  let summary = "interpreter dialect";
  let description = [{
    The `interp` dialect implements attributes and utilities for
    communicating with the interpreter in MLIR.
  }];

  let cppNamespace = "::M";

  let extraClassDeclaration = [{
    /// Register all dialect attributes.
    void registerAttributes();
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif // INTERPRETERDIALECT_TD
