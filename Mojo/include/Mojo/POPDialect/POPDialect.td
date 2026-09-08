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

#ifndef KGEN_POPDIALECT_POPDIALECT_TD
#define KGEN_POPDIALECT_POPDIALECT_TD

include "mlir/IR/OpBase.td"

def POPDialect : Dialect {
  let name = "pop";

  let summary = "The 'pop' dialect";
  let description = [{
    This dialect defines the `pop` dialect, which defines parametric operations
    on top of the LLVM dialect.
  }];

  let cppNamespace = "::M::KGEN::POP";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    /// Register all POP attributes.
    void registerAttributes();
    /// Register all POP types.
    void registerTypes();
  }];
}

#endif // KGEN_POPDIALECT_POPDIALECT_TD
