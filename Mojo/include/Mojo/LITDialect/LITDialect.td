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
//
// This contains the LITDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITDIALECT_TD
#define KGEN_LITDIALECT_LITDIALECT_TD

include "mlir/IR/DialectBase.td"

def LITDialect : Dialect {
  let name = "lit";

  let summary = "The 'lit' dialect for Mojo🔥";
  let description = [{
    This dialect defines the `lit` (🔥) dialect, a set of language-level
    abstractions that lower to the `kgen` dialect.
  }];

  let cppNamespace = "::M::KGEN::LIT";
  let dependentDialects = [
    "KGENDialect",
    "POP::POPDialect",
    "CO::CODialect",
    "HLCF::HLCFDialect",
    "DebugInfo::DebugInfoDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &p) const override;
    void printType(Type type, DialectAsmPrinter &p) const override;
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // KGEN_LITDIALECT_LITDIALECT_TD
