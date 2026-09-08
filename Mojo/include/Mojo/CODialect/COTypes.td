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

#ifndef COTYPES_TD
#define COTYPES_TD

include "Mojo/CODialect/CODialect.td"
include "mlir/IR/AttrTypeBase.td"

include "Mojo/KGENDialect/KGENTypes.td"

class COType<string typeName, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<CODialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// CoroutineType
//===----------------------------------------------------------------------===//

def CoroutineType : COType<"Coroutine", "routine",
    [DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "A coroutine handle.";
  let description = [{
    The `!co.routine` type represents the handle to a coroutine. The
    underlying type is an opaque pointer.

    Example:

    ```mlir
    !co.routine
    ```
  }];

  let assemblyFormat = "";
}

#endif // COTYPES_TD
