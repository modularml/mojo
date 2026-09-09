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

#ifndef MDIALECT_TD
#define MDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

def MDialect : Dialect {
  let name = "M";

  let summary = "M dialect";
  let description = [{
    The `M` dialect contains core types and attributes that depend only on core
    MLIR and which can be shared across all Modular code. Conceptually, this
    dialect is like the `builtin` dialect.

    This dialect also holds a general purpose context that has effective global
    singletons within the Modular stack, such as the runtime.
  }];

  let cppNamespace = "::M";

  let extraClassDeclaration = [{
  public:
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect types.
    void registerTypes();
    /// Inject any external type / attribute models.
    void injectTypeInterfaces();
    void injectAttrInterfaces();

    /// Get the registered context.
    ContextRef getInternal() { return context.copy(); }
    /// Set the registered context.
    void setInternal(ContextRef &&ref) { context = std::move(ref); }
  private:
    ContextRef context;
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // MDIALECT_TD
