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

#ifndef HLCFDIALECT_TD
#define HLCFDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// HLCFDialect
//===----------------------------------------------------------------------===//

def HLCFDialect : Dialect {
  let name = "hlcf";

  let summary = "high-level control-flow dialect";
  let description = [{
    The `HLCF` dialect implements a high-level control-flow paradigm that models
    control-flow across multiple operations and regions in the IR. The HLCF
    dialect aims to provide a structured and scoped representation of program
    control-flow with the same expressibility as standard control-flow graphs
    (CFGs). The HLCF dialect provides a model closer to source-level programs
    than pure structured control-flow and CFGs.

    The HLCF dialect contains scoped control-flow operations and control-flow
    terminators. Any terminator may be used in any control-flow operation,
    pursuant to constraints in the current control-flow tree.

    A "control-flow tree" is a collection of nested control-flow scopes, bound
    from top to the leaves by non-HLCF regions. Control-flow must be well-formed
    within each tree.

    Example:

    ```mlir
    // This body of this loop forms the root of a tree with no children.
    hlcf.loop {

      // A nested non-HLCF region ends the current tree.
      "foo.region"() ({

        // Both regions of this if operation are roots of control-flow trees.
        hlcf.if %arg1 {

          // The body of this loop is a leaf of the current tree.
          hlcf.loop {
            hlcf.continue
          }

          // Both regions of this if operation are leaves of the current tree.
          hlcf.if %arg1 {
            hlcf.yield
          } else {
            hlcf.yield
          }

          hlcf.yield
        } else {
          hlcf.yield
        }

        "foo.terminator"() : () -> ()
      }) : () -> ()

      hlcf.continue
    }
    ```

    In this code snippet, there are three control-flow trees, one rooted at the
    body top-level `hlcf.loop` and two rooted at the regions of the first
    `hlcf.if`. These two trees are completely isolated with respect to
    control-flow.

    Control-flow is validated from the top of each tree. For example, the
    following IR is invalid:

    ```mlir
    hlcf.loop {
      "foo.region"() ({
        hlcf.if %arg0 {
          // No surrounding loop scope.
          hlcf.break
        } else {
          hlcf.yield
        }

        "foo.terminator"() : () -> ()
      }) : () -> ()
    }
    ```

    The nested `hlcf.break` terminator has no surrounding loop scope because it
    is contained in a tree rooted at `hlcf.if`.

    There is special logic for functions, specifically operations that implement
    `FunctionOpInterface`. HLCF regions nested directly underneath a function
    are children of a control-flow tree rooted at the function. `hlcf.return`
    operations within that tree transfer control-flow to the end of the
    function.

    ```mlir
    func.func @some_function(%arg0: i1) {
      hlcf.loop {
        hlcf.if %arg0 {
          // Valid return. Current region is a child of a tree rooted at a
          // function operation.
          hlcf.return
        } else {
          hlcf.yield
        }
        "foo.region"() ({
          hlcf.if %arg0 {
            // Invalid return. Current region is not the child of a tree rooted
            // at a function operation.
            hlcf.return
          } else {
            hlcf.yield
          }
        }) : () -> ()
      }
    }
    ```
  }];

  let cppNamespace = "::M::HLCF";

  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Register all HLCF attributes.
    void registerAttributes();
  }];
  let hasConstantMaterializer = 1;
}

#endif // HLCFDIALECT_TD
