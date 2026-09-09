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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"
#include "nanobind/stl/filesystem.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/typing.h"

namespace nb = nanobind;

namespace M::Graph::Python {

// Returning mlir::Operation* from nanobind is annoying and error prone for a
// few reasons.
//  - Pointer values are assumed to be passing ownership to Python, while
//  mlir::Operation* pointers are used as non-owning references
//  - Operation types don't actually subclass mlir::Operation! They actually
//  subclass mlir::OpState, but in C++ they may be cast via `mlir::dyn_cast`.
//  - In order for nanobind to recognize and downcast the op automatically it
//  needs to follow the correct inheritance. This means bound methods should
//  always return OpState objects rather than Operation pointers.
//  - There's no simple way to get an OpState from an Operation*, only the
//  other way around. This implementation relies on the memory layout of
//  OpState to cast directly.
inline nb::handle_t<mlir::OpState> castOp(mlir::Operation *op) {
  return nb::handle_t<mlir::OpState>(
      nb::cast(std::make_unique<mlir::OpState>(*(mlir::OpState *)&op))
          .release());
}

} // namespace M::Graph::Python
