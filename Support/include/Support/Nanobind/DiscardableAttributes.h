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

#ifndef SUPPORT_NANOBIND_DISCARDABLEATTRIBUTES_H
#define SUPPORT_NANOBIND_DISCARDABLEATTRIBUTES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace M::Graph::Python {

struct DiscardableAttributes {
  mlir::Operation *op;
  DiscardableAttributes(mlir::Operation *op) : op(op) {}
};

} // namespace M::Graph::Python

#endif // SUPPORT_NANOBIND_DISCARDABLEATTRIBUTES_H
