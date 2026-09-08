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

#ifndef KGEN_POPDIALECT_POPTYPES_H
#define KGEN_POPDIALECT_POPTYPES_H

#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Pretty Type Parsing and Printing
//===----------------------------------------------------------------------===//

namespace M::KGEN::POP {
/// Try to parse a pretty type or a standard MLIR type. A pretty type is a POP
/// type without the dialect prefix.
ParseResult parsePrettyType(AsmParser &p, TypedAttr &typeExpr);
/// Try to print a pretty type or a standard MLIR type. A pretty type is a POP
/// type without the dialect prefix.
void printPrettyType(AsmPrinter &p, TypedAttr typeExpr);
} // namespace M::KGEN::POP

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Mojo/POPDialect/POPTypes.h.inc"

#endif // GEN_POPDIALECT_POPTYPES_H
