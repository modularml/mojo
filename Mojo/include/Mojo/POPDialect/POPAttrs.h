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

#ifndef KGEN_POPDIALECT_POPATTRS_H
#define KGEN_POPDIALECT_POPATTRS_H

#include "Mojo/KGENDialect/KGENAttrInterfaces.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/POPDialect/POPEnums.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/IPInt.h"
#include "Support/IPRational.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Mojo/POPDialect/POPAttrs.h.inc"

#endif // GEN_POPDIALECT_POPATTRS_H
