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

#ifndef KGEN_TRANSFORMUTILS_MANGLINGUTILS_H
#define KGEN_TRANSFORMUTILS_MANGLINGUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
class GeneratorOpInterface;

/// Returns a simplified serialization of a parameter that is more readable.
/// Eventually this should be used by `mangleParameterValues` (with MOCO-945),
/// but today it does not guarantee unique serialization for type-values that
/// are identical except for the vtable.
void prettyPrintParameter(TypedAttr value, raw_ostream &os);

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
std::string mangleParameterValues(GeneratorOpInterface generator,
                                  ArrayRef<TypedAttr> inputParamValues);
} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_MANGLINGUTILS_H
