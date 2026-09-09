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

#ifndef KGEN_COMPILERRT_BINARYID_H
#define KGEN_COMPILERRT_BINARYID_H

#include "Support/SymbolExport.h"
#include <string>

namespace M::KGEN {

MODULAR_CXX_EXPORT std::string getCompilerRTBinaryID();

}

#endif // KGEN_COMPILERRT_BINARYID_H
