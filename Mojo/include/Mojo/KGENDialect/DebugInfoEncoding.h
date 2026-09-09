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
// This file declares logic for how KGEN entities are encoded in DebugInfo,
// providing shared utilities between the compiler and debugger.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_DEBUGINFOENCODING_H
#define KGEN_DEBUGINFOENCODING_H

#include "Mojo/KGENDialect/KGENDType.h"

namespace M::KGEN::DebugInfoEncoding {

//===----------------------------------------------------------------------===//
// KGENDType Native Encoding
//===----------------------------------------------------------------------===//

/// Get the fully qualified name used for a KGENDType in debuginfo.
std::string getKGENDTypeAsString(KGENDType dtype);
/// Get a KGENDType from a fully qualified name in debuginfo.
FailureOr<KGENDType> getKGENDTypeFromString(StringRef str);

//===----------------------------------------------------------------------===//
// KGENDType C++ Encoding
//===----------------------------------------------------------------------===//

/// Get the equivalent C++ type name for a KGENDType if supported by C++.
/// Otherwise returns nullopt.
std::optional<std::string> getKGENDTypeAsCppString(KGENDType dtype);

} // namespace M::KGEN::DebugInfoEncoding

#endif // KGEN_DEBUGINFOENCODING_H
