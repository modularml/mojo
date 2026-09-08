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

#ifndef KGEN_NAMEMANGLING_H
#define KGEN_NAMEMANGLING_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Name Mangling
//===----------------------------------------------------------------------===//

/// Many backends don't support arbitrary symbol names. This function will
/// losslessly re-mangle a symbol to only alnum characters and underscores.
/// The mangling scheme will replace all unsupported characters with underscores
/// and then append characters to the end of the symbol to keep it unique, if
/// name length is longer than charToKeep it truncates the sanitized name and
/// append _hash_hex(name) to the end making it at most 64 character long.
StringAttr sanitizeSymbolToAlnum(StringAttr name, size_t charToKeep = 32);

/// Like sanitizeSymbolToAlnum but replaces every run of invalid characters with
/// a single '_' without appending their encoded forms. This produces cleaner
/// PTX names when the source string uses separator characters (e.g. dots) that
/// are meaningful to humans but irrelevant after sanitisation. Long-name
/// hashing and digit-start fixup behave identically to sanitizeSymbolToAlnum.
StringAttr sanitizeSymbolToUnderscores(StringAttr name, size_t charToKeep = 32);

/// Append a uniqueness suffix "_XXXXXXXX" (8 hex chars of xxh3_64) to
/// @p userName, hashing @p symName and @p funcTypeStr together.
///
/// \param userName     The sanitized @__name prefix string.
/// \param symName      The auto-mangled wrapper symbol name, used as a hash
///                     input to disambiguate instantiations that share a
///                     prefix.
/// \param funcTypeStr  The printed MLIR function type, used as a hash input to
///                     disambiguate closures capturing different types.
StringAttr appendAutoMangledSuffix(StringAttr userName, StringRef symName,
                                   StringRef funcTypeStr);

} // namespace M::KGEN

#endif // KGEN_NAMEMANGLING_H
