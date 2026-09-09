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

#include "Support/URI.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

// The code in this file is copied and adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/clangd/URI.h

using namespace M;

namespace {

bool isValidScheme(llvm::StringRef scheme) {
  if (scheme.empty())
    return false;
  if (!llvm::isAlpha(scheme[0]))
    return false;
  return llvm::all_of(llvm::drop_begin(scheme), [](char C) {
    return llvm::isAlnum(C) || C == '+' || C == '.' || C == '-';
  });
}

} // namespace

ErrorOr<URI> URI::parse(llvm::StringRef uri) {
  URI u;
  auto pos = uri.find(':');
  if (pos == llvm::StringRef::npos || (pos == 1 && llvm::isAlpha(uri[0]))) {
    // This is not a URI, or scheme is a single letter (which can happen on
    // Windows with filesystem paths), so we assume it is a filesystem path.
    u.scheme = "file";
  } else {
    u.scheme = uri.substr(0, pos);
    if (!isValidScheme(u.scheme)) {
      return Error("Invalid scheme: " + u.scheme);
    }
    uri = uri.substr(pos + 1);
    if (uri.consume_front("//")) {
      pos = uri.find('/');
      u.authority = uri.substr(0, pos);
      uri = uri.substr(pos);
    }
  }
  u.path = uri;
  return u;
}
