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

// Provides a local definition of
// `llvm::format_provider<lldb_private::ConstString>::format` so that
// `libMojoLLDB.dylib` can resolve references emitted by unoptimised builds
// (e.g. `--config=debug-modular`) without statically linking
// `@llvm-project//lldb:Utility`. Pulling that dep in would drag
// `lldb:Host.o` into the plugin's static link set, giving it a duplicate
// copy of LLDB's `HostInfoBase` (including a file-scope `g_fields` pointer
// that is never initialised), which under macOS's two-level namespace
// crashes `mojo repl` at plugin load time. Matches the upstream
// implementation in `lldb/source/Utility/ConstString.cpp`. See MOTO-1573.

#include "lldb/Utility/ConstString.h"
#include "llvm/Support/FormatProviders.h"

void llvm::format_provider<lldb_private::ConstString>::format(
    const lldb_private::ConstString &CS, llvm::raw_ostream &OS,
    llvm::StringRef Options) {
  format_provider<StringRef>::format(CS.GetStringRef(), OS, Options);
}
