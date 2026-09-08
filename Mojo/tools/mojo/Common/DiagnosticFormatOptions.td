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
// This file defines a diagnostic format option as part of the "common options"
// group, such that it appears in the same group as `--help`.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICFORMATOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICFORMATOPTIONS_TD

include "Support/Driver/CommonOptions.td"

defm diagnostic_format : CommandOption<["--", "-"], "diagnostic-format",
  CommonOptionGroup, Index<0>, MetaVarName<"FORMAT">,
  HelpText<"The format in which diagnostics and error messages are printed. "
           "Must be one of \"text\" or \"json\" (\"text\" is the default).">>;

#endif // KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICFORMATOPTIONS_TD
