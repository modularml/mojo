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

#ifndef KGEN_SUPPORT_CONFIGURATION_H
#define KGEN_SUPPORT_CONFIGURATION_H

#include "Support/Configuration.h"
#include "Support/Context.h"
#include "Support/ErrorOr.h"

#include <filesystem>
#include <variant>

namespace M {
class Config;
} // namespace M

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// MojoConfig
//===----------------------------------------------------------------------===//

/// This class provides easy and type-safe access to values in the Mojo section
/// of the modular configuration file.
class MojoConfig {
public:
  /// Returns the path to the modular.cfg file.
  ErrorOr<std::filesystem::path> getConfigFilePath() const;

  /// Open the default configuration, and parse it.
  static ErrorOr<MojoConfig> open();

  /// Open the configuration using settings from the provided context.
  static MojoConfig fromContext(ContextRef ctx);

  //===--------------------------------------------------------------------===//
  // Parser Configurations
  //===--------------------------------------------------------------------===//

  /// Return the default Mojo parser import paths.
  void getParserImportPaths(SmallVectorImpl<StringRef> &paths);

  //===--------------------------------------------------------------------===//
  // Plugin Configurations
  //===--------------------------------------------------------------------===//

  /// Return the list of plugin file paths to load, parsed from a
  /// semicolon-separated config value.
  SmallVector<std::string> getPluginPaths();

  //===--------------------------------------------------------------------===//
  // LLDB Configurations
  //===--------------------------------------------------------------------===//

  /// Return the path to the lldb-vscode executable within the mojo install.
  StringRef getLLDBVSCodePath();

  /// Return the path to the Mojo lldb plugin within the mojo install.
  StringRef getLLDBPluginPath();

  /// Return the path to the lldb executable within the mojo install.
  StringRef getLLDBPath();

  /// Return the default lldb visualizers to use when starting an LLDB debug
  /// session.
  void getLLDBVisualizers(SmallVectorImpl<StringRef> &paths);

  //===--------------------------------------------------------------------===//
  // JIT Configurations
  //===--------------------------------------------------------------------===//

  /// Return the path to the kgen-compiler-rt library.
  StringRef getCompilerRTPath();

  /// Return the path to the mgp-rt library.
  StringRef getMGPRTPath();

  //===--------------------------------------------------------------------===//
  // Driver Configurations
  //===--------------------------------------------------------------------===//

  /// Return the path to the `mojo` driver in the mojo install.
  StringRef getDriverPath();

  /// Return the path to the Mojo jupyter library.
  StringRef getJupyterPath();

  /// Return the path to the Mojo LSP server.
  StringRef getLSPServerPath();

  /// Return the path to the mblack executable in the mojo install.
  StringRef getMBlackPath();

  /// Return the path to the REPL entry point executable in the mojo install.
  StringRef getREPLEntryPoint();

  /// Return the path to the linker driver to use when linking mojo executables.
  StringRef getLinkerDriver();

  /// Returns the path to lld that should be used for linking shared libraries.
  StringRef getLLDPath();

  /// Sets a process-wide override for the lld path used when linking,
  /// equivalent to setting the `MODULAR_MOJO_MAX_LLD_PATH` environment
  /// variable. The override takes precedence over the environment variable and
  /// the `mojo-max.lld_path` configuration value.
  static void setLLDPathOverride(StringRef path);

  /// Appends the system libraries to link with Mojo when building a standalone
  /// binary.
  void appendSystemLibraryLinkArgs(SmallVectorImpl<StringRef> &libs);

  /// Appends the shared library arguments to link with Mojo when building a
  /// standalone binary.
  void appendSharedLibraryLinkArgs(SmallVectorImpl<StringRef> &args);

  /// Return the section used for this mojo build.
  StringRef getMojoConfigSection();

private:
  MojoConfig(Config config) : configSource(std::move(config)) {}
  MojoConfig(Config *settings) : configSource(settings) {}

  Config &getConfig();
  StringRef getValue(StringLiteral key);
  StringRef getPath(StringLiteral key, StringRef relativePath);

  // This is a little silly, but currently it's used to represent owned vs
  // shared config.
  std::variant<Config, Config *> configSource;
};
} // namespace M::KGEN

#endif // KGEN_SUPPORT_CONFIGURATION_H
