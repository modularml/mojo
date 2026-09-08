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

#include "Mojo/Support/Configuration.h"
#include "Support/Configuration.h"
#include "llvm/Support/FileSystem.h"
#include <variant> // IWYU pragma: keep (std::visit)

using namespace M;
using namespace M::KGEN;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MOJO_CONFIG(path) _X_STRINGIFY(MOJO_CONFIG_SECTION) path

#ifndef MOJO_CONFIG_SECTION // NOLINT(ifdef), Wundef doesn't handle the define
#error "Expected MOJO_CONFIG_SECTION to be set"
#endif

ErrorOr<std::filesystem::path> MojoConfig::getConfigFilePath() const {
  if (const Config *val = std::get_if<Config>(&configSource))
    return val->getConfigFilePath();
  return Error("Configuration file path unavailable from settings");
}

static Config &getConfigFrom(Config &config) { return config; }

static Config &getConfigFrom(Config *config) { return *config; }

Config &MojoConfig::getConfig() {
  return std::visit([](auto &src) -> Config & { return getConfigFrom(src); },
                    configSource);
}

StringRef MojoConfig::getValue(StringLiteral key) {
  return getConfig().getValue(key);
}

StringRef MojoConfig::getPath(StringLiteral key, StringRef relativePath) {
  return getConfig().getPath(key, relativePath);
}

//===----------------------------------------------------------------------===//
// MojoConfig
//===----------------------------------------------------------------------===//

ErrorOr<MojoConfig> MojoConfig::open() {
  ErrorOr<Config> config = Config::open();
  if (config.isError())
    return config.takeError();
  return MojoConfig(std::move(*config));
}

MojoConfig MojoConfig::fromContext(ContextRef ctx) {
  return MojoConfig(ctx->get<Config>());
}

//===----------------------------------------------------------------------===//
// Parser Configurations
//===----------------------------------------------------------------------===//

void MojoConfig::getParserImportPaths(SmallVectorImpl<StringRef> &paths) {
  StringRef importPaths = getValue(STRINGIFY_MOJO_CONFIG(".import_path"));
  importPaths.split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}

//===----------------------------------------------------------------------===//
// Plugin Configurations
//===----------------------------------------------------------------------===//

SmallVector<std::string> MojoConfig::getPluginPaths() {
  SmallVector<StringRef> paths;
  StringRef pluginPaths = getValue(STRINGIFY_MOJO_CONFIG(".mojo_plugin_paths"));
  pluginPaths.split(paths, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  return SmallVector<std::string>(paths.begin(), paths.end());
}

//===----------------------------------------------------------------------===//
// LLDB Configurations
//===----------------------------------------------------------------------===//

#ifdef __APPLE__
#define EXT ".dylib"
#else
#define EXT ".so"
#endif

StringRef MojoConfig::getLLDBPluginPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".lldb_plugin_path"),
                 "lib/libMojoLLDB" EXT);
}

StringRef MojoConfig::getLLDBPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".lldb_path"), "bin/mojo-lldb");
}

//===----------------------------------------------------------------------===//
// JIT Configurations
//===----------------------------------------------------------------------===//

StringRef MojoConfig::getCompilerRTPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".compilerrt_path"),
                 "lib/libKGENCompilerRTShared" EXT);
}

StringRef MojoConfig::getMGPRTPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".mgprt_path"), "lib/libMGPRT" EXT);
}

//===----------------------------------------------------------------------===//
// Driver Configurations
//===----------------------------------------------------------------------===//

StringRef MojoConfig::getDriverPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".driver_path"), "bin/mojo");
}

StringRef MojoConfig::getJupyterPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".jupyter_path"),
                 "lib/libMojoJupyter" EXT);
}

StringRef MojoConfig::getLSPServerPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".lsp_server_path"),
                 "bin/mojo-lsp-server");
}

StringRef MojoConfig::getMBlackPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".mblack_path"), "bin/mblack");
}

StringRef MojoConfig::getREPLEntryPoint() {
  return getPath(STRINGIFY_MOJO_CONFIG(".repl_entry_point"),
                 "lib/mojo-repl-entry-point");
}

StringRef MojoConfig::getLinkerDriver() {
  return getValue(STRINGIFY_MOJO_CONFIG(".linker_driver"));
}

StringRef MojoConfig::getLLDPath() {
  return getPath(STRINGIFY_MOJO_CONFIG(".lld_path"), "bin/lld");
}

void MojoConfig::setLLDPathOverride(StringRef path) {
  Config::setGlobalValue(STRINGIFY_MOJO_CONFIG(".lld_path"), path);
}

void MojoConfig::appendSystemLibraryLinkArgs(SmallVectorImpl<StringRef> &libs) {
  if (auto maybeSystemLibsArg =
          getConfig().maybeGetValue(STRINGIFY_MOJO_CONFIG(".system_libs"))) {
    maybeSystemLibsArg.value().split(libs, ',', /*MaxSplit=*/-1,
                                     /*KeepEmpty=*/false);
  }
}

void MojoConfig::appendSharedLibraryLinkArgs(SmallVectorImpl<StringRef> &args) {
  StringRef sharedLibsArg = getValue(STRINGIFY_MOJO_CONFIG(".shared_libs"));
  if (!sharedLibsArg.empty()) {
    sharedLibsArg.split(args, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  } else {
    // Mini-hack: We make up some imaginary config sections so that the config
    // will intern some strings for us. Otherwise, we'd have to intern the whole
    // string and then parse it back out.
    args.push_back("-Xlinker");
    args.push_back("-rpath");
    args.push_back("-Xlinker");
    args.push_back(getPath(STRINGIFY_MOJO_CONFIG(".shared_libs_lib"), "lib"));
  }

  // The AsyncRT Mojo bindings ship in max-core, not in the mojo compiler
  // package, so a base Mojo install simply does not have them. Link them
  // whenever they are installed: a `.mojoc` records no cc dependencies, so
  // this is the only way `mojo build` can learn that an imported package
  // (`max`) needs them. Bazel builds resolve the same library through the
  // dependency graph instead and never reach this.
  StringRef bindings = getPath(STRINGIFY_MOJO_CONFIG(".shared_libs_artmb"),
                               "lib/libAsyncRTMojoBindings" EXT);
  if (llvm::sys::fs::exists(bindings))
    args.push_back(bindings);
}

StringRef MojoConfig::getMojoConfigSection() {
  return _X_STRINGIFY(MOJO_CONFIG_SECTION);
}
