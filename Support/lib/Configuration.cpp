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

#include "Support/Configuration.h"
#include "Support/BazelRunfiles.h"
#include "Support/CacheLog.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/Globals/Globals.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace M;

/// Folder type for searching.
namespace {
enum class FolderType { Config, Data, Cache };

// Runtime overrides set via Config::setGlobalValue() are stored in
// `libMSupportGlobals.so` so that writes from one shared library (e.g.
// the `max._core` Python extension that wraps `DebugConfig`) are visible
// to reads from another (e.g. `libmax.so` which contains
// `GraphCompiler/FrameworkFrontend`). A function-local static here would
// give each consumer shared library its own copy and break propagation.
std::mutex &globalOverridesMutex() {
  return M::Globals::getConfigOverridesMutex();
}
llvm::StringMap<std::string> &globalOverrides() {
  return M::Globals::getConfigOverrides();
}
} // namespace

static ErrorOrSuccess createPath(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec) {
    // The directory did not exist, and we failed to create it.
    return Error(Twine(path.string()) +
                 " could not be created: " + ec.message());
  }
  return success();
}

ErrorOr<Config> Config::open() {
  // Parse MODULAR_DEBUG once so `max-debug.*` overrides are visible to every
  // subsequent `maybeGetValue()` call — including from standalone Mojo
  // binaries that never load the Python bindings' `DebugConfig`.
  parseModularDebugEnv();

  // Cache the parsed config data to avoid repeated disk I/O.
  // We use std::call_once to ensure thread-safe one-time initialization.
  static std::once_flag initFlag;
  static llvm::StringMap<std::string> cachedKV;
  static std::optional<Error> initError;

  std::call_once(initFlag, [] {
    auto configFilePathOr = getConfigFilePath(/*create=*/false);

    // If we don't have a config, then that's not an error! Simply return an
    // empty config. An error is returned above only if the directory cannot be
    // created.
    if (configFilePathOr.isError())
      return; // Empty cachedKV is fine

    std::error_code ec;
    if (!std::filesystem::exists(*configFilePathOr, ec)) {
      if (ec) {
        initError = Error(ec.message());
        return;
      }
      return; // No file means empty config
    }

    // Set up variables we'll need to get this read.
    Config cfg;
    llvm::SourceMgr sourceMgr;
    unsigned bufferIdx = 0;

    // Check the permissions for the directory containing the configuration. If
    // it's not writeable, then we avoid acquiring the lock.
    if (llvm::sys::fs::access(configFilePathOr->parent_path().string(),
                              llvm::sys::fs::AccessMode::Write)) {
      // We don't have write permission here, so we can just read it without a
      // lock.
      auto mBufOr = llvm::MemoryBuffer::getFile(configFilePathOr->string(),
                                                /*IsText=*/true);
      if (!mBufOr) {
        initError = Error(mBufOr.getError().message());
        return;
      }

      bufferIdx =
          sourceMgr.AddNewSourceBuffer(std::move(*mBufOr), llvm::SMLoc());
    } else {
      std::optional<Error> error = std::nullopt;
      // Read the file atomically - we may have multiple processes writing.
      ErrorOrSuccess err = readFileUnderLock(
          *configFilePathOr, [&](const std::filesystem::path &filePath) {
            auto mBufOr =
                llvm::MemoryBuffer::getFile(filePath.string(), /*IsText=*/true);
            if (!mBufOr) {
              error = Error(mBufOr.getError().message());
              return;
            }

            bufferIdx =
                sourceMgr.AddNewSourceBuffer(std::move(*mBufOr), llvm::SMLoc());
          });
      // Check for errors.
      if (err.isError()) {
        initError = err.takeError();
        return;
      }
      if (error.has_value()) {
        initError = std::move(*error);
        return;
      }
    }

    // Grab the memory buffer and parse from it.
    const llvm::MemoryBuffer *mbuf = sourceMgr.getMemoryBuffer(bufferIdx);
    if (ErrorOrSuccess err = cfg.parseFrom(mbuf->getBuffer(), &sourceMgr)) {
      initError = err.takeError();
      return;
    }

    // Cache the parsed key-value map for future calls.
    cachedKV = std::move(cfg.kv);
  });

  // If initialization failed, return the error.
  if (initError.has_value())
    return initError->copy();

  // Return a fresh Config initialized from the cached data.
  return Config(cachedKV);
}

ErrorOrSuccess Config::parseFrom(StringRef buffer, llvm::SourceMgr *mgr) {
  auto emitError = [&](llvm::SMLoc loc, const Twine &msg) -> Error {
    if (!mgr)
      return {msg};

    std::string errMsg;
    llvm::raw_string_ostream stream(errMsg);
    mgr->PrintMessage(stream, loc, llvm::SourceMgr::DK_Error, msg);

    return {errMsg};
  };

  auto takeLine = [&buffer]() -> StringRef {
    size_t newlineLoc = buffer.find_first_of("\n\r\f\v");
    size_t toDrop;
    if (newlineLoc == StringRef::npos) {
      newlineLoc = buffer.size();
      toDrop = newlineLoc;
    } else {
      toDrop = newlineLoc + 1;
    }
    auto line = buffer.take_front(newlineLoc);
    buffer = buffer.drop_front(toDrop);
    return line;
  };

  // While the current pointer is inside the buffer, parse.
  std::string currentSection;
  while (!buffer.empty()) {
    StringRef tmp = takeLine();

    // If there's nothing but whitespace in it, continue.
    if (tmp.trim().empty())
      continue;

    // Parse a section delimiter.
    if (tmp.consume_front("[")) {
      tmp = tmp.take_until([](char c) { return c == ']'; }).trim();
      currentSection = tmp;
      continue;
    }

    // If there's a comment at the end of this line, drop it and parse
    // anything in front of it.
    tmp = tmp.take_until([](char c) { return c == '#' || c == ';'; });

    // Again, if it's empty, drop this line.
    if (tmp.trim().empty())
      continue;

    // Split on the equals sign.
    auto [k, v] = tmp.split('=');
    if (v.empty()) {
      return emitError(llvm::SMLoc::getFromPointer(k.begin()),
                       "malformed line: expected `key = value`");
    }

    // Allow global properties - anything not in a section. The way this works
    // is by not prefixing with the current section.
    std::string currentSectionPrefix =
        (!currentSection.empty() ? (currentSection + ".") : "");

    // Insert the key and value into the current map, trimming off any extra
    // whitespace. We insert each value under the key `section.key` to make
    // lookups fast.
    std::string constructedKey = (currentSectionPrefix + k.trim()).str();

    // Insert under the lowercase name - section names and property names
    // are case-insensitive. This will also take a copy of the value string
    // so that we don't have to deal with keeping the buffers alive.
    kv[StringRef(constructedKey).lower()] = v.trim();
  }
  return success();
}

StringRef Config::getValue(StringRef key) {
  if (auto maybeValue = maybeGetValue(key))
    return maybeValue.value();
  else
    return "";
}

std::optional<StringRef> Config::maybeGetValue(StringRef key) {
  // Highest priority: runtime overrides set via setGlobalValue().  Cached
  // into the instance's kv so the returned StringRef stays valid for the
  // lifetime of this Config.
  {
    std::lock_guard<std::mutex> lock(globalOverridesMutex());
    auto &overrides = globalOverrides();
    if (auto it = overrides.find(key.lower()); it != overrides.end()) {
      kv[key.lower()] = it->second;
      return kv[key.lower()];
    }
  }

  std::string upper = key.upper();
  std::replace_if(
      upper.begin(), upper.end(),
      [](char c) { return (c == '.') || (c == '-'); }, '_');

  // Check for this environment variable.
  auto envOr = llvm::sys::Process::GetEnv("MODULAR_" + upper);
  // If we have this env variable, save it in the map. We don't care if it
  // overrides something.
  if (envOr) {
    kv[key.lower()] = *envOr;
  } else if (auto runfilesPath = findConfigWithRunfiles(key)) {
    kv[key.lower()] = *runfilesPath;
  }

  if (auto iter = kv.find(key.lower()); iter != kv.end())
    return iter->second;
  else
    return std::nullopt;
}

StringRef Config::getValueOr(StringRef key, StringRef defaultValue) {
  StringRef stringValue = getValue(key);
  if (stringValue.empty())
    return defaultValue;
  return stringValue;
}

StringRef Config::getPath(StringRef key, StringRef relativePath) {
  const std::string keyStr{key.lower()};
  StringRef stringValue = getValue(keyStr);
  if (!stringValue.empty())
    return stringValue;

  const auto [section, _] = key.split('.');
  StringRef packageRoot = getValue((section + ".package_root").str());
  std::string &value = kv[keyStr];
  value = (packageRoot + "/" + relativePath).str();
  return value;
}

bool Config::getValueAsBool(StringRef key, bool defaultValue) {
  auto stringValue = getValue(key);
  if (stringValue.empty())
    return defaultValue;
  return llvm::StringSwitch<bool>(stringValue)
      .CasesLower({"0", "false", "no"}, false)
      .CasesLower({"1", "true", "yes"}, true)
      .Default(defaultValue);
}

void Config::getValueAsList(StringRef key, SmallVectorImpl<StringRef> &values,
                            StringRef sep) {
  StringRef value = getValue(key);
  if (!value.empty())
    value.split(values, sep);
}

bool Config::isValueInList(StringRef key, StringRef value, StringRef sep) {
  SmallVector<StringRef, 16> values;
  getValueAsList(key, values, sep);
  return std::find(values.begin(), values.end(), value) != values.end();
}

void Config::setValue(StringRef key, StringRef value) {
  kv[key.lower()] = value;
}

void Config::setGlobalValue(StringRef key, StringRef value) {
  std::lock_guard<std::mutex> lock(globalOverridesMutex());
  globalOverrides()[key.lower()] = value.str();
}

void Config::unsetGlobalValue(StringRef key) {
  std::lock_guard<std::mutex> lock(globalOverridesMutex());
  globalOverrides().erase(key.lower());
}

std::optional<std::string> Config::getGlobalValueIfSet(StringRef key) {
  std::lock_guard<std::mutex> lock(globalOverridesMutex());
  auto &overrides = globalOverrides();
  if (auto it = overrides.find(key.lower()); it != overrides.end())
    return it->second;
  return std::nullopt;
}

void Config::parseModularDebugEnv() {
  static std::once_flag parseFlag;
  std::call_once(parseFlag, [] {
    const char *env = std::getenv("MODULAR_DEBUG");
    if (!env || env[0] == '\0')
      return;

    // The meta `sensible` token expands to a curated debug bundle; keep
    // this list in sync with `DebugConfig::setSensibleMode` in the Python
    // bindings.
    auto applySensibleMode = [] {
      setGlobalValue("max-debug.sensible-mode", "true");
      setGlobalValue("max-debug.nan-check", "true");
      setGlobalValue("max-debug.assert-level", "all");
      setGlobalValue("max-debug.device-sync-mode", "true");
      setGlobalValue("max-debug.stack-trace-on-error", "true");
      setGlobalValue("max-debug.stack-trace-on-crash", "true");
      setGlobalValue("max-debug.source-tracebacks", "true");
    };

    llvm::StringRef envRef(env);
    while (!envRef.empty()) {
      // Split on comma.
      auto [tokenRef, rest] = envRef.split(',');
      envRef = rest;

      // Trim whitespace.
      tokenRef = tokenRef.trim(" \t");
      if (tokenRef.empty())
        continue;

      // key=value form routes to a Config value override.
      size_t eq = tokenRef.find('=');
      if (eq != llvm::StringRef::npos) {
        llvm::StringRef key = tokenRef.substr(0, eq);
        llvm::StringRef value = tokenRef.substr(eq + 1);
        if (key == "assert-level" || key == "op-log-level" ||
            key == "print-style" || key == "ir-output-dir") {
          setGlobalValue(("max-debug." + key).str(), value.str());
        } else {
          llvm::errs() << "MODULAR_DEBUG: unknown option '" << key
                       << "'; ignoring\n";
        }
        continue;
      }

      // Boolean flags and the `sensible` meta mode.
      if (tokenRef == "sensible") {
        applySensibleMode();
      } else if (llvm::is_contained({"nan-check", "uninitialized-read-check",
                                     "device-sync-mode", "stack-trace-on-error",
                                     "stack-trace-on-crash",
                                     "source-tracebacks"},
                                    tokenRef)) {
        setGlobalValue(("max-debug." + tokenRef).str(), "true");
      } else {
        llvm::errs() << "MODULAR_DEBUG: unknown option '" << tokenRef
                     << "'; ignoring\n";
      }
    }
  });
}

/// Get the list of search paths, in order of preference.
static void getSearchPaths(SmallVectorImpl<std::filesystem::path> &paths,
                           FolderType type) {
  // If MODULAR_HOME is defined, use that and only that.
  auto modularHome = llvm::sys::Process::GetEnv("MODULAR_HOME");
  MODULAR_CACHE_LOG("config")
      << "getSearchPaths: MODULAR_HOME="
      << (modularHome ? *modularHome : "<unset>") << "\n";
  if (modularHome) {
    // Cache folder is a subdirectory in this case.
    if (type == FolderType::Cache) {
      paths.push_back(std::filesystem::path(*modularHome) / "cache");
      return;
    }
    paths.push_back(*modularHome);
    return;
  }

  // If MODULAR_DERIVED_PATH is defined, use that and only that.
  auto derivedPath = llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH");
  MODULAR_CACHE_LOG("config")
      << "getSearchPaths: MODULAR_DERIVED_PATH="
      << (derivedPath ? *derivedPath : "<unset>") << "\n";
  if (derivedPath) {
    // Cache folder is a subdirectory in this case.
    if (type == FolderType::Cache) {
      paths.push_back(std::filesystem::path(*derivedPath) / "cache");
      return;
    }
    paths.push_back(*derivedPath);
    return;
  }

  // To work well in test environments, check for a standardized test
  // environment variable. This is always the last option, if available.
  auto testTempdir = llvm::sys::Process::GetEnv("TEST_TMPDIR");
  MODULAR_CACHE_LOG("config")
      << "getSearchPaths: TEST_TMPDIR="
      << (testTempdir ? *testTempdir : "<unset>") << "\n";
  if (testTempdir) {
    paths.push_back(std::filesystem::path(*testTempdir) / ".modular");
    return;
  }

#ifndef _WIN32
  // To support existing installs, add $HOME/.modular if it exists. If it
  // does it always takes precedence.
  auto homeDir = llvm::sys::Process::GetEnv("HOME");
  bool addedHome = false;
  if (homeDir) {
    auto path = std::filesystem::path(*homeDir) / ".modular";
    // Use the non-throwing overload: if HOME is not traversable by the running
    // UID (e.g., running in a container where the image was built for a
    // different UID), `stat` on this path returns EACCES, and the throwing
    // overload would propagate `std::filesystem::filesystem_error` to
    // `std::terminate`. Treat any error as "path is not usable, skip it".
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
      if (type == FolderType::Cache)
        paths.push_back(path / ".cache" / "modular");
      else
        paths.push_back(std::filesystem::path(*homeDir) / ".modular");
      paths.push_back(path);
      addedHome = true;
    }
  }

  // Follow the XDG spec https://specifications.freedesktop.org/basedir-spec
  auto xdgConfigHome = llvm::sys::Process::GetEnv("XDG_CONFIG_HOME");
  auto xdgConfigData = llvm::sys::Process::GetEnv("XDG_DATA_HOME");
  auto xdgConfigCache = llvm::sys::Process::GetEnv("XDG_CACHE_HOME");
  if (homeDir) {
    if (!xdgConfigHome)
      xdgConfigHome = std::filesystem::path(*homeDir) / ".config";
    if (!xdgConfigData)
      xdgConfigData = std::filesystem::path(*homeDir) / ".local" / "share";
    if (!xdgConfigCache)
      xdgConfigCache = std::filesystem::path(*homeDir) / ".cache";
  }
  switch (type) {
  case FolderType::Config:
    if (xdgConfigHome)
      paths.push_back(std::filesystem::path(*xdgConfigHome) / "modular");
    break;
  case FolderType::Data:
    if (xdgConfigData)
      paths.push_back(std::filesystem::path(*xdgConfigData) / "modular");
    break;
  case FolderType::Cache:
    if (xdgConfigCache)
      paths.push_back(std::filesystem::path(*xdgConfigCache) / "modular");
    break;
  }

  // Lastly if we haven't added $HOME/.modular in first step, add it now.
  if (!addedHome && homeDir) {
    if (type == FolderType::Cache)
      paths.push_back(std::filesystem::path(*homeDir) / ".modular" / "cache");
    else
      paths.push_back(std::filesystem::path(*homeDir) / ".modular");
  }

  // Add /opt/modular as a global destination.
  paths.push_back("/opt/modular");
#else  // _WIN32
  // Add $APPDATA\Local\Modular
  auto defaultRoot = llvm::sys::Process::GetEnv("APPDATA");
  assert(defaultRoot.has_value() && "Must have APPDATA");
  paths.push_back(std::filesystem::path(*defaultRoot) / "Local" / "Modular");
#endif // _WIN32

  if (isCacheLogEnabled()) {
    MODULAR_CACHE_LOG("config")
        << "getSearchPaths: " << paths.size() << " candidate paths\n";
    for (const auto &p : paths)
      MODULAR_CACHE_LOG("config") << "  " << p.string() << "\n";
  }
}

static ErrorOr<std::filesystem::path> findBestPathForType(FolderType type,
                                                          bool create) {
  // Get the list of search paths.
  SmallVector<std::filesystem::path, 3> searchPaths;
  getSearchPaths(searchPaths, type);

  // Check each of the search paths for existence - if none of them exist
  // return the first of the paths as MODULAR_HOME. An error (e.g. EACCES when
  // a parent directory of the candidate is not traversable by the running UID)
  // is treated as "not found" so we quietly move on to the next candidate.
  auto found =
      llvm::find_if(searchPaths, [&](const std::filesystem::path &path) {
        std::error_code ec;
        return std::filesystem::exists(path, ec);
      });
  if (found != searchPaths.end()) {
    MODULAR_CACHE_LOG("config")
        << "findBestPathForType: found existing: " << found->string() << "\n";
    return *found;
  }

  // If we aren't supposed to create the directory, then just return the path
  // directly. It is still our "best choice", even if we can't use it. The
  // caller must specifically set create=false to exercise this path.
  if (!create)
    return searchPaths[0];

  // None of the above directories exist. Attempt to create a directory, in the
  // order provided. We iterate and return the first one we can create.
  Error firstErr = Error("no candidates for directory");
  bool firstErrFound = false;
  found = llvm::find_if(searchPaths, [&](const std::filesystem::path &path) {
    auto err = createPath(path);
    if (err.isError()) {
      if (!firstErrFound) {
        firstErrFound = true;
        firstErr = err.takeError();
      }
      return false;
    }
    return true;
  });
  if (found != searchPaths.end()) {
    MODULAR_CACHE_LOG("config")
        << "findBestPathForType: created new: " << found->string() << "\n";
    return *found;
  }

  // Nothing could be created. Return the first error encountered (which is the
  // directory we'd want to use with the highest priority).
  MODULAR_CACHE_LOG("config") << "findBestPathForType: all candidates failed\n";
  return firstErr;
}

ErrorOr<std::filesystem::path> Config::getModularConfigFolderPath(bool create) {
  return findBestPathForType(FolderType::Config, create);
}

ErrorOr<std::filesystem::path> Config::getModularDataFolderPath(bool create) {
  return findBestPathForType(FolderType::Data, create);
}

ErrorOr<std::filesystem::path> Config::getModularCacheFolderPath(bool create) {
  auto cacheDir = getValue("cache_dir");
  if (!cacheDir.empty()) {
    MODULAR_CACHE_LOG("config")
        << "getModularCacheFolderPath: explicit cache_dir=" << cacheDir << "\n";
    return std::filesystem::path(cacheDir.str());
  }

  auto defaultCacheDir = findBestPathForType(FolderType::Cache, create);
  if (defaultCacheDir.isError())
    return defaultCacheDir.takeError();
  MODULAR_CACHE_LOG("config")
      << "getModularCacheFolderPath: default path=" << defaultCacheDir->string()
      << "\n";
  return defaultCacheDir.takeValue();
}

ErrorOr<std::filesystem::path> Config::getConfigFilePath(bool create) {
  constexpr llvm::StringLiteral kModularConfigFileName = "modular.cfg";

  // If we found the config file this way, then return it.
  auto configFile = findModularFile(kModularConfigFileName);
  if (configFile)
    return *configFile;

  // Otherwise, return where it should be placed.
  auto configFolderOr = getModularConfigFolderPath(create);
  if (configFolderOr.isError())
    return configFolderOr.takeError();
  return *configFolderOr / kModularConfigFileName.str();
}

std::optional<std::filesystem::path> M::findModularFile(StringRef fileName) {
  SmallVector<std::filesystem::path, 4> searchPaths;
  getSearchPaths(searchPaths, FolderType::Config);
#ifndef _WIN32
  // Append a path to the search paths on UNIX systems.
  searchPaths.push_back(std::filesystem::path("/etc/modular"));
#endif // _WIN32

#ifdef __APPLE__
  // Homebrew installs into /opt/homebrew on arm64 and we symlink our /etc
  // package files into HOMEBREW_PREFIX/etc/modular location.
  searchPaths.push_back(std::filesystem::path("/opt/homebrew/etc/modular"));
#endif // __APPLE__

  // Try to find the file in the provided paths. Treat any error (e.g. EACCES
  // when HOME is not traversable by the running UID) as "not found" and move
  // on to the next candidate rather than aborting.
  auto found =
      llvm::find_if(searchPaths, [&](const std::filesystem::path &path) {
        std::error_code ec;
        return std::filesystem::exists(path / fileName.str(), ec);
      });

  // Was not found, return nullopt.
  if (found == searchPaths.end())
    return std::nullopt;

  // We did find it, return that path.
  return *found / fileName.str();
}

#if defined(__APPLE__)
static constexpr llvm::StringLiteral kSharedLibExt = ".dylib";
#else
static constexpr llvm::StringLiteral kSharedLibExt = ".so";
#endif

bool M::isMaxInstalled() {
  // A runfiles-provided library (Bazel) counts as installed.
  if (findConfigWithRunfiles("max.lib_path").has_value())
    return true;

  ErrorOr<Config> configOr = Config::open();
  if (configOr.isError()) {
    // probably won't happen, return false anyways
    return false;
  }

  std::optional<StringRef> packageRoot =
      configOr->maybeGetValue("max.package_root");
  if (packageRoot.has_value()) {
    std::filesystem::path libmaxPath =
        std::filesystem::path(packageRoot.value().str()) / "lib" /
        ("libmax" + kSharedLibExt.str());
    bool exists = std::filesystem::exists(libmaxPath);
    // Breadcrumb for CI flakes (MXF-560): unconditional since a flake can't
    // opt in after the fact.
    if (!exists) {
      llvm::errs() << "isMaxInstalled: max.package_root resolved to "
                   << packageRoot.value() << " but " << libmaxPath.string()
                   << " does not exist\n";
    }
    return exists;
  } else {
    // No value, so probably in bazel, pretend we have MAX.
    return true;
  }
}
