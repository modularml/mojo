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

#include "Support/NixlPluginDir.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// Fixtures staged as test data by the BUILD rule, reachable from the runfiles
// root. Each pair carries one SONAME in two builds: the plugin binds a symbol
// only the EFA libfabric exports (FABRIC_1.8), and the libefa it pulls in binds
// one only the EFA-patched libibverbs exports (IBVERBS_PRIVATE_59).
constexpr const char *kEfaLibfabric =
    "Support/unittests/nixl_libfabric_efa_fixture.so";
constexpr const char *kDistroLibfabric =
    "Support/unittests/nixl_libfabric_distro_fixture.so";
constexpr const char *kEfaIbverbs =
    "Support/unittests/nixl_ibverbs_efa_fixture.so";
constexpr const char *kDistroIbverbs =
    "Support/unittests/nixl_ibverbs_distro_fixture.so";
constexpr const char *kLibefa = "Support/unittests/nixl_efa_fixture.so";
constexpr const char *kPlugin = "Support/unittests/libplugin_FIXTURE.so";

// Reproduces the staged package layout: transport plugins in
// <prefix>/lib/nixl/<flavor>, the fabric stack flat in <prefix>/lib. The distro
// copies sit outside the prefix, standing in for /usr/lib/x86_64-linux-gnu.
class StagedLayout {
public:
  explicit StagedLayout(const std::string &name, bool stageFabricLibs = true)
      : prefix_(std::filesystem::temp_directory_path() /
                ("nixl-plugin-dir-" + name)) {
    std::filesystem::remove_all(prefix_);
    std::filesystem::create_directories(pluginDir());
    std::filesystem::create_directories(prefix_ / "distro");
    std::filesystem::copy_file(kPlugin, plugin());
    std::filesystem::copy_file(kDistroLibfabric, distroLibfabric());
    std::filesystem::copy_file(kDistroIbverbs, distroIbverbs());
    if (stageFabricLibs) {
      const std::filesystem::path lib = prefix_ / "lib";
      std::filesystem::copy_file(kEfaLibfabric, lib / "libfabric.so.1");
      std::filesystem::copy_file(kEfaIbverbs, lib / "libibverbs.so.1");
      std::filesystem::copy_file(kLibefa, lib / "libefa.so.1");
    }
  }

  ~StagedLayout() {
    std::error_code ec;
    std::filesystem::remove_all(prefix_, ec);
  }

  std::filesystem::path pluginDir() const {
    return prefix_ / "lib" / "nixl" / "cuda";
  }
  std::filesystem::path plugin() const {
    return pluginDir() / "libplugin_FIXTURE.so";
  }
  std::filesystem::path distroLibfabric() const {
    return prefix_ / "distro" / "libfabric.so.1";
  }
  std::filesystem::path distroIbverbs() const {
    return prefix_ / "distro" / "libibverbs.so.1";
  }
  std::filesystem::path errorFile() const { return prefix_ / "dlerror.txt"; }

private:
  std::filesystem::path prefix_;
};

// Whether the plugin loaded, plus the loader's complaint when it did not.
struct LoadResult {
  bool loaded = false;
  std::string error;
};

// When the staged copies are claimed, relative to the foreign copy arriving.
enum class Claim {
  Never,
  BeforeForeign,
  AfterForeign,
};

// Loads the plugin the way NIXL does, with `foreign` pulled into the process
// the way an unrelated component would. Runs in a forked child because dlopen
// state is process-wide, and which copy claimed the SONAME first is precisely
// what is under test.
LoadResult loadPluginAlongsideForeign(const StagedLayout &layout,
                                      const std::filesystem::path &foreign,
                                      Claim claim) {
  const pid_t pid = fork();
  if (pid == 0) {
    if (claim == Claim::BeforeForeign)
      M::preloadStagedFabricLibs(layout.pluginDir());
    ::dlopen(foreign.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (claim == Claim::AfterForeign)
      M::preloadStagedFabricLibs(layout.pluginDir());
    if (!::dlopen(layout.plugin().c_str(), RTLD_NOW | RTLD_LOCAL)) {
      std::ofstream(layout.errorFile()) << ::dlerror();
      ::_exit(1);
    }
    ::_exit(0);
  }
  int status = 0;
  ::waitpid(pid, &status, 0);

  LoadResult result;
  result.loaded = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  if (std::ifstream err{layout.errorFile()})
    std::getline(err, result.error);
  return result;
}

void requestBackend(const char *backend) {
  setenv("MODULAR_NIXL_TRANSFER_BACKEND", backend, /*overwrite=*/1);
}

void requestNoBackend() { unsetenv("MODULAR_NIXL_TRANSFER_BACKEND"); }

// A flavor directory holding just the plugin filenames the packages stage in
// it. Only the names matter here -- nothing is loaded.
class FlavorDir {
public:
  FlavorDir(const std::string &name,
            std::initializer_list<const char *> plugins)
      : dir_(std::filesystem::temp_directory_path() / ("nixl-flavor-" + name)) {
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
    for (const char *plugin : plugins)
      std::ofstream(dir_ / plugin);
  }

  ~FlavorDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  const std::filesystem::path &path() const { return dir_; }

private:
  std::filesystem::path dir_;
};

// The dual-fabric host: an EFA device for libfabric alongside IB NICs whose
// presence otherwise steers the resolver onto the UCX-only verbs flavor. That
// flavor cannot serve the libfabric request, so it must not be a candidate.
TEST(StagesRequestedBackend, TheVerbsFlavorCannotServeALibfabricRequest) {
  FlavorDir verbs("verbs", {"libplugin_UCX.so"});
  FlavorDir plain("plain", {"libplugin_UCX.so", "libplugin_LIBFABRIC.so"});
  requestBackend("libfabric");

  EXPECT_FALSE(M::stagesRequestedBackend(verbs.path()));
  EXPECT_TRUE(M::stagesRequestedBackend(plain.path()));
}

// UCX is what the KVTransferEngine asks for when nothing says otherwise, so an
// unset backend must not disqualify the UCX-only flavors.
TEST(StagesRequestedBackend, AnUnsetBackendMeansUcx) {
  FlavorDir verbs("unset", {"libplugin_UCX.so"});
  requestNoBackend();

  EXPECT_TRUE(M::stagesRequestedBackend(verbs.path()));
}

TEST(StagesRequestedBackend, AcceptsAnyCasingOfTheBackendRequest) {
  FlavorDir plain("casing-flavor", {"libplugin_LIBFABRIC.so"});
  requestBackend("LibFabric");

  EXPECT_TRUE(M::stagesRequestedBackend(plain.path()));
}

// Asserts the loader's behavior rather than ours: the plugin must fail, and
// fail over `version` specifically, or the test is passing for the wrong
// reason.
void expectVersionMismatch(const LoadResult &result, const char *version) {
  EXPECT_FALSE(result.loaded);
  EXPECT_NE(result.error.find(version), std::string::npos)
      << "expected a " << version << " mismatch, got: " << result.error;
}

// The trap this all exists for: an older libfabric that got into the process
// first owns the SONAME, and the loader never reconsiders the plugin's rpath,
// so the plugin becomes permanently unloadable. If this ever stops failing to
// load, the hazard is gone and preloadStagedFabricLibs can go with it.
TEST(PreloadStagedFabricLibs, AForeignLibfabricLoadedFirstBreaksThePlugin) {
  StagedLayout layout("hijacked");

  expectVersionMismatch(loadPluginAlongsideForeign(
                            layout, layout.distroLibfabric(), Claim::Never),
                        "FABRIC_1.8");
}

// The same trap one level below the plugin's own DT_NEEDED entries: it is
// libefa that needs the EFA-patched libibverbs, and it is only reached when the
// plugin is dlopened -- long after an unrelated component (torch's cuFile RDMA
// plugin) has claimed the SONAME with the distro copy.
TEST(PreloadStagedFabricLibs, AForeignLibibverbsLoadedFirstBreaksThePlugin) {
  StagedLayout layout("hijacked-ibverbs");

  expectVersionMismatch(
      loadPluginAlongsideForeign(layout, layout.distroIbverbs(), Claim::Never),
      "IBVERBS_PRIVATE_59");
}

// ... which claiming the staged copies prevents, because ours get the SONAMEs
// and are a strict superset of the older copies' symbol versions.
TEST(PreloadStagedFabricLibs, ClaimingBeatsAForeignLibfabric) {
  StagedLayout layout("claimed-libfabric");
  requestBackend("libfabric");

  const LoadResult result = loadPluginAlongsideForeign(
      layout, layout.distroLibfabric(), Claim::BeforeForeign);

  EXPECT_TRUE(result.loaded) << "plugin failed to load: " << result.error;
}

TEST(PreloadStagedFabricLibs, ClaimingBeatsAForeignLibibverbs) {
  StagedLayout layout("claimed-ibverbs");
  requestBackend("libfabric");

  const LoadResult result = loadPluginAlongsideForeign(
      layout, layout.distroIbverbs(), Claim::BeforeForeign);

  EXPECT_TRUE(result.loaded) << "plugin failed to load: " << result.error;
}

// Why the claim runs at max._core import and not at transfer-engine
// construction: once the foreign copy owns the SONAME, claiming ours -- even by
// absolute path -- cannot take it back. A late claim is indistinguishable from
// no claim at all, which is what the container's LD_LIBRARY_PATH guards.
TEST(PreloadStagedFabricLibs, ClaimingAfterTheForeignCopyIsTooLate) {
  StagedLayout layout("claimed-late");
  requestBackend("libfabric");

  expectVersionMismatch(loadPluginAlongsideForeign(layout,
                                                   layout.distroIbverbs(),
                                                   Claim::AfterForeign),
                        "IBVERBS_PRIVATE_59");
}

TEST(PreloadStagedFabricLibs, AcceptsAnyCasingOfTheBackendRequest) {
  StagedLayout layout("casing");
  requestBackend("LibFabric");
  EXPECT_TRUE(M::preloadStagedFabricLibs(layout.pluginDir()));
}

// Only the libfabric backend binds the versioned symbols that make the SONAME
// race matter, the EFA libfabric pulls in the CUDA driver stack, and the verbs
// UCX flavors need the host's own rdma-core, whose provider plugins bind the
// matching IBVERBS_PRIVATE version. So no other backend should pay for it.
TEST(PreloadStagedFabricLibs, SkipsBackendsThatDoNotNeedIt) {
  StagedLayout layout("ucx");
  requestBackend("ucx");
  EXPECT_FALSE(M::preloadStagedFabricLibs(layout.pluginDir()));
}

// Layouts with no staged fabric stack (bazel runfiles, ROCm-only packages) are
// ordinary, not errors.
TEST(PreloadStagedFabricLibs, ToleratesAnUnstagedFabricStack) {
  StagedLayout layout("missing", /*stageFabricLibs=*/false);
  requestBackend("libfabric");
  EXPECT_FALSE(M::preloadStagedFabricLibs(layout.pluginDir()));
}

} // namespace
