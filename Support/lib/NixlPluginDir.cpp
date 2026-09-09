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

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <string>

using namespace M;

// Probes whether a shared library resolves on this host, without keeping it
// loaded. Used to decide between plugin flavors whose load-time dependencies
// differ.
static bool canLoadSharedLib(const char *name) {
  if (void *handle = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL)) {
    ::dlclose(handle);
    return true;
  }
  return false;
}

// The lowercased MODULAR_NIXL_TRANSFER_BACKEND value, or "" if unset. Lets an
// explicit backend request override the AMD default flavor.
static std::string requestedBackend() {
  const char *b = std::getenv("MODULAR_NIXL_TRANSFER_BACKEND");
  if (!b)
    return "";
  std::string s(b);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

bool M::stagesRequestedBackend(const std::filesystem::path &pluginDir) {
  // Upstream plugin filenames are uppercase; the Modular-facing env var is
  // lowercase. An unset backend means ucx, matching the KVTransferEngine
  // default.
  std::string backend = requestedBackend();
  if (backend.empty())
    backend = "ucx";
  std::transform(backend.begin(), backend.end(), backend.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  std::error_code ec;
  return std::filesystem::exists(pluginDir / ("libplugin_" + backend + ".so"),
                                 ec);
}

bool M::preloadStagedFabricLibs(const std::filesystem::path &pluginDir) {
  if (requestedBackend() != "libfabric")
    return false;
  // Leaf-first: libnl carries no versioned symbols of its own but underpins
  // libibverbs, which underpins librdmacm and the libefa verbs library the
  // plugin needs, which libfabric drives. Claiming in this order means each
  // library's own DT_NEEDED entries bind to a copy claimed just above rather
  // than to whatever ld.so.cache resolves them to.
  static constexpr const char *kFabricLibs[] = {
      "libnl-3.so.200", "libnl-route-3.so.200", "libibverbs.so.1",
      "librdmacm.so.1", "libefa.so.1",          "libfabric.so.1",
  };
  // The plugin dir is <prefix>/lib/nixl/<flavor>; the fabric stack is staged
  // flat in <prefix>/lib next to libnixl.
  const std::filesystem::path libDir = pluginDir.parent_path().parent_path();
  bool claimedAny = false;
  for (const char *name : kFabricLibs) {
    const std::filesystem::path lib = libDir / name;
    std::error_code ec;
    if (!std::filesystem::exists(lib, ec))
      continue;
    // Deliberately never dlclosed: the point is to hold the SONAME for the life
    // of the process.
    if (::dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL))
      claimedAny = true;
  }
  return claimedAny;
}

// Returns true if this host has an RDMA device with an InfiniBand-link-layer
// port. It distinguishes a real IB fabric (where the UCX verbs transports are
// needed) from other rdma-core users — notably AWS EFA, whose device reports a
// non-InfiniBand link layer. libmlx5 ships in ibverbs-providers and loads on
// EFA hosts too, so the library probe alone would steer them onto the verbs
// flavor; gating on an actual IB port keeps EFA (and any non-IB) host on the
// plain flavor exactly as before.
static bool hasInfinibandPort() {
  std::error_code ec;
  const std::filesystem::path root("/sys/class/infiniband");
  for (std::filesystem::directory_iterator dev(root, ec), devEnd;
       dev != devEnd && !ec; dev.increment(ec)) {
    std::error_code pec;
    const std::filesystem::path ports = dev->path() / "ports";
    for (std::filesystem::directory_iterator port(ports, pec), portEnd;
         port != portEnd && !pec; port.increment(pec)) {
      std::ifstream f(port->path() / "link_layer");
      std::string layer;
      if (f && std::getline(f, layer) && layer == "InfiniBand")
        return true;
    }
  }
  return false;
}

// Selects a GPU vendor's plugin directory: the verbs flavor when it stages the
// requested backend's plugin and its rdma-core load-time deps
// (libibverbs/libmlx5) resolve, else the plain flavor when staged at all
// (libplugin_UCX.so is its marker — it is the flavor that carries every
// backend), else nullopt. The verbs flavor adds the uct_ib RDMA transports for
// internode InfiniBand transfers (UCX still uses cuda_ipc/rocm_ipc/shm for
// same-node peers) but stages UCX alone, so a host asking for libfabric stays
// on the plain flavor even where its hardware would otherwise warrant verbs —
// the dual-fabric case, an EFA device and IB NICs side by side. `requireIbPort`
// additionally gates the verbs flavor on a real IB port, so a host with
// rdma-core but no IB — notably AWS EFA, where libmlx5 ships in
// ibverbs-providers — stays on the plain flavor, byte-identical to before.
static std::optional<std::filesystem::path>
selectVendorFlavor(const std::filesystem::path &base, const char *verbsDir,
                   const char *plainDir, bool requireIbPort, bool allowVerbs) {
  std::error_code ec;
  if (allowVerbs && M::stagesRequestedBackend(base / verbsDir) &&
      (!requireIbPort || hasInfinibandPort()) &&
      canLoadSharedLib("libibverbs.so.1") && canLoadSharedLib("libmlx5.so.1"))
    return base / verbsDir;
  if (std::filesystem::exists(base / plainDir / "libplugin_UCX.so", ec))
    return base / plainDir;
  return std::nullopt;
}
std::optional<std::filesystem::path>
M::resolveNixlPluginDir(const std::filesystem::path &base,
                        bool allowVerbsFlavor) {
  std::error_code ec;
  // Detect each vendor explicitly by its kernel device node (never assumed from
  // the other's absence). NVIDIA additionally requires a real IB port before
  // preferring verbs, so EFA hosts — rdma-core present, no IB — stay on cuda.
  if (std::filesystem::exists("/dev/nvidiactl", ec))
    if (auto dir = selectVendorFlavor(base, "cuda-verbs", "cuda",
                                      /*requireIbPort=*/true, allowVerbsFlavor))
      return dir;
  if (std::filesystem::exists("/dev/kfd", ec)) {
    // UCCL is the AMD default (speed-of-light on RoCE fabrics UCX cannot
    // saturate). An explicit UCX/libfabric request opts out; and where the
    // UCCL flavor is not staged (e.g. the hermetic test runfiles), this falls
    // through to the UCX flavors below.
    const std::string backend = requestedBackend();
    if ((backend.empty() || backend == "uccl") &&
        std::filesystem::exists(base / "rocm-uccl" / "libplugin_UCCL.so", ec))
      return base / "rocm-uccl";
    if (auto dir =
            selectVendorFlavor(base, "rocm-verbs", "rocm",
                               /*requireIbPort=*/false, allowVerbsFlavor))
      return dir;
  }
  return std::nullopt;
}
