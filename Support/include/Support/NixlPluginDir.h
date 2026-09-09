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

#ifndef SUPPORT_NIXLPLUGINDIR_H
#define SUPPORT_NIXLPLUGINDIR_H

#include <filesystem>
#include <optional>

namespace M {

/// Picks the NIXL plugin directory for the detected host GPU vendor.
///
/// The staged layout ships the NIXL transport plugins in per-vendor
/// subdirectories of `base` (cuda/ and cuda-verbs/ for the CUDA flavors plus
/// the EFA libfabric plugin in cuda/; rocm/, rocm-verbs/, and rocm-uccl/ for
/// the ROCm flavors; cpu/ for the GPU-free flavor); NIXL discovers plugins by
/// filename in exactly one directory, so the caller must pick one before the
/// NIXL plugin manager reads NIXL_PLUGIN_DIR. Each vendor is detected
/// explicitly via its kernel device node (/dev/nvidiactl for NVIDIA, /dev/kfd
/// for amdgpu) — never assumed from the absence of the other.
///
/// On NVIDIA the verbs flavor (cuda-verbs/, which adds the uct_ib RDMA
/// transports for internode InfiniBand to UCX) is preferred when it stages the
/// requested backend's plugin AND rdma-core resolves AND a real InfiniBand port
/// is present, so EFA hosts (rdma-core but no IB) and libfabric requests (the
/// verbs flavors are UCX-only) keep resolving to the plain cuda flavor. On AMD
/// the rocm-uccl flavor is the default; an explicit
/// MODULAR_NIXL_TRANSFER_BACKEND=ucx (or libfabric), or an environment where
/// rocm-uccl is not staged, falls back to the UCX flavors, where rocm-verbs is
/// preferred over rocm when its hard load-time dependencies (rdma-core)
/// resolve.
///
/// Returns std::nullopt when no vendor (or no staged flavor for the detected
/// vendor) is found; callers then leave NIXL_PLUGIN_DIR unset and NIXL
/// transport construction fails downstream with its normal plugin-not-found
/// error. This runs in contexts that include CPU-only and macOS hosts where
/// NIXL is legitimately unused — so it must not hard-error.
///
/// `allowVerbsFlavor` (default true) selects the verbs flavor when the host
/// warrants it. Pass false to force the plain flavor even on an IB host: two
/// NIXL agents in ONE process open two mlx5 device contexts on the same HCA,
/// which host rdma-core cannot tear down cleanly (a reserved-QPN mutex
/// assertion in mlx5_free_context). Production creates one agent per process
/// and is unaffected, but in-process loopback tests must opt out. The real IB
/// transport is covered instead by the cross-process cross-node smoke.
std::optional<std::filesystem::path>
resolveNixlPluginDir(const std::filesystem::path &base,
                     bool allowVerbsFlavor = true);

/// Whether `pluginDir` stages the plugin for the requested transfer backend.
///
/// NIXL loads exactly one plugin out of the directory it is pointed at — the
/// one named after the backend the KVTransferEngine asks for
/// (`libplugin_<BACKEND>.so`, uppercase; `MODULAR_NIXL_TRANSFER_BACKEND` unset
/// means ucx). A flavor that omits that plugin cannot serve the host however
/// well its hardware matches, so the resolver requires this of any flavor it
/// prefers over the fully-stocked plain one.
bool stagesRequestedBackend(const std::filesystem::path &pluginDir);

/// Claims the SONAMEs of the EFA fabric stack staged alongside `pluginDir`, for
/// hosts that asked for the libfabric backend.
///
/// The libfabric plugin and the libraries it pulls in bind versioned symbols
/// that only the EFA build staged in `<prefix>/lib` provides — FABRIC_1.8 from
/// `libfabric.so.1`, IBVERBS_PRIVATE_59 from `libibverbs.so.1` — yet they
/// record generic SONAMEs that a much older distro copy also claims (Ubuntu
/// packages libfabric 1.14 and rdma-core as dependencies of the Open MPI we
/// install for expert parallelism), and that unrelated components drag in:
/// torch's bundled NVSHMEM ships its own libfabric transport, and its bundled
/// cuFile RDMA plugin (`libcufile_rdma.so.1`) needs libibverbs. Whichever copy
/// loads first owns the SONAME process-wide, because an already-loaded SONAME
/// is never re-resolved — not against a later rpath, and not even against a
/// later request by absolute path. Losing the race leaves the plugin
/// permanently unloadable, reported as a generic "backend not found".
///
/// Claiming our copies up front (and keeping them loaded) makes the outcome
/// independent of load order: each is a strict superset of the distro copy's
/// symbol versions, so later consumers bind against it happily. They are
/// claimed leaf-first, so each one's own dependencies resolve against a copy
/// already claimed here rather than against whatever ld.so.cache offers.
///
/// Only the libfabric backend gets this. The verbs UCX flavors deliberately run
/// on the host's rdma-core, whose provider plugins (libmlx5) bind the matching
/// IBVERBS_PRIVATE version and would fail to load against ours.
///
/// Returns false when nothing was claimed — another backend was requested,
/// nothing is staged next to the plugins, or every load failed (e.g. no CUDA
/// driver, which the EFA libfabric needs). None of those are errors here; the
/// backend that needs them reports its own failure downstream.
bool preloadStagedFabricLibs(const std::filesystem::path &pluginDir);

} // namespace M

#endif // SUPPORT_NIXLPLUGINDIR_H
