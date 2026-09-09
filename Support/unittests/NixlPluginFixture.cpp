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

// Stands in for cuda/libplugin_LIBFABRIC.so in NixlPluginDirTest: it binds a
// symbol that only the EFA libfabric fixture exports at FABRIC_1.8, so it loads
// against that copy and fails against the distro one -- exactly how the real
// plugin behaves. It also pulls in the libefa fixture, whose own requirement on
// the EFA-patched libibverbs is the second SONAME the preload has to claim.

extern "C" int fi_fixture_open_v18();
extern "C" int efa_fixture_open();

extern "C" __attribute__((visibility("default"))) int nixl_plugin_init() {
  return fi_fixture_open_v18() + efa_fixture_open();
}
