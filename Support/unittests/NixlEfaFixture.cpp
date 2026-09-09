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

// Stands in for libefa.so.1 in NixlPluginDirTest: the EFA verbs library the
// libfabric plugin needs, which binds a symbol only the EFA-patched libibverbs
// exports at IBVERBS_PRIVATE_59. It sits between the plugin and libibverbs on
// purpose — the SONAME this guards is claimed one level below what the plugin
// itself names, and only reached when the plugin is dlopened.

extern "C" int ibv_fixture_private();

extern "C" __attribute__((visibility("default"))) int efa_fixture_open() {
  return ibv_fixture_private();
}
