# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -I=%S/inputs %s

# Regression test: a plain dotted `import a.b.c` is built into its gated
# ImportOp chain at parse time, so every segment it binds (`a`, `a.b`, `a.b.c`)
# is resolvable before any reference. `timing_user` is resolved on-demand (as a
# dependency, not a top-level script), and uses its dotted imports from default
# arguments — which are resolved before function bodies. This previously failed
# with "use of unknown declaration" because the import's tree was only built
# lazily, after signatures.
from timing_user import body_use, default_use


def main():
    body_use()
    default_use()
