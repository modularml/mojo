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
"""Declares `foreign_only_symbol`, which no package re-exports from its own
subtree — only the foreign `sole_reexporter` package surfaces it. Exercises the
foreign-only fallback: with no native owner, the re-exporter is suggested."""


def foreign_only_symbol():
    pass
