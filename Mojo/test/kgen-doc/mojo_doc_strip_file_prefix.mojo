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


# RUN: not kgen-doc -strip-file-prefix=%S %s 2>&1 | FileCheck %s
# Diagnostics from the parser go through SourceMgr, not the tool's error
# prefix, so the expected output starts directly with the filename (no
# "kgen-doc: error:" prefix).
# CHECK: {{^}}mojo_doc_strip_file_prefix.mojo:21:5: error
def main():
    4 = "hello"
