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

# RUN: %parse-mojo-isolated %s | FileCheck --check-prefix=COMPTIME %s -DMOJO_MAJOR=%{mojo_version_major} -DMOJO_MINOR=%{mojo_version_minor} -DMOJO_PATCH=%{mojo_version_patch}
# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck --check-prefix=LOWER-LIT %s -DMOJO_MAJOR=%{mojo_version_major} -DMOJO_MINOR=%{mojo_version_minor} -DMOJO_PATCH=%{mojo_version_patch}

# Test that `lit.mojo.version.*` ops are lowered correctly by the parser
# and can be used at compile time via comptime folding.


# COMPTIME-LABEL: lit.fn @"get_mojo_version_major_builtin()"() -> !kgen.scalar<index> always_inline_builtin
# COMPTIME:         "lit.mojo.version.major"() : () -> !kgen.scalar<index>


# LOWER-LIT-LABEL: kgen.generator @"mojo_version_ops::get_mojo_version_major_builtin()"()
# LOWER-LIT: kgen.param.constant: scalar<index> = <[[MOJO_MAJOR]]>
@always_inline("builtin")
def get_mojo_version_major_builtin() -> __mlir_type.`!kgen.scalar<index>`:
    return __mlir_op.`lit.mojo.version.major`()


# COMPTIME-LABEL: lit.fn @"get_mojo_version_minor_builtin()"() -> !kgen.scalar<index> always_inline_builtin
# COMPTIME:         "lit.mojo.version.minor"() : () -> !kgen.scalar<index>


# LOWER-LIT-LABEL: kgen.generator @"mojo_version_ops::get_mojo_version_minor_builtin()"()
# LOWER-LIT: kgen.param.constant: scalar<index> = <[[MOJO_MINOR]]>
@always_inline("builtin")
def get_mojo_version_minor_builtin() -> __mlir_type.`!kgen.scalar<index>`:
    return __mlir_op.`lit.mojo.version.minor`()


# COMPTIME-LABEL: lit.fn @"get_mojo_version_patch_builtin()"() -> !kgen.scalar<index> always_inline_builtin
# COMPTIME:         "lit.mojo.version.patch"() : () -> !kgen.scalar<index>


# LOWER-LIT-LABEL: kgen.generator @"mojo_version_ops::get_mojo_version_patch_builtin()"()
# LOWER-LIT: kgen.param.constant: scalar<index> = <[[MOJO_PATCH]]>
@always_inline("builtin")
def get_mojo_version_patch_builtin() -> __mlir_type.`!kgen.scalar<index>`:
    return __mlir_op.`lit.mojo.version.patch`()


# COMPTIME-LABEL: lit.fn @"get_mojo_version_major()"() -> !kgen.scalar<index>
# COMPTIME:         "lit.mojo.version.major"() : () -> !kgen.scalar<index>


# LOWER-LIT-LABEL: kgen.generator @"mojo_version_ops::get_mojo_version_major()"()
# LOWER-LIT: kgen.param.constant: scalar<index> = <[[MOJO_MAJOR]]>
def get_mojo_version_major() -> __mlir_type.`!kgen.scalar<index>`:
    return __mlir_op.`lit.mojo.version.major`()


# COMPTIME-LABEL: lit.fn @"get_mojo_version_minor()"() -> !kgen.scalar<index>
# COMPTIME:         "lit.mojo.version.minor"() : () -> !kgen.scalar<index>


# LOWER-LIT-LABEL: kgen.generator @"mojo_version_ops::get_mojo_version_minor()"()
# LOWER-LIT: kgen.param.constant: scalar<index> = <[[MOJO_MINOR]]>
def get_mojo_version_minor() -> __mlir_type.`!kgen.scalar<index>`:
    return __mlir_op.`lit.mojo.version.minor`()


# COMPTIME-LABEL: lit.fn @"get_mojo_version_patch()"() -> !kgen.scalar<index>
# COMPTIME:         "lit.mojo.version.patch"() : () -> !kgen.scalar<index>


# LOWER-LIT-LABEL: kgen.generator @"mojo_version_ops::get_mojo_version_patch()"()
# LOWER-LIT: kgen.param.constant: scalar<index> = <[[MOJO_PATCH]]>
def get_mojo_version_patch() -> __mlir_type.`!kgen.scalar<index>`:
    return __mlir_op.`lit.mojo.version.patch`()


# COMPTIME: lit.alias.decl {{.*}}major{{.*}}[[MOJO_MAJOR]]
comptime major = get_mojo_version_major_builtin()

# COMPTIME: lit.alias.decl {{.*}}minor{{.*}}[[MOJO_MINOR]]
comptime minor = get_mojo_version_minor_builtin()

# COMPTIME: lit.alias.decl {{.*}}patch{{.*}}[[MOJO_PATCH]]
comptime patch = get_mojo_version_patch_builtin()
