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

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo demangle --one -two --help | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: mojo-demangle

# Reject unknown options.
# RUN: not mojo demangle -one --two 'aModule::main()' 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo{{.*}}: error: unrecognized argument '-one'

# RUN: mojo demangle 'aModule::main()' | FileCheck -check-prefix="SIMPLE" %s
# SIMPLE: Mangled: "aModule::main()" - Modules: ["aModule"], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> ()

# Names can be passed in via stdin.
# RUN: echo "aModule::main()" | mojo demangle | FileCheck %s --check-prefix SIMPLE

# RUN: mojo demangle 'aModule::AStruct::main()' | FileCheck -check-prefix="STRUCT" %s
# STRUCT: Mangled: "aModule::AStruct::main()" - Modules: ["aModule", "AStruct"], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle 'aPackage::aModule::AStruct::BStruct::main()' | FileCheck -check-prefix="NESTED" %s
# NESTED: Mangled: "aPackage::aModule::AStruct::BStruct::main()" - Modules: ["aPackage", "aModule", "AStruct", "BStruct"], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle 'aModule::AStruct::BStruct::main(index,!kgen.struct<(dtype)>)' | FileCheck -check-prefix="NESTED2" %s
# NESTED2: Mangled: "aModule::AStruct::BStruct::main(index,!kgen.struct<(dtype)>)" - Modules: ["aModule", "AStruct", "BStruct"], Structs: [], Symbol: "main", Identifier: "main", Signature: (index, !kgen.struct<(dtype)>) -> ()

# RUN: mojo demangle 'AStruct::main()' | FileCheck -check-prefix="NOMODULE" %s
# NOMODULE: Mangled: "AStruct::main()" - Modules: ["AStruct"], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle 'main()' | FileCheck -check-prefix="NOMODULE2" %s
# NOMODULE2: Mangled: "main()" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle 'main(index,index)!kgen.struct<(index,index)>' | FileCheck -check-prefix="NOMODULE3" %s
# NOMODULE3: Mangled: "main(index,index)!kgen.struct<(index,index)>" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: (index, index) -> !kgen.struct<(index, index)>

# RUN: mojo demangle 'main()!kgen.struct<(index,index)>' | FileCheck -check-prefix="NOMODULE4" %s
# NOMODULE4: Mangled: "main()!kgen.struct<(index,index)>" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> !kgen.struct<(index, index)>

# RUN: mojo demangle 'aModule::AStruct' | FileCheck -check-prefix="NOFUNC" %s
# NOFUNC: Mangled: "aModule::AStruct" - Modules: ["aModule"], Structs: [], Symbol: "AStruct", Identifier: "AStruct", Signature: (none)

# RUN: mojo demangle 'aModule::AStruct::BStruct' | FileCheck -check-prefix="NESTEDNOFUNC" %s
# NESTEDNOFUNC: Mangled: "aModule::AStruct::BStruct" - Modules: ["aModule", "AStruct"], Structs: [], Symbol: "BStruct", Identifier: "BStruct", Signature: (none)

# RUN: mojo demangle 'Mod::AStruct::main(Index::Int)' | FileCheck -check-prefix=MANGLEDTYPE  %s
# MANGLEDTYPE: Mangled: "Mod::AStruct::main(Index::Int)" - Modules: ["Mod", "AStruct"], Structs: [], Symbol: "main", Identifier: "main"

# RUN: mojo demangle 'main(functions::SomeStruct[size, other_param]&)' | FileCheck -check-prefix=PARAMETRIZEDARG %s
# PARAMETRIZEDARG: Mangled: "main(functions::SomeStruct[size, other_param]&)" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: (none)

# RUN: mojo demangle 'AModule::AStruct::print[builtin::Index::Int,DType::DType](builtin::simd::SIMD[*(0,1), *(0,0)])' | FileCheck -check-prefix=PARAMETRIZEDARG2 %s
# PARAMETRIZEDARG2: Mangled: "AModule::AStruct::print[builtin::Index::Int,DType::DType](builtin::simd::SIMD[*(0,1), *(0,0)])" - Modules: ["AModule", "AStruct"], Structs: [], Symbol: "print[builtin::Index::Int,DType::DType]", Identifier: "print", Signature: (none)

# Demangling failures are printed to stderr.
# RUN: not mojo demangle 'aModule::AStruct::BStruct(!invalid.type)' 2>&1 | FileCheck -check-prefix="FAILURE" %s
# FAILURE: demangling failed

# Only one name at a time.
# RUN: not mojo demangle 'one' 'two' 2>&1 | FileCheck --check-prefix TOO-MANY %s
# TOO-MANY: cannot demangle both 'one' and 'two'

# An empty string can be demangled.
# RUN: mojo demangle "" | FileCheck %s --check-prefix EMPTY
# EMPTY: Mangled: "" - Modules: [], Structs: [], Symbol: "", Identifier: "", Signature: (none)
