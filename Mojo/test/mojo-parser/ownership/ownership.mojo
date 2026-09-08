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

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

def marker(): pass

# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample(ImplicitlyCopyable, Copyable):
  var x : Int
  def __init__(out self): self.x = 42; pass
  def noop(self): pass
  def __bool__(self) -> Bool: return True

  # Destructor should not recurse.
  # CHECK-LABEL: lit.fn @"__deinit__
  # CHECK-NEXT:    [[IMMREF:%.*]] = lit.ref.immut %self
  # CHECK-NEXT:    lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT:    %none = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT:    lit.ownership.mark_destroyed %self
  # CHECK-NEXT:    kgen.return %none : !kgen.none
  def __deinit__(deinit self):
    self.noop()

def consume(var a: MemExample): pass

struct MemPair(Movable where False):
  var a: MemExample
  var b: MemExample
  def __init__(out self):
    self.a = self.b := MemExample()

  def use(self): pass


# CHECK-LABEL: lit.struct.decl @RegExample
struct RegExample(ImplicitlyCopyable, RegisterPassable):
  def __init__(out self):
    return

  @implicit
  def __init__(out self, value: Int):
    pass

  def __init__(out self, *, copy: Self): # CHECK: lit.fn @"__init__{{.*}}%copy
    return

  def noop(self): pass
  # CHECK-LABEL: lit.fn @"__deinit__
  # CHECK-NEXT:  = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT: lit.ownership.mark_destroyed %self
  # CHECK-NEXT: kgen.return
  def __deinit__(deinit self):
    pass

  def mutate(mut self):
    pass

def consume(var a: RegExample): pass

# CHECK-LABEL: lit.fn @"destructors
# CHECK-SAME: (%arg0: !lit.ref<!MemExample, mut {{.*}}> owned_in_mem)
def destructors(var arg0: MemExample):
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%arg0)

  # CHECK-NEXT: %mem1 = lit.var.decl "mem1" var
  # expected-warning @+1 {{assignment to 'mem1' was never used}}
  var mem1 = MemExample()
  # CHECK-NEXT: lifetime.start %mem1
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem1)
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem1)
  # CHECK-NEXT: lifetime.end %mem1

  var mem2 = MemExample()
  # CHECK-NEXT: %mem2 = lit.var.decl "mem2" var
  # CHECK-NEXT: lifetime.start %mem2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem2)
  mem2.noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem2
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem2)
  # CHECK-NEXT: lifetime.end %mem2

  mem2 = MemExample()
  # CHECK-NEXT: lifetime.start %mem2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem2)

  # expected-warning @+1 {{assignment to 'reg' was never used}}
  var reg = RegExample()
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: %reg = lit.var.decl "reg"
  # CHECK-NEXT: lifetime.start %reg
  # CHECK-NEXT: lit.ref.store [[TMP]], %reg
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%reg)
  # CHECK-NEXT: lifetime.end

  mem2.noop()
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem2
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%mem2)
  # CHECK-NEXT: lifetime.end %mem2

  # CHECK-NEXT: %mem3 = lit.var.decl "mem3"
  # CHECK-NEXT: lifetime.start %mem3
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem3)
  var mem3 = MemExample()

  # CHECK-NEXT: lit.ownership.use %mem3
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}(%mem3)
  # CHECK-NEXT: lifetime.end %mem3
  consume(mem3^)

  # CHECK-NEXT: [[MEMTMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lifetime.start [[MEMTMP]]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[MEMTMP]])
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[MEMTMP]])
  # CHECK-NEXT: lifetime.end [[MEMTMP]]
  consume(MemExample())

  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@RegExample::@"__init__{{.*}}()
  # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lit.var.lifetime.start [[ANON]]
  # CHECK-NEXT: lit.ref.store [[TMP]], [[ANON]]
  # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[ANON]]
  # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMM]])
  RegExample().noop()
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[ANON]])
  # CHECK-NEXT: lit.var.lifetime.end [[ANON]]

  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@RegExample::@"__init__{{.*}}()
  # CHECK-NEXT: %localReg = lit.var.decl
  # CHECK-NEXT: lifetime.start %localReg
  # CHECK-NEXT: lit.ref.store [[TMP]], %localReg
  # expected-warning @+1 {{assignment to 'localReg' was never used}}
  var localReg = RegExample()


# CHECK-LABEL: lit.fn @"indirect_call
def indirect_call[detail_fn: def() thin -> MemExample]():
       # CHECK: %mem = lit.var.decl
       # CHECK-NEXT: lifetime.start %mem
       # CHECK-NEXT: lit.call{{.*}}(%mem)
       var mem = detail_fn()
       # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %mem
       # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
       mem.noop()
       # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%mem)

# CHECK-LABEL: lit.struct.decl @Parameterized<level: !Int>
struct Parameterized[level: Int](Movable where False):
    def __init__(out self): pass

    def __deinit__(deinit self):
        pass

# CHECK-LABEL: lit.fn @"test_parameterized
def test_parameterized():
  # CHECK: %x = lit.var.decl "x"
  # expected-warning @+1 {{assignment to 'x' was never used}}
  var x = Parameterized[4]()
  # CHECK: lit.call {{.*}}@"__init__{{.*}}(%x)
  # CHECK: lit.call {{.*}}__deinit__{{.*}}<:!Int {:scalar<index> 4}>(%x)

struct Complicated(Movable where False):
  var a: MemExample
  var b: MemExample

# This exercises turning a pop.pointer into an RValue, which produces an 'owned'
# pointer magically from memory.
# CHECK-LABEL: lit.fn @"testTakePointeeAsOwned1
def testTakePointeeAsOwned1(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`]):
  # This should run the destructor.
  # CHECK-NEXT: [[REF1:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.ownership.use [[REF1]]
  _ = __get_address_as_owned_value(ptr)

  # This should run the destructor and not get omitted.
  # CHECK-NEXT: [[REF2:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.ownership.use [[REF2]]
  _ = __get_address_as_owned_value(ptr)
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[REF2]])
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[REF1]])


# CHECK-LABEL: lit.fn @"testTakePointeeAsOwned2
def testTakePointeeAsOwned2(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`],
                          i1ptr: __mlir_type.`!kgen.pointer<i1>`):

  # The RValue can be consumed directly.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr end_uninit :
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[REF]])
  consume(__get_address_as_owned_value(ptr))

  # i1 doesn't have ownership but should still work for generality.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %i1ptr end_uninit :
  # CHECK-NEXT: %ownedI1 = lit.var.decl
  # CHECK-NEXT: [[I1VAL:%.*]] = lit.load.consume [[REF]]
  # CHECK-NEXT: lifetime.start %ownedI1
  # CHECK-NEXT: lit.ref.store [[I1VAL]], %ownedI1
  # CHECK-NEXT: lifetime.end %ownedI1
  # expected-warning @+1 {{assignment to 'ownedI1' was never used}}
  var ownedI1 = __get_address_as_owned_value(i1ptr)

  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>


# CHECK-LABEL: testGetAsUninitializedObject
def testGetAsUninitializedObject(ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`]):
   # Overwriting the value in a __get_address_as_uninit_lvalue does not destroy
  # the memory, because it is uninit.
  # CHECK-NEXT: [[REF:%.*]] = lit.ref.from_pointer %ptr start_uninit :
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])
  __get_address_as_uninit_lvalue(ptr) = MemExample()

  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: testCondGetAsUninitializedObject
# Early exit from def using __get_address_as_uninit_lvalue should work.
# https://github.com/modularml/modular/issues/27472
def testCondGetAsUninitializedObject(exit_early: __mlir_type.`!kgen.scalar<bool>`,
                                  ptr: __mlir_type[`!kgen.pointer<`, MemExample, `>`]):
  # CHECK: hlcf.elif
  if exit_early:
      return

  # CHECK: [[REF:%.*]] = lit.ref.from_pointer %ptr start_uninit :
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[REF]])
  # CHECK-NEXT: %none = kgen.param.constant: none
  # CHECK-NEXT: kgen.return %none
  __get_address_as_uninit_lvalue(ptr) = MemExample()


# CHECK-LABEL: lit.struct.decl @FieldSensitiveMemExample
struct FieldSensitiveMemExample(ImplicitlyCopyable):
  var f1 : MemExample
  var f2 : MemExample

  # CHECK: lit.fn @"__init__
  def __init__(out self):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%0)
    self.f1 = MemExample()
    # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f2]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%2)
    self.f2 = MemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return

  # CHECK: lit.fn @"__init__
  def __init__(out self, a: MemExample, b: MemExample):
    self.f1 = a
    self.f2 = b

  def __init__(out self, *, copy: Self):
    self = Self(copy.f1, copy.f2)

  # CHECK-LABEL: lit.fn @"mutate
  def mutate(mut self):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%0)

    # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%2)
    self.f1 = MemExample()
    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>

  # CHECK-LABEL: lit.fn @"mutate2
  def mutate2(deinit self):
    # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%self)
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%self)
    self = FieldSensitiveMemExample()

    # This is a 'deinit' method, so both F1 and F2 need to be destroyed
    # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %self[f1]
    # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}([[F1R]])
    # CHECK-NEXT: [[F2R:%.*]] = lit.ref.struct.ger %self[f2]
    # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}([[F2R]])

    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: lit.ownership.mark_destroyed %self

  # This disables the destructor of 'x' which causes the fields to be destroyed.
  # CHECK-LABEL: lit.fn @"disableDtor
  def disableDtor(deinit x):
    # CHECK-NEXT: [[F1R:%.*]] = lit.ref.struct.ger %x[f1]
    # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}([[F1R]])
    # CHECK-NEXT: [[F2R:%.*]] = lit.ref.struct.ger %x[f2]
    # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}([[F2R]])
    # CHECK-NEXT: kgen.param.constant: none
    # CHECK-NEXT: lit.ownership.mark_destroyed %x
    pass


  # CHECK-LABEL: lit.fn @"__deinit__
  # CHECK-NEXT: %0 = lit.ref.struct.ger %self[f1]
  # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%0)
  # CHECK-NEXT: %2 = lit.ref.struct.ger %self[f2]
  # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%2)
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: lit.ownership.mark_destroyed %self


# CHECK-LABEL: lit.fn @"regpassable_owned_args_mutable
def regpassable_owned_args_mutable(var x: RegExample):
  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x)
  x.mutate()

  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%x)
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}"__init__{{.*}}()
  # CHECK-NEXT: lit.ref.store [[TMP]], %x
  x = RegExample()

  # CHECK-NEXT: lit.call {{.*}}mutate{{.*}}(%x)
  x.mutate()
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%x)

# Result optimization cannot emit directly into a value that is passed as an
# argument, because this forms a mutable reference to something immutable
# implicitly.  We must invoke the copy ctor.
# CHECK-LABEL: lit.fn @"use_and_return
def use_and_return(a: FieldSensitiveMemExample) -> FieldSensitiveMemExample:
  # This will read from 'a' and write into the result slot in an arbitrary
  # order. They cannot alias.
  return FieldSensitiveMemExample(a.f2, a.f1)

def use_and_return2(a: FieldSensitiveMemExample) -> MemExample:
  return a.f2

def use_inout_and_return(mut a: FieldSensitiveMemExample) -> FieldSensitiveMemExample:
  return a

def return_ref(mut a: FieldSensitiveMemExample) -> ref [a] FieldSensitiveMemExample:
  return a

# CHECK-LABEL: lit.fn @"test_result_optimization
def test_result_optimization():
  # CHECK-NEXT: %example = lit.var.decl "example"
  # CHECK-NEXT: lifetime.start %example
  # CHECK-NEXT: lit.call {{.*}}"__init__{{.*}}(%example)
  var example = FieldSensitiveMemExample()

  # Direct reuse of the result slot forces a temporary.

  # CHECK: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
  # CHECK-NEXT: lifetime.start %__call_result_tmp__
  # CHECK-NEXT: lit.call {{.*}}use_and_return{{.*}}([[IMMREF]], %__call_result_tmp__)
  # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%example)
  # CHECK-NEXT: lit.var.lifetime.end %example
  # CHECK-NEXT: lit.var.lifetime.start %example
  # CHECK-NEXT: lit.call {{.*}}@"__init__{{.*}}move"
  # CHECK-NEXT: lit.var.lifetime.end %__call_result_tmp__
  example = use_and_return(example)

  # Aliased reuse of part of the result slot forces a temporary.

  # CHECK-NEXT: [[F1:%.*]] = lit.ref.struct.ger %example[f1]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: %__call_result_tmp___0 = lit.var.decl
  # CHECK-NEXT: lifetime.start %__call_result_tmp___0
  # CHECK-NEXT: lit.call @ownership::@"use_and_return2{{.*}}([[IMMREF]], %__call_result_tmp___0)
  example.f1 = use_and_return2(example)
  # CHECK-NEXT: [[F1_2:%.*]] = lit.ref.struct.ger %example[f1]
  # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}([[F1_2]])
  # CHECK-NEXT: lit.call {{.*}}@"__init__{{.*}}move"
  # CHECK-NEXT: lifetime.end %__call_result_tmp___0

  # Mutating self through a reference forces a temporary.
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: [[RETREF:%.*]] = lit.call {{.*}}return_ref{{.*}}(%example)
  # CHECK-NEXT: [[TMPVAR:%.*]] = lit.var.decl
  # CHECK-NEXT: lifetime.start [[TMPVAR]]
  # CHECK-NEXT: lit.call {{.*}}use_and_return{{.*}}([[IMMREF]], [[TMPVAR]])

  # Delete the old thing at the reference pointed-to-by return_ref before we
  # copy into it.
  # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}([[RETREF]])

  # CHECK-NEXT: lit.call {{.*}}@"__init__{{.*}}move"
  # CHECK-NEXT: lifetime.end [[TMPVAR]]
  return_ref(example) = use_and_return(example)

  # CHECK-NEXT: lit.call {{.*}}@"mutate{{.*}}(%example)
  example.mutate()
  # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%example)
  # CHECK-NEXT: lifetime.end %example

  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>

# CHECK-LABEL: lit.fn @"impl_mutable_arg
def impl_mutable_arg(mut a: FieldSensitiveMemExample, mut b: FieldSensitiveMemExample):
  # CHECK-NEXT: lit.call {{.*}}@"__deinit__{{.*}}(%b)
  # CHECK-NEXT: lit.call {{.*}}use_inout_and_return{{.*}}(%a, %b)

  b = use_inout_and_return(a)

##===----------------------------------------------------------------------===##
# Consume Expressions
##===----------------------------------------------------------------------===##

# CHECK: lit.fn @"test_result_consume_reg
def test_result_consume_reg(cond: __mlir_type.`!kgen.scalar<bool>`) -> RegExample:
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: %example2 = lit.var.decl
  # CHECK-NEXT: lit.var.lifetime.start %example2
  # CHECK-NEXT: lit.ref.store [[TMP]], %example2
  var example2 = RegExample()

  # CHECK-NEXT: hlcf.elif %cond {
  if (cond):
    # CHECK-NEXT: lit.ownership.use %example2
    # CHECK-NEXT: [[TMP2:%.*]] = lit.load.consume %example2
    # CHECK-NEXT: lifetime.end %example2
    # CHECK-NEXT: kgen.return [[TMP2]]
    return example2^
  else: # CHECK-NEXT: } else {
    # CHECK-NEXT: [[TMP2:%.*]] = lit.load.consume %example2
    # CHECK-NEXT: lifetime.end %example2
    # CHECK-NEXT: kgen.return [[TMP2]]
    return example2  # copy/deinit -> move optimization.

# CHECK: lit.fn @"consumeMem
def consumeMem(var x: MemExample):
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%x)
  # CHECK-NEXT: kgen.param.constant: none
  pass

# CHECK: lit.fn @"test_result_consume_mem
def test_result_consume_mem(cond: __mlir_type.`!kgen.scalar<bool>`) -> MemExample:
  # CHECK-NEXT: %example = lit.var.decl
  # CHECK-NEXT: lifetime.start %example
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%example)
  var example = MemExample()

  # This doesn't consume example, so it must copy it. It does consume the copy.
  # COM: [[IMMREF:%.*]] = lit.ref.immut %example
  # CHECK-NEXT: [[MEMTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lifetime.start [[MEMTMP]]
  # CHECK-NEXT: lit.memcpy %example, [[MEMTMP]]
  # CHECK-NEXT: lit.call {{.*}}consumeMem{{.*}}([[MEMTMP]])
  # CHECK-NEXT: lifetime.end [[MEMTMP]]
  consumeMem(example)

  # This does consume example, so no copy needed.
  # CHECK-NEXT: lit.ownership.use %example
  # CHECK-NEXT: lit.call {{.*}}consumeMem{{.*}}(%example)
  # CHECK-NEXT: lifetime.end %example
  consumeMem(example^)

  # CHECK-NEXT: %example2 = lit.var.decl
  # CHECK-NEXT: lifetime.start %example2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%example2)
  var example2 = MemExample()

  # CHECK-NEXT: lit.ownership.use %example2
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}move"
  # CHECK-NEXT: lifetime.end %example2
  # CHECK-NEXT: kgen.param.constant: none
  return example2^

# CHECK-LABEL: lit.struct.decl @BigRegExample
struct BigRegExample(ImplicitlyCopyable, RegisterPassable):
  var a: RegExample
  var b: RegExample

  # CHECK-LABEL: lit.fn @"__init__()"
  def __init__(out self):
    # CHECK-NEXT: %self = lit.var.decl "self" initoutarg
    # CHECK-NEXT: [[A:%.*]] = lit.ref.struct.ger %self[a]
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: lit.ref.store [[TMP]], [[A]]
    # CHECK-NEXT: [[B:%.*]] = lit.ref.struct.ger %self[b]
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: lit.ref.store [[TMP]], [[B]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %self
    # CHECK-NEXT: lit.var.lifetime.end %self
    # CHECK-NEXT: kgen.return [[TMP]]
    self.a = RegExample()
    self.b = RegExample()

  # CHECK-LABEL: lit.fn @"__init__{{.*}}"{{.*}}*, %copy:
  def __init__(out self, *, copy: Self):
    # CHECK-NEXT: %self = lit.var.decl "self" initoutarg
    # CHECK-NEXT: [[SA:%.*]] = lit.ref.struct.ger %self[a]
    # CHECK-NEXT: [[EA:%.*]] = lit.ref.struct.ger %copy[a]
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}copy"
    # CHECK-NEXT: lit.ref.store [[TMP]], [[SA]]
    # CHECK-NEXT: [[SB:%.*]] = lit.ref.struct.ger %self[b]
    # CHECK-NEXT: [[EB:%.*]] = lit.ref.struct.ger %copy[b]
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}copy"
    # CHECK-NEXT: lit.ref.store [[TMP]], [[SB]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %self
    # CHECK-NEXT: lit.var.lifetime.end %self
    # CHECK-NEXT: kgen.return [[TMP]]
    self.a = copy.a
    self.b = copy.b

  # CHECK-LABEL: lit.fn @"__deinit__
  # CHECK-NEXT: [[APTR:%.*]] = lit.ref.struct.ger %self[a]
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[APTR]])
  # CHECK-NEXT: [[BPTR:%.*]] = lit.ref.struct.ger %self[b]
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[BPTR]])
  # CHECK-NEXT:  = kgen.param.constant{{.*}} <#kgen.none>
  # CHECK-NEXT: lit.ownership.mark_destroyed %self
  # CHECK-NEXT: kgen.return


def take_regexample_ref(ref r: RegExample): pass
def ret_big_reg() -> BigRegExample:
  return BigRegExample()

# CHECK-LABEL: lit.fn @"bigreg_test
def bigreg_test():
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: %varThing = lit.var.decl "varThing"
  # CHECK-NEXT: lifetime.start %varThing
  # CHECK-NEXT: lit.ref.store [[TMP]], %varThing
  var varThing = BigRegExample()

  # CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: lit.ownership.use [[FIELD]]
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[FIELD]])
  consume(varThing.a^)

  # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %varThing[b]
  # CHECK-NEXT: [[BVAL:%.*]] = lit.ref.immut [[BREF]]
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}copy"
  # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: lifetime.start [[ANON]]
  # CHECK-NEXT: lit.ref.store [[TMP]], [[ANON]]
  # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[ANON]])
  # CHECK-NEXT: lifetime.end [[ANON]]
  consume(varThing.b)

  # CHECK-NEXT: [[AREF:%.*]] = lit.ref.struct.ger %varThing[a]
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
  # CHECK-NEXT: lit.ref.store [[TMP]], [[AREF]]
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%varThing)
  # CHECK-NEXT: lifetime.end %varThing
  varThing.a = RegExample()

  # Must drop the value in a register to pass by-ref
  # CHECK-NEXT: [[TMPREG:%.*]] = lit.call {{.*}}ret_big_reg
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lit.var.lifetime.start [[TMP]]
  # CHECK-NEXT: lit.ref.store [[TMPREG]], [[TMP]]
  # CHECK-NEXT: [[ELT:%.*]] = lit.ref.struct.ger [[TMP]][a]
  # CHECK-NEXT: [[ELTIMM:%.*]] = lit.ref.immut [[ELT]]
  # CHECK-NEXT: lit.call {{.*}}take_regexample_ref{{.*}}([[ELTIMM]])
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[TMP]])
  # CHECK-NEXT: lit.var.lifetime.end [[TMP]]
  take_regexample_ref(ret_big_reg().a)

  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: lit.struct.decl @ExoticDelExample
struct ExoticDelExample(RegisterPassable):
  var cond: __mlir_type.`!kgen.scalar<bool>`
  var b: BigRegExample
  var c: RegExample

 # CHECK-LABEL: lit.fn @"__deinit__
  def __deinit__(deinit self):
    # self.b gets destroyed ASAP since it isn't used.
    # CHECK-NEXT: [[BPTR:%.*]] = lit.ref.struct.ger %self[b]
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[BPTR]])

    # Test the condition
    # CHECK-NEXT: [[CONDPTR:%.*]] = lit.ref.struct.ger %self[cond]
    # CHECK-NEXT: [[CONDVAL:%.*]] = lit.ref.load [[CONDPTR]]
    # CHECK-NEXT: hlcf.elif [[CONDVAL]] {
    if self.cond:
      # This side we manually consume for c.

      # CHECK-NEXT: [[CREF:%.*]] = lit.ref.struct.ger %self[c]
      # CHECK-NEXT: lit.ownership.use [[CREF]]
      # CHECK-NEXT: lit.call {{.*}}consume{{.*}}([[CREF]])
      # CHECK-NEXT: hlcf.yield
      consume(self.c^)

    # CHECK-NEXT: } else {
    # Destroy C automatically on the else side.
    # CHECK-NEXT:  [[CPTR:%.*]] = lit.ref.struct.ger %self[c]
    # CHECK-NEXT:  lit.call {{.*}}__deinit__{{.*}}([[CPTR]])
    # CHECK-NEXT:  hlcf.yield
    # CHECK-NEXT:}

    # CHECK-NEXT: = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT:lit.ownership.mark_destroyed %self
    # CHECK-NEXT:kgen.return


# CHECK-LABEL: lit.fn @"def_borrowed
# CHECK-SAME: %a: !lit.ref<!MemExample, imm {{.*}}> imm_mem
def def_borrowed(a: MemExample) raises -> None:
  # CHECK: lit.ref.store %none, %__result__
  # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
  # CHECK-NEXT: return [[FALSE]]
  pass


# An explicit `imm` argument produces the same immutable-reference IR as the
# implicit default above.
# CHECK-LABEL: lit.fn @"def_imm
# CHECK-SAME: %a: !lit.ref<!MemExample, imm {{.*}}> imm_mem
def def_imm(imm a: MemExample) raises -> None:
  # CHECK: lit.ref.store %none, %__result__
  # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
  # CHECK-NEXT: return [[FALSE]]
  pass


# https://github.com/modularml/modular/issues/24161
struct AddrSpace(TrivialRegisterPassable):
    var _value: __mlir_type.index
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.index):
        self._value = value
    def value(self) -> __mlir_type.index:
        return self._value
@fieldwise_init
struct MemExamplePtr[addrspace: AddrSpace = __mlir_attr.`0:index`](TrivialRegisterPassable):
    var value: __mlir_type[
        `!kgen.pointer<`, MemExample, `, `, Self.addrspace._value, `>`
    ]

def sadge(ptr: MemExamplePtr[]):
    __get_address_as_uninit_lvalue(ptr.value) = MemExample()
    return


trait SomeTrait:
    pass

struct GenericType(SomeTrait, Movable where False):
    def __deinit__(deinit self):
        pass

struct GenericRegType(RegisterPassable, SomeTrait):
    def __deinit__(deinit self):
        pass

# CHECK-LABEL: lit.fn @"destruct_generic_return
def destruct_generic_return():
    @__parameter
    def return_generic_type[T: SomeTrait]() -> T:
        while True:
            pass

    # CHECK: call {{.*}}@GenericType::@"__deinit__
    _ = return_generic_type[GenericType]()
    # CHECK: call {{.*}}@GenericRegType::@"__deinit__
    _ = return_generic_type[GenericRegType]()


# CHECK-LABEL: lit.struct.decl @RegisterExistingDtor
struct RegisterExistingDtor(RegisterPassable):
    def __deinit__(deinit self):
        pass

struct RegisterNoDtor(RegisterPassable):
    pass

struct MemoryNoDtor(Movable where False):
    pass


# CHECK-LABEL: lit.struct.decl @RegExampleValue({{.*}}) register_passable
# Compiler crashes trying to insert a destructor call
# https://github.com/modularml/modular/issues/26410
@fieldwise_init
struct RegExampleValue(ImplicitlyCopyable, RegisterPassable):
  var x: RegExample
  def __init__(out self):
    self.x = RegExample()

  # Make sure the synthesized dtor is taken register style.
  # CHECK: lit.fn @"__deinit__{{.*}}(%self: !lit.ref<!RegExampleValue
  # CHECK-NEXT: lit.ref.struct.ger %self[x]
  # CHECK-NEXT: lit.call {{.*}}__deinit__
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK: lit.ownership.mark_destroyed %self

# [Bug] __result__ is uninitialized
# https://github.com/modularml/modular/issues/27792
# CHECK-LABEL: lit.fn @"test_or
def test_or(a: MemExample) -> MemExample:
  # CHECK: hlcf.elif {{.*}} {
  # CHECK:   lit.memcpy %a,
  # CHECK: } else {
  # CHECK:   lit.memcpy %a,
  # CHECK: }
  return a or a


# ===----------------------------------------------------------------------=== #
# Variadics
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"variadic_mems
# CHECK-SAME: [imm *"mems`2"](
# CHECK-SAME: %mems: !lit.ref<!lit.struct<#VariadicList <:!Bool {:scalar<bool> false}, :origin<false> *"mems.origin._mlir_origin``", :!lit.struct<#Origin <:!Bool {:scalar<bool> false}, :origin<false> *"mems.origin._mlir_origin``">> *"mems.origin`1", :!AnyType !MemExample, :!Bool {:scalar<bool> false}>>, imm *"mems`2"> imm_mem|pos_vararg)
def variadic_mems(*mems: MemExample):
  # CHECK-NEXT: %none = kgen.param.constant
  pass

# CHECK-LABEL: lit.fn @"call_variadic_mems
def call_variadic_mems(a: MemExample, b: MemExample):
  # CHECK-NEXT: %0 = lit.ref.upcast %a : <!MemExample, imm *"a`"> -> <!MemExample, imm {*"a`", *"b`1"}>
  # CHECK-NEXT: %1 = lit.ref.upcast %b : <!MemExample, imm *"b`1"> -> <!MemExample, imm {*"a`", *"b`1"}>
  # CHECK-NEXT: %__passed_varargs__ = lit.var.decl
  # CHECK-NEXT: [[ARRAY:%.*]] = pop.array.create [%0, %1]
  # CHECK-NEXT: lit.var.lifetime.start %__passed_varargs__
  # CHECK-NEXT: lit.ref.store [[ARRAY]], %__passed_varargs__
  # CHECK: lit.call {{.*}}VariadicList::@"__init__
  # CHECK: lit.call {{.*}}variadic_mems{{.*}}<:origin<false> {*"a`", *"b`1"}
  variadic_mems(a, b)
  # CHECK-NEXT: lit.var.lifetime.end %anonymous
  # CHECK-NEXT: lit.var.lifetime.end %__passed_varargs__

  # Variadic use keeps the memory value alive.
  # CHECK: %c = lit.var.decl "c"
  # CHECK-NEXT: lifetime.start %c
  # CHECK-NEXT: lit.memcpy %a, %c
  var c = a
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
  # CHECK-NEXT: {{%.*}} = lit.var.decl "__passed_varargs__"
  # CHECK-NEXT: {{%.*}} = pop.array.create {{.}}[[IMMREF]]
  # CHECK: lit.call {{.*}}VariadicList::@"__init__
  # CHECK: lit.call {{.*}}variadic_mems{{.*}}:origin<false> (mutcast mut *"c`4")>
  variadic_mems(c)
  # CHECK-NEXT: lit.var.lifetime.end %anonymous
  # CHECK-NEXT: lit.var.lifetime.end %__passed_varargs__
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%c)
  # CHECK-NEXT: lit.var.lifetime.end %c
  # CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: lit.fn @"variadic_field_sensitivity
def variadic_field_sensitivity():
  # Test that we field sensitively track variadics.
  # CHECK:  %memPair = lit.var.decl
  var memPair = MemPair()

  # CHECK: [[AREF:%.*]] = lit.ref.struct.ger %memPair[a]
  # One for the transfer, one for the assignment to _.
  # CHECK-NEXT: lit.ownership.use [[AREF]]
  # CHECK-NEXT: lit.ownership.use [[AREF]]
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[AREF]])
  _ = memPair.a^  # Destroy a.

  # Can still pass b through varargs.
  # CHECK: [[BREF:%.*]] = lit.ref.struct.ger %memPair[b]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[BREF]]
  # CHECK-NEXT: {{%.*}} = lit.var.decl "__passed_varargs__"
  # CHECK-NEXT: {{%.*}} = pop.array.create {{.}}[[IMMREF]]
  # CHECK: lit.call {{.*}}VariadicList::@"__init__
  # CHECK: lit.call {{.*}}variadic_mems{{.*}}:origin<false> (mutcast mut *"memPair`"->b)
  variadic_mems(memPair.b)

  # Need to restore 'a' so memPair may destruct.
  # CHECK: [[AREF:%.*]] = lit.ref.struct.ger %memPair[a]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[AREF]])
  memPair.a = MemExample()

  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%memPair)
  # CHECK-NEXT: lifetime.end %memPair
  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.fn @"variadic_inout_mems
# CHECK-SAME: [imm *"mems`2"]
# CHECK-SAME: (%mems: !lit.ref<!lit.struct<#VariadicList <:!Bool {:scalar<bool> true}, :origin<true> *"mems.origin._mlir_origin``", :!lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *"mems.origin._mlir_origin``">> *"mems.origin`1", :!AnyType !MemExample, :!Bool {:scalar<bool> false}>>, imm *"mems`2"> mut|pos_vararg)
def variadic_inout_mems(mut *mems: MemExample):
  # CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant
  # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}__getitem__{{.*}}(%mems, [[ZERO]])
  # CHECK-NEXT: [[XGER:%.*]] = lit.ref.struct.ger [[REF]][x]
  # CHECK-NEXT: [[ONE:%.*]] = kgen.param.constant
  # CHECK-NEXT: [[XREF:%.*]] = kgen.rebind [[XGER]]
  # CHECK-NEXT: lit.call {{.*}}__iadd__{{.*}}([[XREF]], [[ONE]])
  mems[0].x += 1

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.fn @"call_variadic_inout_mems
def call_variadic_inout_mems():
  var a = MemExample()
  var b = MemExample()
  # CHECK: [[AR:%.*]] = lit.ref.upcast %a : <!MemExample, mut *"a`"> -> <!MemExample, mut {*"a`", *"b`1"}>
  # CHECK-NEXT: [[BR:%.*]] = lit.ref.upcast %b : <!MemExample, mut *"b`1"> -> <!MemExample, mut {*"a`", *"b`1"}>
  # CHECK-NEXT: {{%.*}} = lit.var.decl "__passed_varargs__"
  # CHECK-NEXT: {{%.*}} = pop.array.create [[[AR]], [[BR]]]
  # CHECK: lit.call {{.*}}VariadicList::@"__init__
  # CHECK: lit.call {{.*}}variadic_inout_mems{{.*}}:origin<true> {*"a`", *"b`1"}
  variadic_inout_mems(a, b)
  # CHECK-NEXT: lit.var.lifetime.end %anonymous
  # CHECK-NEXT: lit.var.lifetime.end %__passed_varargs__
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[BR]])
  # CHECK-NEXT: lifetime.end %b
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[AR]])
  # CHECK-NEXT: lifetime.end %a

  # CHECK-NEXT: kgen.param.constant: none
  # CHECK-NEXT: kgen.return

# CHECK-LABEL: lit.fn @"variadic_owned_mems
def variadic_owned_mems(var *mems: MemExample):
    # CHECK: lit.ref.immut %mems
    # CHECK: VariadicList::@"__getitem__
    mems[0].x += 1


# CHECK-LABEL: lit.fn @"call_variadic_owned_mems
def call_variadic_owned_mems(var c: MemExample, var d: MemExample):
    variadic_owned_mems(c^, d^)
    # COM: Ensure owned convention of callee is honored.
    # CHECK:  lit.call {{.*}}::@"variadic_owned_mems
    # CHECK-NEXT: lit.var.lifetime.end %anonymous
    # CHECK-NEXT: lit.var.lifetime.end %__passed_varargs__
    # CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return %none : !kgen.none


# CHECK-LABEL: lit.fn @"test_partial_overwrite
def test_partial_overwrite(cond: __mlir_type.`!kgen.scalar<bool>`):
  # CHECK-NEXT: %pair = lit.var.decl "pair"
  # CHECK-NEXT: lifetime.start %pair
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%pair)
  var pair = MemPair()

  # CHECK-NEXT: hlcf.elif %cond {
  if cond:
    # Inserted destruction of incoming pair.b
    # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %pair[b]
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[BREF]])

    # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %pair[b]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[BREF]])
    pair.b = MemExample()

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %pair
    # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[IMMREF]])
    pair.use()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%pair)
    # CHECK-NEXT: lifetime.end %pair
    # CHECK-NEXT: hlcf.yield
  else: # CHECK-NEXT: } else {
    # Inserted destruction of whole pair.
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%pair)
    # CHECK-NEXT: lifetime.end %pair

    # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: kgen.return
    return
  # CHECK-NEXT: }

# CHECK-LABEL: lit.struct.decl @UninitField
struct UninitField(Movable where False):
  var field: MemExample

  # CHECK: lit.fn @"__init__()"
  def __init__(out self):
      # Show that we can mark a field as intentionally uninitialized.
      # Even after checklifetimes, we don't want the thing initialized.
      __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self.field))

      # CHECK-NEXT: %0 = lit.ref.struct.ger %self[field]
      # CHECK-NEXT: lit.ownership.mark_initialized %0
      # CHECK-NEXT: %none = kgen.param.constant
      # CHECK-NEXT: kgen.return %none

def maybeMemExample() raises -> MemExample:
   return MemExample()

struct HasMemExample(Movable where False):
  var fh: MemExample
  # CHECK-LABEL: lit.fn @"destroyPotentiallyOverwrittenValueRegardlessOfOutcome
  def destroyPotentiallyOverwrittenValueRegardlessOfOutcome(mut self):
    # CHECK-NEXT: %__try_error__ = lit.var.dec
    # CHECK-NEXT: lit.try "try0" {
    try:
      # CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %self[fh]
      # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
      # CHECK-NEXT: lifetime.start %__try_error__
      # CHECK-NEXT: lifetime.start %__call_result_tmp__
      # CHECK-NEXT: lit.call {{.*}}maybeMemExample{{.*}}(%__try_error__, %__call_result_tmp__)
      self.fh = maybeMemExample()

      # Handle the error and other case.  The error isn't used, so delete it
      # here.

      # CHECK-NEXT: if
      # CHECK-NEXT:   lit.call {{.*}}Error::@"__deinit__{{.*}}(%__try_error__)
      # CHECK-NEXT:   lifetime.end %__try_error__
      # CHECK-NEXT:   mark_consumed %__call_result_tmp__
      # CHECK-NEXT:   lifetime.end %__call_result_tmp__
      # CHECK-NEXT:   lit.try.raise
      # CHECK-NEXT: } else {

      # On success, we overwrite the field.
      # CHECK-NEXT:   [[FIELD2:%.*]] = lit.ref.struct.ger
      # CHECK-NEXT:   lit.call {{.*}}__deinit__{{.*}}([[FIELD2]])
      # CHECK-NEXT:   mark_consumed %__try_error__
      # CHECK-NEXT:   lifetime.end %__try_error__
      # CHECK-NEXT:   yield
      # CHECK-NEXT: }

      # On success we move the result value into the destination.
      # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}move"
      # CHECK-NEXT: lifetime.end %__call_result_tmp__
      # CHECK-NEXT: lit.try.yield
    except:
      pass

@fieldwise_init
struct Dim(ImplicitlyCopyable):
  var dim: Int

def maybeDim() raises -> Dim:
   return Dim(3)

@fieldwise_init
struct List(ImplicitlyCopyable):
  def append(self, d: MemExample):
     pass

@fieldwise_init
struct DoNotPropagateErrorStateIntoContinueSet(Movable where False):
  var dims: List
  # CHECK-LABEL: lit.fn @"__init__(
  def __init__(out self, cond: __mlir_type.`!kgen.scalar<bool>`, var list: List) raises:
    # CHECK:     hlcf.loop "_loop_0" {
    # CHECK-NEXT:  hlcf.elif %cond {
    # CHECK-NEXT:    hlcf.yield
    # CHECK-NEXT:  } else {
    # CHECK-NEXT:    hlcf.break "_loop_0"
    # CHECK-NEXT:  }
    while cond:
      list.append(maybeMemExample())
    self.dims = list

def use(x: MemExample): pass

# CHECK-LABEL: lit.fn @"destroyWholeValuesIfLastReferenceWasInLoop
def destroyWholeValuesIfLastReferenceWasInLoop(cond: __mlir_type.`!kgen.scalar<bool>`,
                                              var memPair: MemPair):
   # Part of mempair is used in the loop, but this keeps the entire thing
   # alive during the loop.  The solution here is to destroy memPair immediately
   # before the implicit break out of the loop
   while cond:
     # CHECK:      hlcf.elif %cond {
     # CHECK-NEXT:   hlcf.yield
     # CHECK-NEXT: } else {
     # CHECK-NEXT:   lit.call {{.*}}::@MemPair::@"__deinit__({{.*}}(%memPair)
     # CHECK-NEXT:   hlcf.break "_loop_0"
     # CHECK-NEXT: }
     if cond:
        use(memPair.a)

# CHECK-LABEL: lit.fn @"overwrite
# MOCO-700
def overwrite(y: MemExample, x: Bool) raises:
   var foo = MemPair()
   if x:
   # CHECK: hlcf.elif %{{.*}} {
   # CHECK-NEXT: lit.call {{.*}}::@MemPair::@"__deinit__
      raise Error()
   # CHECK: } else {
   # CHECK-NEXT: [[V7:%.*]] = lit.ref.struct.ger %foo[a]
   # CHECK-NEXT: lit.call {{.*}}@MemExample::@"__deinit__
   # CHECK-NEXT: hlcf.yield
   # CHECK-NEXT: }
   # CHECK: lit.call {{.*}}::@MemPair::@"__deinit__{{.*}}(%foo)
   foo.a = MemExample()


# CHECK-LABEL: lit.fn @"test_if_ownership
# MOCO-721: Test that ownership is transferred and all the move optimizations are
# done.
def test_if_ownership(x: Bool, var a: RegExample, var b: RegExample) -> RegExample:
    # CHECK-NEXT: lit.call {{.*}}__mlir_bool__
    # CHECK-NEXT: [[RES:%.*]] = hlcf.elif
    # CHECK-NEXT:    [[TMP:%.*]] = lit.ref.upcast %a
    # CHECK-NEXT:    hlcf.yield [[TMP]]
    # CHECK-NEXT:  } else {
    # CHECK-NEXT:    [[TMP:%.*]] = lit.ref.upcast %b
    # CHECK-NEXT:    hlcf.yield [[TMP]]{{.*}}
    # CHECK-NEXT:  }

    # Copy into a local temporary.
    # CHECK-NEXT:  [[IRES:%.*]] = lit.ref.immut [[RES]]
    # CHECK-NEXT:  [[RESULT:%.*]] = lit.call {{.*}}__init__{{.*}}copy"

    # Last use of both x and b.
    # CHECK-NEXT:    lit.call {{.*}}__deinit__{{.*}}(%b)
    # CHECK-NEXT:    lit.call {{.*}}__deinit__{{.*}}(%a)

    # CHECK-NEXT:  kgen.return [[RESULT]]
    return a if x else b


struct MyStructWithMarkDestroyed[T: ImplicitlyCopyable & Deinitable](Movable where False):
    var a: Self.T
    var b: Self.T

# CHECK-LABEL: lit.fn @{{.*}}reap
    def reap(deinit self, out result: Self.T):
        # "a" field is never used here so it is destroyed early.
        # CHECK-NEXT: [[AREF:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT: lit.call{{.*}}__deinit__{{.*}}([[AREF]]

        # Transfer operator includes a lit.ownership.use.
        # CHECK-NEXT: [[BREF:%.*]] = lit.ref.struct.ger %self[b]
        # CHECK-NEXT: lit.ownership.use [[BREF]]

        # Rvalue can be moved into the result slot.
        # CHECK-NEXT: lit.call{{.*}}*, "move"{{.*}}__init__
        result = self.b^

        # Full object bit is explicitly destroyed.
        # CHECK-NEXT: kgen.param.constant: none
        # CHECK-NEXT: lit.ownership.mark_destroyed %self
        # CHECK-NEXT: kgen.return


# CHECK-LABEL: lit.fn @"field_sensitive_ref_last_use
def field_sensitive_ref_last_use(var write_state : IntAndOptional):
    # should destroy ALL OF write_state after the copy into msg.
    var msg = write_state.error.value()

    _ = msg.__len__()

    # CHECK: lit.call {{.*}}__init__{{.*}}copy"
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%write_state)

@fieldwise_init
struct IntAndOptional(Movable where False):
    var handle: Int
    var error: Optional[String]


# CHECK: lit.fn @"caught_eh_cleanup
def caught_eh_cleanup():
    # CHECK-NEXT: %eh1 = lit.var.decl "eh1"
    # CHECK-NEXT: lit.try "{{.*}}" {
    try:
      # CHECK-NEXT: [[NORMALRESULT:%.*]] = lit.var.decl

      # CHECK: lit.var.lifetime.start %eh1
      # This function raises, potentially defining %eh1.
      _ = maybeDim()
      # CHECK: [[RAISE:%.*]] = lit.call @ownership::@"maybeDim

      # Check for the error and handle it.
      # CHECK-NEXT: hlcf.if [[RAISE]] {
      # EH is never used, so it can be immediately released.
      # CHECK-NEXT:    lit.call {{.*}}__deinit__{{.*}}(%eh1)
      # CHECK-NEXT:    lit.var.lifetime.end %eh1

      # Normal result is never used
      # CHECK-NEXT: lit.ownership.mark_consumed [[NORMALRESULT]]
      # CHECK-NEXT: lit.var.lifetime.end [[NORMALRESULT]]
      # CHECK-NEXT: lit.try.raise
    except eh1:
      pass

    # CHECK: %eh2 = lit.var.decl "eh2"
    # CHECK-NEXT: lit.try "{{.*}}" {
    try:
      # CHECK-NEXT: [[NORMALRESULT:%.*]] = lit.var.decl

      # CHECK: lit.var.lifetime.start %eh2
      # This function raises, potentially defining %eh2.
      _ = maybeDim()
      # CHECK: [[RAISE:%.*]] = lit.call @ownership::@"maybeDim

      # Check for the error and handle it.
      # CHECK-NEXT: hlcf.if [[RAISE]] {
      # Normal result is never used
      # CHECK-NEXT: lit.ownership.mark_consumed [[NORMALRESULT]]
      # CHECK-NEXT: lit.var.lifetime.end [[NORMALRESULT]]
      # CHECK-NEXT: lit.try.raise

    # CHECK: } except {
    except eh2:
      # CHECK-NEXT: [[EH2:%.*]] = lit.ref.immut %eh2
      # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[EH2]])
      eh2.use()

    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%eh2)
    # CHECK-NEXT: lit.var.lifetime.end %eh2

# CHECK-LABEL: lit.fn @"test_ref_field
# https://linear.app/modularml/issue/MOCO-1251
def test_ref_field(var mem: MemPair):
  # Pointer to subfield.
  var r = Pointer(to=mem.a)

  # Subfield reference keeps entire value alive.
  _ = r[].x
  # CHECK: lit.ref.load
  # CHECK: lit.ownership.use

  # Pointer comparison doesn't access the pointee so the value is already deleted.

  # CHECK: lit.ref.load %r
  # CHECK: lit.ref.load %r
  # CHECK: lit.call {{.*}}__eq__
  _ = r == r
  # CHECK-NEXT: lit.call {{.*}}MemPair::@"__deinit__

def get_inner_ptr(s: String) -> UnsafePointer[UInt8, AnyOrigin[mut=True]]:
  return {}
def use_inner_pointer[origin: Origin[mut=True]](ptr: UnsafePointer[UInt8, origin]): pass

# CHECK-LABEL: lit.fn @"handleAnyLifetime1
def handleAnyLifetime1():
  var str = String()
  # Make sure this keeps alive str until after the call.
  # CHECK: lit.call {{.*}}use_inner_pointer
  use_inner_pointer(get_inner_ptr(str))
  # CHECK-NEXT: lit.call {{.*}}String::@"__deinit__{{.*}}(%str)
  # CHECK-NEXT: lit.var.lifetime.end %str

# CHECK-LABEL: lit.fn @"handleAnyLifetime2
def handleAnyLifetime2():
  var ui8 = UInt8()

  # Make sure this keeps 'ui8' alive until after the call even though
  # the element is trivial.
  # CHECK: lit.call {{.*}}use_inner_pointer
  use_inner_pointer(UnsafePointer(to=ui8))
  # CHECK-NEXT: lit.var.lifetime.end %ui8

# CHECK-LABEL: lit.fn @"handleAnyLifetime3
def handleAnyLifetime3():
    # CHECK-NEXT: lit.call {{.*}}unsafe_dangling
    # CHECK-NEXT: %a_packed_ptr = lit.var.decl
    # CHECK-NEXT: lit.var.lifetime.start %a_packed_ptr
    # CHECK-NEXT: lit.ref.store
    # CHECK-NEXT: lit.var.lifetime.end %a_packed_ptr
    # expected-warning @+1 {{assignment to 'a_packed_ptr' was never used}}
    var a_packed_ptr = UnsafePointer[Int, AnyOrigin[mut=True]].unsafe_dangling()

    # CHECK-NEXT: lit.call {{.*}}unsafe_dangling
    # CHECK-NEXT: lit.var.lifetime.start %a_packed_ptr
    # CHECK-NEXT: lit.ref.store
    # CHECK-NEXT: lit.var.lifetime.end %a_packed_ptr

    # This shouldn't be treated as a use of `a_packed_ptr`
    # expected-warning @+1 {{assignment to 'a_packed_ptr' was never used}}
    a_packed_ptr = UnsafePointer[Int, AnyOrigin[mut=True]].unsafe_dangling()


def take_pack[*Ts: AnyType](*values: *Ts): pass

# CHECK-LABEL: lit.fn @"handleAnyLifetime4
# VariadicPack's need to extend the lifetime in the pack
# https://github.com/modular/mojo/issues/3559
def handleAnyLifetime4():
  var str = String()
  var ptr = UnsafePointer(to=str)

  # Should extend the lifetime of 'str'.
  take_pack(ptr)

  # CHECK: lit.call {{.*}}take_pack
  # CHECK: lit.call {{.*}}__deinit__{{.*}}(%str)
  # CHECK: lit.var.lifetime.end %str


struct A[origin: MutOrigin](Movable where False):
    var data: UnsafePointer[Int, Self.origin]
    def __init__(out self):
        self.data = UnsafePointer[Int, Self.origin].unsafe_dangling()
    def __deinit__(deinit self): pass

def use_int(a: Int): pass

# CHECK-LABEL: lit.fn @"handleAnyLifetime5
def handleAnyLifetime5():
    # lit.ref.load needs to extend the lifetime of A.
    var a = A[AnyOrigin[mut=True]]()
    # CHECK: [[INT_REF:%.*]] = {{.*}}Pointer::@"__getitem__
    # CHECK-NOT: lit.call {{.*}}__deinit__
    # CHECK: lit.ref.load [[INT_REF]]
    # CHECK: lit.call {{.*}}__deinit__{{.*}}(%a)
    use_int(a.data[0])


# This checks that the Mojo parser successfully folds the initializer call for
# origin into a struct attr, which is important for lifetime analysis to be able
# to reason about these.

# CHECK-LABEL: lit.fn @"test_origin_ctor_folding
def test_origin_ctor_folding[orig1: MutOrigin](abcdef: A[orig1]):
    # CHECK-NEXT: lit.alias.decl *"x{{.*}}:origin<false> *"abcdef`1">>
    comptime x = origin_of(abcdef)

    # MOCO-1467: Origin type equality problem.
    # CHECK-NEXT: lit.alias.decl *"y{{.*}}:origin<true> *"orig1._mlir_origin`{{.*}}>>
    comptime y = Origin[_mlir_origin = orig1._mlir_origin]()

    # Check that origin_of works on origins as well as MValues.
    # CHECK-NEXT: lit.alias.decl *"o2{{.*}}:origin<false> {*"abcdef`1", (mutcast mut *"orig1._mlir_origin`"){{.*}}>>
    comptime o2 = origin_of(orig1, abcdef)

def useMemory(a: MemExample): pass

# CHECK-LABEL: lit.fn @"testConds1
def testConds1(cond: __mlir_type.`!kgen.scalar<bool>`, reg: RegExample, i: Int):
  # Implicit conversions.
  # Mojo Issue #49: https://github.com/modular/mojo/issues/49

  # CHECK-NEXT: hlcf.elif %cond -> !RegExample {
  # CHECK:        [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}copy"
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = reg if cond else i

  # CHECK: hlcf.elif %cond -> !RegExample {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK:        [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}copy"
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = i if cond else reg

  _ = reg
  _ = i

# Memory only conds. Issue (#13379)
# CHECK-LABEL: lit.fn @"testConds2
def testConds2(cond: __mlir_type.`!kgen.scalar<bool>`, a: MemExample, b: MemExample) -> MemExample:
  # CHECK:      [[IF:%.*]] = hlcf.elif %cond
  # CHECK-NEXT:   [[TMP:%.*]] = lit.ref.upcast %a
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.ref.upcast %b
  # CHECK-NEXT:   hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }
  # CHECK-NEXT: lit.call {{.*}}useMemory{{.*}}([[IF]])
  useMemory(a if cond else b)

  # Handle a local temp correctly.
  # TODO(ternary memory optimization): The moveinit doesn't seem necessary,
  # could direct construct into the dest and elide the temp.

  # CHECK-NEXT: [[IF:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: hlcf.elif %cond
  # CHECK-NEXT:   [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT:   lit.var.lifetime.start [[TMP]]
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}([[TMP]])
  # CHECK-NEXT:   lit.var.lifetime.start [[IF]]
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}move"

  # CHECK-NEXT:   lit.var.lifetime.end [[TMP]]
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.var.lifetime.start [[IF]]
  # CHECK-NEXT:   lit.memcpy %b, [[IF]]
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  # CHECK-NEXT: [[IFI:%.*]] = lit.ref.immut [[IF]]
  # CHECK-NEXT: lit.call {{.*}}useMemory{{.*}}([[IFI]])
  useMemory(MemExample() if cond else b)
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[IF]])
  # CHECK-NEXT: lit.var.lifetime.end [[IF]]


  # CHECK-NEXT: [[IF:%.*]] = hlcf.elif %cond
  # CHECK-NEXT:   [[TMP:%.*]] = lit.ref.upcast %a
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.ref.upcast %b
  # CHECK-NEXT:   hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }
  # CHECK-NEXT:   lit.memcpy [[IF]], %__result__
  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
  return a if cond else b

# CHECK-LABEL: lit.fn @"testConds3
def testConds3(cond: __mlir_type.`!kgen.scalar<bool>`, var a: MemExample, var b: MemExample,
              var m: RegExample, var n: RegExample):
  # CHECK-NEXT: %t1 = lit.var.decl
  # CHECK-NEXT: hlcf.elif %cond
  # CHECK-NEXT:    lit.call {{.*}}__deinit__{{.*}}(%b)
  # CHECK-NEXT:    lit.ownership.use %a
  # CHECK-NEXT:    lit.var.lifetime.start %t1
  # CHECK-NEXT:    lit.call {{.*}}__init__{{.*}}move"
  # CHECK-NEXT:    hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%a)
  # CHECK-NEXT:    lit.ownership.use %b
  # CHECK-NEXT:    lit.var.lifetime.start %t1
  # CHECK-NEXT:    lit.call {{.*}}__init__{{.*}}"{{.*}}(%b, %t1){{.*}}*, "move"
  # CHECK-NEXT:    hlcf.yield
  # CHECK-NEXT: }
  var t1 = a^ if cond else b^

  # CHECK-NEXT: [[IF:%.*]] = hlcf.elif %cond
  # CHECK-NEXT:    lit.call {{.*}}RegExample::@"__deinit__{{.*}}(%n)
  # CHECK-NEXT:    lit.ownership.use %m
  # CHECK-NEXT:    [[TMP:%.*]] = lit.load.consume %m
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:    lit.call {{.*}}RegExample::@"__deinit__{{.*}}(%m)
  # CHECK-NEXT:    lit.ownership.use %n
  # CHECK-NEXT:    [[TMP:%.*]] = lit.load.consume %n
  # CHECK-NEXT:    hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }
  # CHECK-NEXT: %t2 = lit.var.decl
  # CHECK-NEXT: lit.var.lifetime.start %t2
  # CHECK-NEXT: lit.ref.store [[IF]], %t2
  var t2 = m^ if cond else n^

  consume(t1^)
  consume(t2^)

# CHECK-LABEL: lit.fn @"my_min1
# CHECK-SAME: !lit.ref<:meta<!Int> #alias_Int, mut=and(*"x_is_mut`", *"y_is_mut`2"), {(mutcast mut=*"x_is_mut`", *"x_is_origin`1"), (mutcast mut=*"y_is_mut`2", *"y_is_origin`3")}>
def my_min1(cond: __mlir_type.`!kgen.scalar<bool>`, ref x: Int, ref y: Int) -> ref [x, y] Int:
  # CHECK-NEXT: [[IF:%.*]] = hlcf.elif %cond
  # CHECK-NEXT:    [[TMP:%.*]] = lit.ref.upcast %x
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:    [[TMP:%.*]]  = lit.ref.upcast %y
  # CHECK-NEXT:    hlcf.yield [[TMP]]{{.*}}
  # CHECK-NEXT: }

  # CHECK-NEXT: [[RET:%.*]] = kgen.rebind [[IF]]
  # CHECK-NEXT: kgen.return [[RET]]
  return x if cond else y

# CHECK-LABEL: lit.fn @"my_min2
def my_min2[T: AnyType](ref a: T, ref b: T) -> ref [a, b] T:
    return a

# CHECK-LABEL: lit.fn @"test_min2
# https://github.com/modular/mojo/issues/3815
def test_min2(a: String):
    # CHECK: lit.call {{.*}}String::@"__init__
    var x = String()
    # CHECK: lit.call {{.*}}String::@"__init__
    var y = String()
    # CHECK: [[REF:%.*]] = lit.call {{.*}}my_min2
    # CHECK-NEXT: [[SLICE:%.*]] = lit.call {{.*}}StringSpan::@"__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call {{.*}}String::@"__iadd__{{.*}}([[REF]], [[SLICE]])
    my_min2(x, y) += a
    # CHECK-NEXT: lit.call {{.*}}String::@"__deinit__{{.*}}(%y)
    # CHECK-NEXT: lit.var.lifetime.end %y
    # CHECK-NEXT: lit.call {{.*}}String::@"__deinit__{{.*}}(%x)
    # CHECK-NEXT: lit.var.lifetime.end %x

# MOCO-1500: Can't take origin of read-only String arg
def origin_of_def_arg(a: String) raises:
    _ = origin_of(a)

# MOCO-1542: Need to rebind field type when checking size.
@fieldwise_init
struct MyParameterizedField[T: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable, Deinitable):
  var a: Self.T
  var b: Self.T

def use_parameterized_field():
  var s = MyParameterizedField[Dim](Dim(8), Dim(3))
  var litref = __get_mvalue_as_litref(s.b)
  var rebind = __mlir_op.`kgen.rebind`
    [_type=Pointer[Int, origin_of(s.b)]._mlir_lit_ref](litref)
  # expected-warning @+1 {{assignment to 'mvalue' was never used}}
  var mvalue = __get_litref_as_mvalue(rebind)

# MOCO-1558: Failure handling sub-type elements
# CHECK-LABEL: OuterStruct
struct OuterStruct(Movable where False):
    var outers_field: RefResultStruct
    # CHECK: lit.fn @"__deinit__
    def __deinit__(deinit self):
        # CHECK: lit.call {{.*}}use(::String)
        use(self.outers_field.x)
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[outers_field]
        # CHECK-NEXT: lit.call {{.*}}RefResultStruct::@"__deinit__{{.*}}([[TMP]])
        # CHECK: lit.ownership.mark_destroyed %self

struct RefResultStruct(Movable where False):
  var x: String
  def __init__(out self):
    self.x = String()

  def method(self) -> ref [self.x] String:
      return self.x

def use(a: String): pass

# https://github.com/modular/modular/issues/4163
#  BUG] Mojo compiler error when two instance variables of type PythonObject are initialized by Python.import_module in a struct's __init__()
struct SomeStruct(Movable where False):
    var test_agent: SomeValue[Int]
    def __init__(out self) raises:
        self.test_agent = SomeValue(123)

struct SomeValue[T: ImplicitlyCopyable & Deinitable](Movable where False):
    var value: Self.T
    var name: String
    var tmp: Int

    def __init__(out self, value: Self.T) raises:
        self.value = value
        self.name = "example"
        self.tmp = 1 #<- remove this field and it works

# This triggered a bug handling parameterized types with substitutions, reported
# on discord.
# CHECK-LABEL: ParametricTask
struct ParametricTask[T1: Movable & Deinitable,
                      T2: Movable & Deinitable](Movable):
    var t1: Self.T1
    var t2: Self.T2
    def __init__(out self, var t1: Self.T1, var t2: Self.T2):
        self.t1 = t1^
        self.t2 = t2^
    def concat(var self, var other: Self) -> ParametricTask[Self, Self]:
        return ParametricTask(self^, other^)

    # Verify that deinit causes the individual fields to be destroyed.
    # CHECK: lit.fn @"explicit_destroy
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[t1]
    # CHECK-NEXT: lit.call{{.*}}__deinit__
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[t2]
    # CHECK-NEXT: lit.call{{.*}}__deinit__
    def explicit_destroy(deinit self):
      pass

    # Verify that var causes the whole thing to be destroyed.
    # CHECK: lit.fn @"var_method
    # CHECK-NEXT: lit.call {{.*}}ParametricTask::@"__deinit__
    def var_method(var self):
      pass


# https://github.com/modular/modular/issues/4518
def issue4518():
    var val1 = 0

    try:
        val1 = issue4518_fn_that_raises()
    except:
        pass

    if val1:  # use of uninitialized value 'val1'
        pass

def issue4518_fn_that_raises() raises -> Int: return 0


# MOCO-2918: Default traits methods don't work if they have variadic packs
# Test direct passing of the variadic pack.
trait HasBarWVariadicPack:
    def method_with_pack[*Ts: AnyType](mut self, *args: *Ts):
        pass
# CHECK-LABEL: lit.struct.decl @InheritDefaultFromHasBarWVariadicPack
struct InheritDefaultFromHasBarWVariadicPack(HasBarWVariadicPack, Movable where False):
    pass

# CHECK: lit.fn @"method_with_pack{{.*}}(%self: !lit.ref<!InheritDefaultFromHasBarWVariadicPack, mut *"0_unnamed`"> mut,
# CHECK-SAME: %args: !lit.ref<!lit.struct<#VariadicPack
# CHECK-NEXT: lit.call {{.*}}@"method_with_pack{{.*}}(%self, %args)
# CHECK-NEXT: kgen.return %0

# https://github.com/modular/modular/issues/5722
# `__deinit__` incorrectly runs when `__init__` raises before all fields are initialized.
# CHECK-LABEL: lit.struct.decl @TestRaiseFromInit
struct TestRaiseFromInit(Movable where False):
    var x: Int
    var y: String

  # CHECK: lit.fn @"__init__
    def __init__(out self) raises Int:
        # CHECK-NEXT: %0 = lit.ref.struct.ger %self[y]
        # CHECK-NEXT: lit.call {{.*}}String::@"__init__{{.*}}(%0)
        # CHECK-NEXT: lit.call {{.*}}String::@"__deinit__{{.*}}(%0)
        # CHECK-NEXT: kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
        self.y = String()
        raise 42

# This is a type whose members are sometimes trivial!
struct OccasionallyTrivial[a: Int](Movable where False):
    # This is trivial when a == 0.
    def __deinit__(deinit self):
        pass

    comptime __del__is_trivial: Bool = Self.a == 0

    def use(ref self): pass

# CHECK-LABEL: lit.fn @"test_is_trivial
def test_is_trivial(var a0: OccasionallyTrivial[0],
                    var a1: OccasionallyTrivial[1]):

    a0.use()
    # CHECK-NEXT: lit.call {{.*}}OccasionallyTrivial::@"use{{.*}}(%a0)
    # CHECK-NOT: __deinit__

    # CHECK-NEXT: lit.call {{.*}}marker
    marker()

    a1.use()
    # CHECK-NEXT: lit.call {{.*}}OccasionallyTrivial::@"use{{.*}}(%a1)
    # CHECK-NEXT: lit.call {{.*}}OccasionallyTrivial::@"__deinit__{{.*}}(%a1)

    # CHECK-NEXT: kgen.param.constant: none


# CHECK-LABEL: lit.struct.decl @TestConditionallyLinearType
struct TestConditionallyLinearType[T: Movable](
    Deinitable where conforms_to(T, Deinitable), Movable where False
):
    var data: Self.T

# CHECK: lit.fn @"__deinit__
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[data]
# CHECK-NEXT: lit.call{{.*}}__deinit__{{.*}}([[TMP]])
# CHECK-NEXT: kgen.param.constant: none

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        pass


# `T` must not already be bound by `Deinitable`, or the conditional
# conformance folds away and the synthesized dtor comes out unconditional.
# CHECK-LABEL: lit.struct.decl @TestSynthesizedConditionalDtor
struct TestSynthesizedConditionalDtor[T: Movable](
    Deinitable where conforms_to(T, Deinitable), Movable where False
):
    var data: Self.T

# CHECK: lit.fn @"__deinit__{{.*}}TestSynthesizedConditionalDtor{{.*}}conforms_to(:!AnyType_Movable T, {{.*}}Deinitable{{.*}}synthetic

# MOCO-4059: Conditionally linear type with concrete struct.
# CHECK-LABEL: lit.struct.decl @AConditionallyLinearType
@explicit_destroy("I am conditionally linear!")
@fieldwise_init
struct AConditionallyLinearType[T: AnyType](
    Copyable,
    Deinitable where conforms_to(T, Deinitable),
):

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
      pass

    # CHECK-LABEL: lit.fn @"example_method
    def example_method(self) where conforms_to(Self.T, Deinitable):
        var _copy = self.copy()
        # CHECK-NEXT: %_copy = lit.var.decl
        # CHECK-NEXT: lit.var.lifetime.start %_copy
        # CHECK-NEXT: lit.call {{.*}}@"copy
        # CHECK-NEXT: lit.call {{.*}}@AConditionallyLinearType::@"__deinit__
        # CHECK-NEXT: lit.var.lifetime.end %_copy



# MOCO-3880: The destructor for one value can extend the lifetime of another value.
@fieldwise_init
struct Driver(Movable where False):
    def __deinit__(deinit self): pass

@fieldwise_init
struct Event[origin: ImmOrigin](Movable where False):
    def __deinit__(deinit self): pass

def record_event(driver: Driver) -> Event[origin_of(driver)]:
    return {}

# CHECK-LABEL: lit.fn @"test_mojo_3880
def test_mojo_3880():
    var driver = Driver()
    _ = record_event(driver)

    # CHECK: lit.call {{.*}}record_event
    # CHECK-NEXT: lit.ownership.use [[EVENT:%.*]] : !lit.ref<
    # CHECK-NEXT: lit.call {{.*}}Event::@"__deinit__{{.*}}([[EVENT]])
    # CHECK-NEXT: lit.var.lifetime.end [[EVENT]]
    # CHECK-NEXT: lit.call {{.*}}Driver::@"__deinit__{{.*}}(%driver)
    # CHECK-NEXT: lit.var.lifetime.end %driver
