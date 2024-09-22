from memory import Boxed

@value
struct ObservableDel(CollectionElement):
    var target: UnsafePointer[Bool]

    fn __init__(inout self, *, other: Self):
        self = other

    fn __del__(owned self):
        self.target.init_pointee_move(True)

def test_basic_ref():
    var b = Boxed(1)
    assert_equal(1, b[])

def test_trivial_copy():
    var b = Boxed(1)
    var b2 = b

    assert_equal(1, b[])
    assert_equal(1, b2[])

def test_basic_ref_mutate():
    var b = Boxed(1)
    assert_equal(1, b[])

    b[] = 2

    assert_equal(2, b[])

def test_basic_del():
    var deleted = False
    var b = Boxed(ObservableDel(UnsafePointer.address_of(deleted)))

    assert_false(deleted)
    
    _ = b

    assert_true(deleted)

def fail():
    assert_true(false)

def main():
    test_basic_ref()
    test_trivial_copy()
    test_basic_ref_mutate()
    test_basic_del()
    fail()
