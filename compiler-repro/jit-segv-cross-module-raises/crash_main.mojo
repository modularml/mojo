# crash_main.mojo — imports and CALLS crash_lib.make().
from crash_lib import S, make


def main() raises:
    var s = make(65536)
    print("survived", s.a)
