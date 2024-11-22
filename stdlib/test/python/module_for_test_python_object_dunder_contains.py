class Class_no_iterable_no_contains:
    x = 1

class Class_no_iterable_but_contains:
    x = 123
    def __contains__(self, rhs):
        return rhs == self.x

class Class_iterable_no_contains:
    def __init__(self):
        self.data = [123, 456]
    
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.data):
            raise StopIteration
        else:
            tmp = self.data[self.i]
            self.i += 1
            return tmp
