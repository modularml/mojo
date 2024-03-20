# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from abc import ABC, abstractmethod


class Person(ABC):
    pass


class Foo:
    def __init__(self, bar):
        self.bar = bar


class AbstractPerson(ABC):
    @abstractmethod
    def method(self):
        ...


def my_function(name):
    return f"Formatting the string from Lit with Python: {name}"


def eat_it_all(veggie, *args, fruit, **kwargs):
    return f"{veggie} {args} fruit={fruit} {kwargs}"
