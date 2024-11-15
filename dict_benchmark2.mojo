from collections import Dict
from time import now
from random import *

alias iteration_size = 2048
def main():
    var result: Int=0
    var start = now()
    var stop = now()

    small2 = Dict[Int,Int]()
    start = now()
    for x in range(100):
        for i in range(iteration_size):
            small2[i] = i
        for i in range(iteration_size):
            result += small2[i]
    stop = now()
    print("Int dicts:", stop-start, "ns", result, "rows")

    small3 = Dict[String,String]()
    start = now()
    for x in range(100):
        for i in range(iteration_size):
            small3[str(i)]=str(i)
        for i in range(iteration_size):
            result += len(small3[str(i)])
    stop = now()
    print("String dicts:", stop-start, "ns", result, "rows")
