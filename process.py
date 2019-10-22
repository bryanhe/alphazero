#!/usr/bin/env python3

import multiprocessing
import numpy as np
def f(i, a):
    print(i)

def main(n):
    arr = multiprocessing.Array("b", np.zeros(10, np.bool))
    x = [{} for _ in range(n)]
    pool = multiprocessing.Pool(processes=4)
    pool.map(f, [(i, arr) for i in range(10)])

if __name__ == "__main__":
    main(10)
