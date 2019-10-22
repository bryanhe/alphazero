#!/usr/bin/env python3

import threading

def main(n):
    x = [{} for _ in range(n)]
    def f(i):
        print(i)
        x[i]["Hello"] = "World"
    t = [threading.Thread(target=f, args=(i, )) for i in range(n)]
    for i in t:
        i.start()
    for i in t:
        i.join()
    print(x)

if __name__ == "__main__":
    main(10)
