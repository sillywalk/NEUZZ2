from pathlib import Path, PosixPath


import sys


class Test:
    def __init__(self):
        self.ss = 0
        print(self.ss)

    def func(self) -> None:
        del self.ss
        self.ss = 1
        print(self.ss)


if __name__ == "__main__":
    t = Test()
    t.func()
