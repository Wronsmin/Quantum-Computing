import numpy as np 

def fA(x, A, B):
    return (x - A) / B

def fB(x, A, B):
    return (x - A) ** 2 / B ** 3

def fBA(x, A, B):
    return 2 * (A - x) / B ** 3

def fAB(x, A, B):
    return 2 * (A - x) / B ** 3

def fAA(B):
    return -1 / B ** 2

def fBB(x, A, B):
    return -3 * (x - A) ** 2 / B ** 4