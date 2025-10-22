import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    rgb = rgb.astype(float)
    return((rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) * 1/3)

def LUGauss(A):
    l, c = A.shape
    L = np.eye(l)

    for i in range(l):
        pivo = A[i, i]
        for j in range(i+1, l):
            m = A[j, i]/pivo
            A[j, :] = A[j, :] - m*A[i, :]
            L[j, i] = m
    U = A        
    return L, U

