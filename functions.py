import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    rgb = rgb.astype(float)
    return((rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) * 1/3)

def LUGauss(A):
    l, c = A.shape
    L = np.eye(l)
    Amod = np.copy(A)

    for i in range(l):
        pivo = Amod[i, i]
        for j in range(i+1, l):
            m = Amod[j, i]/pivo
            Amod[j, :] = Amod[j, :] - m*Amod[i, :]
            L[j, i] = m
    U = Amod        
    return L, U

def Criterio_linhas(A):
    A_abs = np.abs(A)
    diag = np.diag(A_abs)

    soma_linhas = np.sum(A_abs, axis=1)
    soma_resto = soma_linhas - diag

    alfa = soma_resto/diag
    alfa_max = np.max(alfa)

    if alfa_max < 1:
        return True, alfa_max
    else:
        return False, alfa_max

def Criterio_colunas(A):
    A_abs = np.abs(A)
    diag = np.diag(A_abs)

    soma_colunas = np.sum(A_abs, axis = 0)
    soma_resto = soma_colunas - diag

    alfa = soma_resto/diag
    alfa_max = np.max(alfa)

    if alfa_max < 1:
        return True, alfa_max
    else:
        return False, alfa_max

def Sassenfeld(A):
    l, c = A.shape

    A_abs = np.abs(A)
    Beta = np.zeros(l)
    diag = np.diag(A_abs)

    for i in range(l):
        soma_antigo = 0.0
        soma_novo = 0.0

        for j in range(l):
            if i != j:
                if j < i:
                    soma_antigo += A_abs[i, j] * Beta[j]
                else:
                    soma_novo += A_abs[i, j]
        Beta[i] =  (soma_antigo + soma_novo)/diag[i]

    beta_max = np.max(Beta)
    if beta_max < 1:
        return True, beta_max
    else:
        return False, beta_max

def Jacobi(A, b, x0, N, T):
    l, c = A.shape
    k = 0
    xk1 = np.copy(x0)

    flag1, alfa1 = Criterio_linhas(A)
    flag2, alfa2 = Criterio_colunas(A)

    if flag1:
        print(f"Convergência garantida pelo critério das linhas, alfa = {alfa1:.4f}\n")
    elif flag2:
        print(f"Convergência garantida pelo critério das colunas, alfa = {alfa2:.4f}\n")
    else:
        print("Convergência não garantida.\n")

    while k < N:
        xk = np.copy(xk1)
        for i in range(l):
            soma = 0.0
            for j in range(l):
                if i != j:
                    soma += A[i, j] * xk[j]
            
            xk1[i] = (b[i] - soma) / A[i, i]

        ek1 = xk1 - xk
        erro = np.linalg.norm(ek1, ord=np.inf)
        k += 1

        print(f"Iteração {k}: Solução: {xk1}, Erro:{erro:.4f}")

        if erro < T:
            print("\n Erro atingido pela tolerância.")
            break
    
    if k == N and erro >= T:
        print(f"\nNúmero máximo de iterações ({N}) atingido.")

    return xk1

def GaussSeidel(A, b, x0, N, T):
    l, c = A.shape
    k = 0
    xk1 = np.copy(x0)

    flag1, alfa1 = Criterio_linhas(A)
    flag2, alfa2 = Sassenfeld(A)
    
    if flag1:
        print(f"Convergência garantida pelo critério das linhas, alfa = {alfa1:.4f}\n")
    elif flag2:
        print(f"Convergência garantida pelo critério de Sassenfeld, alfa = {alfa2:.4f}\n")
    else:
        print("Convergência não garantida.\n")

    while k < N:
        xk = np.copy(xk1)
        for i in range(l):
            soma = 0.0
            for j in range(l):
                if i != j:
                    if j < i:
                        soma += A[i, j] * xk1[j]
                    else:
                        soma += A[i, j] * xk[j]

            xk1[i] = (b[i] - soma) / A[i, i]

        ek1 = xk1 - xk
        erro = np.linalg.norm(ek1, ord=np.inf)
        k += 1

        print(f"Iteração {k}: Solução: {xk1}, Erro:{erro:.4f}\n")

        if erro < T:
            print("Erro menor que a tolerância.\n")
            break
    
    if k == N and erro >= T:
        print(f"\nNúmero máximo de iterações ({N}) atingido.")

    return xk1