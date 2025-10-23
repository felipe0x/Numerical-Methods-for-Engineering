import matplotlib.pyplot as plt
import numpy as np
import functions as func

img_orig = plt.imread('imagem.jpeg')

# plt.imshow(img_orig)
# plt.title('Imagem original')
# plt.show()

img = plt.imread('imagem540.jpeg')
img_gray = func.rgb2gray(img)

# plt.imshow(img)
# plt.title('Imagem cortada')
# plt.show()

# plt.imshow(img_gray, cmap='gray')
# plt.title('Imagem em tons de cinza sem ruído)
# plt.show()

t = img_gray.shape
imgruido = img_gray + np.random.rand(t[0], t[1])

# plt.imshow(imgruido, cmap='gray')
# plt.title('Imagem com ruído)
# plt.show()

# L, U = func.LUGauss(imgruido)
# plt.imshow(np.uint8(L), cmap="gray")
# plt.show()

# plt.imshow(np.uint8(U), cmap="gray")
# plt.show()

# A = L @ U
# print (A.shape, img.shape)
# plt.imshow(np.uint8(A), cmap="gray")
# plt.show()

b = np.array([1784, 1781, 1060, 1793, 1766, 1714, 1695, 1763,
              1713, 1149, 1246, 1096])

C = np.array([
    [10, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 10, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 10, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 10, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 10, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 10, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 10, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 10, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 10, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 10, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 10, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 10]
])

l = C.shape[0]
x0 = np.zeros(l)

s1 = func.GaussSeidel(C, b, x0, 10, 0.01)
s2 = func.Jacobi(C, b, x0, 10, 0.01)

print(s1)
print(s2)

s1 = [chr(int(np.round(v))) for v in s1]
s2 = [chr(int(np.round(v))) for v in s2]

print(s1)
print(s2)