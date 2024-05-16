import numpy as np
import matplotlib.pyplot as plt
import copy
import math

base = 5

not_included_x = [1, 3]
not_included_y = [0, 2, 4]

xs = np.linspace(0,base-1, base, dtype=int)
ys = np.linspace(0,base-1, base, dtype=int)
xs = np.delete(xs, not_included_x)
ys = np.delete(ys, not_included_y)

M = np.zeros((base, base), dtype=int)
N=0
for x in xs:
    for y in ys:
        M[y][x] = 1
        N+=1
S = 1 - M
r = M.shape[0]
dim = math.log(N)/math.log(r)
plt.title(f'Fractal dimetion = {np.floor(dim)} < {dim} < {np.ceil(dim)}')
plt.imshow(S, cmap='gray')
plt.show()
plt.pause(2)

# M = np.ones((1, 1))  # Initial matrix
print(M)
for _ in range(2):
    M = M.tolist()
    M_copy = copy.deepcopy(M)
    print('M_LIST:', M_copy)
    N=0
    for i in range(len(M)):
        for j in range(len(M[0])):
            if(M[i][j] == 1):
                N+=1
                M[i][j] = np.array(M_copy)
            else:
                M[i][j] = np.zeros_like(M_copy)
    M = np.block(M)
    S = 1 - M
    r = M.shape[0]
    dim = math.log(N)/math.log(r)
    plt.title(f'Fractal dimetion = {np.floor(dim)} < {dim} < {np.ceil(dim)}')
    plt.imshow(S, cmap='gray')
    plt.show()
    plt.pause(2)