# Lab4 Fractals using iterating systems
import numpy as np
import matplotlib.pyplot as plt
import random


# def xy_set(a,b,c,d,e,f, iter=1000000):
#     set = []
#     set.append([0,0])
#     for i in range(iter):
#         x, y = set[-1][0], set[-1][1]
#         x, y = a*x + b*y + e, c*x + d*y + f
#         set.append([x,y])
    
#     print(set)
#     return set


def val_by_chances(chances=[0.5, 0.5]):
    choice = []
    for i in range(len(chances)):
        choice = np.concatenate((choice, np.full(int(chances[i]*100), i)))
    return int(choice[random.randint(0,99)])


# coefs = [[a, b, c, d, e, f], [a, b, c, d, e, f], ...]
def xy_set_differ(coefs, chance=[], iter=10000):
    if(len(chance)==0):
        chance = [1/len(coefs)]*len(coefs)
        print(chance)
    values = np.array([[0,0,0]])
    for i in range(iter):
        choice = val_by_chances(chance)
        x = (values[-1])[0]*coefs[choice][0] + (values[-1])[1]*coefs[choice][1] + coefs[choice][4]
        y = (values[-1])[0]*coefs[choice][2] + (values[-1])[1]*coefs[choice][3] + coefs[choice][5]

        values = np.vstack((values, [x,y,0]))
    # print(values)
    return values


# maple leaf
coefs = np.array([[0.14, 0.01, 0, 0.51, -0.08, -1.31],
                  [0.43, 0.52, -0.45, 0.5, 1.49, -0.75],
                  [0.45, -0.49, 0.47, 0.47, -1.62, -0.74],
                  [0.49, 0, 0, 0.51, 0.02, 1.62]])

vals = xy_set_differ(coefs, iter=50000)
plt.scatter(vals[:, 0], vals[:, 1], s=0.1)
plt.show()


# spiral
coefs = np.array([[0.787879, -0.121212, 0.181818],
                [-0.424242, 0.257576, -0.136364],
                [0.242424, 0.151515, 0.090909],
                [0.859848, 0.053030, 0.181818],
                [1.758647, -6.721654, 6.086107],
                [1.408065, 1.377236, 1.568035]]).T

chances = [0.9, 0.05, 0.05]

vals = xy_set_differ(coefs, chances, iter=20000)
plt.scatter(vals[:, 0], vals[:, 1], s=0.1)
plt.show()


# Maplebrot
coefs = np.array([[0.2020, 0.1380],
                [-0.8050, 0.6650],
                [-0.6890, -0.5020],
                [-0.3420, -0.2220],
                [-0.3730, 0.6600],
                [-0.6530, -0.2770]]).T

vals = xy_set_differ(coefs)
plt.scatter(vals[:, 0], vals[:, 1], s=0.1)
plt.show()
