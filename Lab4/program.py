import numpy as np
import matplotlib.pyplot as plt
import random
import math as m

def val_by_chances(chances=[0.5, 0.5], equal=False):
    if(equal):
        limit = int(chances[0]*100)*len(chances) -1
    else:
        limit = 99
        
    choice = []
    for i in range(len(chances)):
        choice = np.concatenate((choice, np.full(int(chances[i]*100), i)))

    # print(limit, choice)
    return int(choice[random.randint(0,limit)])


# coefs = [[a, b, c, d, e, f], [a, b, c, d, e, f], ...]
def xy_set_differ(coefs, chance=[], iter=10000, angular=False):
    if(len(chance)==0):
        chance = [1/len(coefs)]*len(coefs)
        equal = True
    else:
        equal = False
        
    values = np.array([[0,0,0]])
    if(not angular):
        for i in range(iter):
            choice = val_by_chances(chance, equal=equal)
            x = (values[-1])[0]*coefs[choice][0] + (values[-1])[1]*coefs[choice][1] + coefs[choice][4]
            y = (values[-1])[0]*coefs[choice][2] + (values[-1])[1]*coefs[choice][3] + coefs[choice][5]
            values = np.vstack((values, [x,y,0]))
    else:
        for i in range(iter):
            choice = val_by_chances(chance, equal=equal)
            x = (values[-1])[0]*coefs[choice][0]*m.cos(coefs[choice][2]) - (values[-1])[1]*coefs[choice][1]*m.sin(coefs[choice][3]) + coefs[choice][4]
            y = (values[-1])[0]*coefs[choice][0]*m.sin(coefs[choice][2]) + (values[-1])[1]*coefs[choice][1]*m.cos(coefs[choice][3]) + coefs[choice][5]
            values = np.vstack((values, [x,y,0]))
    return values

from matplotlib.animation import FuncAnimation, PillowWriter

def paint_with_anima(coefs, chance=[], iter=10000, step=100, name='name_not_given'):
    values = xy_set_differ(coefs, chance, iter)
    fig, ax = plt.subplots()
    ax.scatter(values[:, 0], values[:, 1], s=0.1)
    def update(frame):
        frame = frame * step
        ax.clear()
        x, y = values[:frame, 0], values[:frame, 1]
        ax.scatter(x, y, s=1)
    
    ani = FuncAnimation(fig=fig, func=update, frames=iter//step, interval=30)
    ani.save(name+'.gif', writer=PillowWriter(fps=30))
    plt.show()



coefs = np.array([[0.14, 0.01, 0, 0.51, -0.08, -1.31],
                  [0.43, 0.52, -0.45, 0.5, 1.49, -0.75],
                  [0.45, -0.49, 0.47, 0.47, -1.62, -0.74],
                  [0.49, 0, 0, 0.51, 0.02, 1.62]])


paint_with_anima(coefs, iter=10000, name='maple leaf')