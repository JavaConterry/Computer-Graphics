import matplotlib.pyplot as plt
import numpy as np
import math as m

vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])
vertices = np.stack((vertices[:, 0],vertices[:, 1], vertices[:, 2], np.ones(len(vertices))))

edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

x_start, x_end = np.random.uniform(0, 10, 2);print(x_start)
y_start, y_end = np.random.uniform(0, 10, 2);print(y_start)
z_start, z_end = np.random.uniform(0, 10, 2);print(z_start)

xs = np.linspace(x_start, x_end, 100)
ys = np.linspace(y_start, y_end, 100)
zs = np.linspace(z_start, z_end, 100)

line_vector = np.array((x_end-x_start, y_end-y_start, z_end-z_start, 1))

line = np.column_stack((xs, ys, zs, np.ones((100, 1))))

M_transpose_to_0 = np.array([[1, 0, 0, -x_start],
                            [0, 1, 0, -y_start],
                            [0, 0, 1, -z_start],
                            [0, 0, 0, 1]])

cosxi = line_vector[2]/m.sqrt(line_vector[1]**2+line_vector[2]**2)
sinxi = line_vector[1]/m.sqrt(line_vector[1]**2+line_vector[2]**2)

R_x = np.array([[1,     0,    0,   0],
                [0, cosxi, sinxi,  0],
                [0, -sinxi, cosxi, 0],
                [0,     0,      0, 1]])

l_m_vector = line_vector.dot(M_transpose_to_0.T).dot(R_x)
print(l_m_vector)
costeta = l_m_vector[2]
sinteta = l_m_vector[0]

R_y = np.array([[costeta, 0, -sinteta, 0],
                [0,       1,        0, 0],
                [sinteta, 0,  costeta, 0],
                [0,       0,        0, 1]])

#rotate by φ=pi/3
fi = m.pi/3
cosfi = m.cos(fi)
sinfi = m.sin(fi)

R_z = np.array([[cosfi, -sinfi,  0, 0],
                [sinfi, cosfi, 0, 0],    
                [0,         0,  1, 0],
                [0,         0,  0, 1]])

R_y_back = np.array([[costeta, 0, sinteta, 0],
                [0,       1,        0, 0],
                [-sinteta, 0,  costeta, 0],
                [0,       0,        0, 1]]) 

R_x_back = np.array([[1, 0, 0, 0],
                [0, cosxi, -sinxi, 0],
                [0, sinxi, cosxi, 0],
                [0,     0,     0, 1]])

M_transpose_from_0 = np.array([[1, 0, 0, x_start],
                    [0, 1, 0, y_start],
                    [0, 0, 1, z_start],
                    [0, 0, 0, 1]])

print(vertices.T)

result_rotation = (vertices.T).dot(M_transpose_to_0.T).dot(R_x).dot(R_y).dot(R_z).dot(M_transpose_from_0.T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#plot lines
ax.plot(line[:,0], line[:,1], line[:,2])
ax.plot(line.dot(M_transpose_to_0.T).dot(R_x)[:,0], line.dot(M_transpose_to_0.T).dot(R_x)[:,1], line.dot(M_transpose_to_0.T).dot(R_x)[:,2])
ax.plot(line.dot(M_transpose_to_0).dot(R_x).dot(R_y)[:,0], line.dot(M_transpose_to_0).dot(R_x).dot(R_y)[:,1], line.dot(M_transpose_to_0).dot(R_x).dot(R_y)[:,2])
ax.plot([0, 10], [0, 0], [0, 0], color='black')  # x-axis
ax.plot([0, 0], [0, 10], [0, 0], color='black')  # y-axis
ax.plot([0, 0], [0, 0], [0, 10], color='black')  # z-axis

#plot cube
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='r')

vertices = vertices.T
for edge in edges:
    ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]],
            [vertices[edge[0]][1], vertices[edge[1]][1]],
            [vertices[edge[0]][2], vertices[edge[1]][2]], color='b')
    


#plot cube
ax.scatter(result_rotation[:,0], result_rotation[:,1], result_rotation[:,2], color='r')

for edge in edges:
    ax.plot([result_rotation[edge[0]][0], result_rotation[edge[1]][0]],
            [result_rotation[edge[0]][1], result_rotation[edge[1]][1]],
            [result_rotation[edge[0]][2], result_rotation[edge[1]][2]], color='b')
    

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)



plt.show()