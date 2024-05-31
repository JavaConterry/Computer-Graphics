import numpy as np
import math as m
import matplotlib.pyplot as plt
import numpy as np

x_data = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
# x_data = np.linspace(0, 100, 100, dtype=int)
y_data = np.array([0, 12, 8, 20, 4, 73, 43, 9, 13, 7])
# y_data = np.sin(x_data/4)

def g(t):
    return

def g_hat(freq, xs, ys):
    sum = 0
    for i in range(len(ys)): #g(x)

        print('huinya:', -2*m.pi*xs[i]*freq)
        sum += ys[i]*m.exp(-2*m.pi*xs[i]*freq)
    # return sum/len(ys)
    return sum


def g_hat_inv(x, frequencies, g_hat):
    sum = 0
    for i in range(len(g_hat)):
        # sum += g_hat[i]*m.exp(2*m.pi*x*frequencies[i]) # doesn't have i
        sum += g_hat[i]*m.cos(2*m.pi*x*frequencies[i])
    return sum





fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# First plot
ax1.plot(x_data, y_data, label='g(x)')
ax1.scatter(x_data, y_data, color='red', label='Data points')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('g(x)')
ax1.legend()
ax1.grid(True)




frequencies = np.linspace(0, 2*m.pi, 100)
g_hat_data = []

for freq in frequencies:
    g_hat_data.append(g_hat(freq, x_data, y_data))

# Second plot
ax2.plot(frequencies, g_hat_data, label='g_hat(freq)')
ax2.scatter(frequencies, g_hat_data, color='red', label='Data points')
ax2.set_xlabel('freq')
ax2.set_ylabel('ĝ')
ax2.set_title('ĝ(freq)')
ax2.legend()
ax2.grid(True)

new_xs = np.linspace(0, 120, 1000)
g_inv = []

for x in new_xs:
    g_inv.append(g_hat_inv(x, frequencies, g_hat_data))

# Third plot
ax3.plot(new_xs, g_inv, label='g')
ax3.scatter(new_xs, g_inv, color='red', label='Inverse')
ax3.set_xlabel('x')
ax3.set_ylabel('g')
ax3.set_title('Inverse Fourier transform')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()