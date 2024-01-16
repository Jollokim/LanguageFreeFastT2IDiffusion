import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt



np.random.seed(2)

a = [0, 1]
a = a / norm(a)
p_level = 1


n_rand = 10000

x = np.random.randn(n_rand)
y = np.random.randn(n_rand)

noise = np.array([
    np.array([x[i], y[i]]) / norm(np.array([x[i], y[i]])) for i in range(len(x))
])

h = a + (p_level*noise)
h_norm = norm(h, axis=1)

for i in range(len(h_norm)):
    h[i] = h[i] / h_norm[i]

# print(h.shape)
# quit()



x = h[:, 0]
y = h[:, 1]

# Create scatterplot
plt.scatter(x, y, label='Data Points')
plt.scatter(a[0], a[1], label='Actual')
plt.scatter(0, 0, label='center')
plt.title('Scatterplot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.legend()

# Save the plot as a PNG file
plt.savefig('./angle/angle.png')

cos_sim_lst = []
for i in range(h.shape[0]):
    cos_sim = dot(a, h[i])/(norm(a)*norm(h[i]))
    cos_sim_lst.append(cos_sim)


print(np.arccos(min(cos_sim_lst)) * (180/np.pi))
















# a = [0, 1]
# noise = np.random.randn(2)


# print('a', a, norm(a))
# print('noise', noise, norm(noise))
# print()

# a = a / norm(a)
# noise = noise / norm(noise)

# print('a', a, norm(a))
# print('noise', noise, norm(noise))
# print()

# h = a + (p_level*noise)

# print('h', h, norm(h))
# print()

# h = h / norm(h)

# print('h', h, norm(h))
# print()

# cos_sim = dot(a, h)/(norm(a)*norm(h))

# print(cos_sim)
# print(np.arccos(cos_sim))