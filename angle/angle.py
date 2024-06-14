import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt



# np.random.seed(2)

vector = np.array([0, 1])
perturbation_level = 0.7

n_rand = 10_000

per_vectors = []
for i in range(n_rand):
    noise = np.random.randn(2)
    h = vector + perturbation_level*noise*norm(vector)/norm(noise)

    h /= norm(h)

    per_vectors.append(h)


per_vectors = np.array(per_vectors)


# Create scatterplot
plt.scatter(per_vectors[:, 0], per_vectors[:, 1], label='Perturbed Points')
plt.scatter(vector[0], vector[1], label='Actual')
plt.scatter(0, 0, label='center')
plt.title('Perturbation of datapoint')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.legend()

# Save the plot as a PNG file
plt.savefig('./angle/angle.png')

cos_sim_lst = []
for i in range(len(per_vectors)):
    cos_sim = dot(vector, per_vectors[i])/(norm(vector)*norm(per_vectors[i]))
    cos_sim_lst.append(cos_sim)

cos_sim_lst = np.array(cos_sim_lst)

print()
print('Angle')
print('min', min(np.arccos((cos_sim_lst)) * (180/np.pi)))
print('max', max(np.arccos((cos_sim_lst)) * (180/np.pi)))
print('mean', np.mean(np.arccos((cos_sim_lst)) * (180/np.pi)))
print('std', np.std(np.arccos((cos_sim_lst)) * (180/np.pi)))
















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