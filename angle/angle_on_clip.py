import numpy as np
from numpy import dot
from numpy.linalg import norm




def main():
    clip_vectors = np.load('data/ffhq/clip_encoded_img.npz')

    perturb = 0.0174

    clip_v = clip_vectors['0'].reshape(512)
    clip_v_l2 = np.linalg.norm(clip_v)
    clip_v /= clip_v_l2

    noise = np.random.randn(10_000, 512)
    noise_l2 = np.linalg.norm(noise, axis=-1)

    # print(noise_norm.shape)

    cos_sim_lst = []
    for i in range(10_000):
        h = clip_v + (perturb*noise[i]/noise_l2[i])

        h /= norm(h)
        
        cos_sim = dot(clip_v, h)/(norm(clip_v)*norm(h))
        cos_sim_lst.append(cos_sim)
        # print(cos_sim)

    print()
    print()
    print()
    print('min', np.min(cos_sim_lst))
    print('max', np.max(cos_sim_lst))
    print('mean', np.mean(cos_sim_lst))
    print('std', np.std(cos_sim_lst))
    




if __name__ == '__main__':
    main()