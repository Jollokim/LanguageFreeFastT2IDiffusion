import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from scipy.stats import truncnorm

def scipy_trunc_gaussian(truncation, std, size):
    return truncnorm.rvs(-2*truncation, 2*truncation, scale=std, size=size)

def trunc_gaus(limit, std, size):
    rand = np.random.randn(size) * std
    rand = np.clip(rand, 0, limit)

    return rand

def main():
    np.random.seed(0)
    clip_vectors = np.load('data/ffhq/clip_encoded_img.npz')

    # perturb = 0.1

    clip_v = clip_vectors['0'].reshape(512)
    clip_v_l2 = np.linalg.norm(clip_v)
    clip_v /= clip_v_l2

    noise = np.random.randn(10_000, 512)
    # noise = np.random.uniform(0, 3, (10_000, 512))
    noise_l2 = np.linalg.norm(noise, axis=-1)
    
    # perturb = np.abs(np.random.randn(10_000)) * 10 # gaussian
    perturb = 4 # fixed
    # perturb = truncnorm.rvs(0, 0.8, size=10_000) * 0.5  # truncated gaussian
    # perturb = trunc_gaus(0.8, 1.2, 10_000) # truncated gaussian
    # perturb = scipy_trunc_gaussian(0.8, 0.53, 10_000) # scipy truncated gaussian
    # print(perturb.shape)
    # perturb = np.random.uniform(0, 0.9, 10_000) # uniform

    # print(noise_norm.shape)

    cos_lst = []
    h_lst = []
    for i in tqdm(range(10_000)):

        noise = np.random.randn(512)
        noise_l2 = np.linalg.norm(noise, axis=-1)


        h = clip_v + (perturb*(noise/noise_l2))
        # h = clip_v + (perturb*noise*clip_v_l2/noise_l2)

        h /= norm(h)

        h_lst.append(h)
        
        cos_sim = dot(clip_v, h)/(norm(clip_v)*norm(h))
        cos_lst.append(cos_sim)


    degree_lst = np.degrees(np.arccos(cos_lst))
        


        
            # print(cos_sim)

    print()
    print()
    print('between noise and clip')
    print('min', np.min(cos_lst), np.min(degree_lst))
    print('max', np.max(cos_lst), np.max(degree_lst))
    print('mean', np.mean(cos_lst), np.mean(degree_lst))
    print('std', np.std(cos_lst), np.std(degree_lst))

    # exit()

    compare_h = h_lst[np.argmin(cos_lst)]
    cos_sim_lst = []


    for i in tqdm(range(len(h_lst))):
        cos_sim = dot(compare_h, h_lst[i])/(norm(compare_h)*norm(h_lst[i]))
        cos_sim_lst.append(cos_sim)



    print()
    print()
    print('between noise vectors')
    print('min', np.min(cos_sim_lst), np.degrees(np.arccos(np.min(cos_sim_lst))))
    print('max', np.max(cos_sim_lst), np.degrees(np.arccos(np.max(cos_sim_lst))))
    print('mean', np.mean(cos_sim_lst), np.degrees(np.arccos(np.mean(cos_sim_lst))))
    print('std', np.std(cos_sim_lst))
        
    




if __name__ == '__main__':
    main()