import numpy as np
import lmdb
from tqdm import tqdm


def put_clip(npz: str, n_samples: int, env):
    # key: image number, value: arr size = (n, 10, 512), 10 descriptions encoded
    embedded_txt = np.load(npz)

    with env.begin(write=True) as txn:
        for sample_idx in tqdm(range(n_samples)):
            sample_descriptions = embedded_txt[str(sample_idx)].astype(np.float32)

            for descrip_idx in range(sample_descriptions.shape[0]):
                txn.put(f'y-{sample_idx}-{descrip_idx}'.encode('utf-8'), sample_descriptions[descrip_idx])







    

def main():
    lmdb_dataset_root = 'my_test_lmdb'
    clip_npz = 'data/MM_CelebA_HQ/clip_encoded_text.npz' # TODO: encode a new, to low float precision (float16, needs to be float32)
    dataset = 'data/MM_CelebA_HQ/images/faces'
    n_samples = 30_000

    z_shape = (4, 32, 32)
    clip_shape= (512)

    env = lmdb.open(f'{lmdb_dataset_root}/', map_size=1_260_000_000)

    put_clip(clip_npz, n_samples, env)

    # with env.begin(write=True) as txn:
    #     z = np.ones(z_shape, dtype=np.float32)
    #     cls = 'face'
    #     clip = np.ones((10, 512), dtype=np.float32)

    #     txn.put('X'.encode('utf-8'), z)
    #     txn.put('class'.encode('utf-8'), cls.encode('utf-8'))
    #     txn.put('clip'.encode('utf-8'), clip)



    # with env.begin(write=False) as txn:
    #     z = txn.get('X'.encode('utf-8'))
    #     z = np.frombuffer(z, dtype=np.float32).reshape(z_shape).copy()

    #     cls = txn.get('class'.encode('utf-8')).decode('utf-8')
        
    #     clip = txn.get('clip'.encode('utf-8'))
    #     clip = np.frombuffer(clip, dtype=np.float32).reshape((10, 512)).copy()

        


    #     print(z)
    #     print(type(z))
    #     print(z.shape)

    #     print(cls)
    #     print(clip.shape)

    




if __name__ == '__main__':
    main()    