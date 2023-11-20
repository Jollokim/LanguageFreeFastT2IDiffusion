import numpy as np
import lmdb
from tqdm import tqdm

from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
import os

import sys
sys.path.insert(0, './')  # unfortunate, result of the project structure

from autoencoder import get_model
from train_utils.datasets import center_crop_arr


def get_image_transform(resolution: int = 256):
    return transforms.Compose([
        transforms.Resize(size=(resolution)), # for purpose of outside dataset.
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

class VAEDataset(VisionDataset):
    """
    Dataloader for MM-CelebA-HQ with clip encoded text.
    """
    def __init__(self, root: str, transform=None, target_transform=None, 
                 resolution=256, img_end='.jpg'):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.root: str = root
        self.resolution: int = resolution
        self.img_end: str = img_end

        self.transform = transform
        
        # get all image names
        self.ids: list[str] = sorted([int(file[:file.index(img_end)]) for file in os.listdir(root)])
        
    
    def __getitem__(self, index: int):
        path: str = f'{self.root}/{self.ids[index]}{self.img_end}'

        img = Image.open(path).convert('RGB')
        arr = center_crop_arr(img, self.resolution)
        if self.transform is not None:
            arr = self.transform(arr)

        return arr.float(), self.ids[index]

    def __len__(self) -> int:
        return len(self.ids)


@torch.no_grad()
def put_z(vae, env, dataset_path, batch_size=5, device='cuda'):
    transform = get_image_transform()
    dataset = VAEDataset(dataset_path, transform)
    dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=2
    )

    vae = vae.to(device)

    with env.begin(write=True) as txn:
        for x, id in tqdm(dataloader):
            x = x.to(device)
            id = id.to(device)

            z = vae.encode(x)
            z = z.detach().cpu().numpy()

            id = id.detach().cpu().numpy()

            for i in range(z.shape[0]):
                txn.put(f'z-{id[i]}'.encode('utf-8'), z[i])


def put_feature(npz: str, n_samples: int, env):
    # key: image number, value: arr size = (n, 10, 512), 10 descriptions encoded
    embedded_txt = np.load(npz)

    with env.begin(write=True) as txn:
        for sample_idx in tqdm(range(n_samples)):
            sample_descriptions = embedded_txt[str(sample_idx)].astype(np.float32)

            for descrip_idx in range(sample_descriptions.shape[0]):
                txn.put(f'y-{sample_idx}-{descrip_idx}'.encode('utf-8'), sample_descriptions[descrip_idx])


def put_single(env, key: str, value: int):
    with env.begin(write=True) as txn:
        txn.put(key.encode('utf-8'), value.to_bytes(8, 'big'))

    with env.begin(write=False) as txn:
        print(f"Put {key} with value {int.from_bytes(txn.get(key.encode('utf-8')), 'big')}")


def sanity_check(env, shape, dtype, n_features=None):
    print('------Begin sanity check------')
    with env.begin(write=False) as txn:
        try:
            length_bi = txn.get('length'.encode('utf-8'))
            length = int.from_bytes(length_bi, 'big')
            print('Dataset size', length)
            
        except:
            print('Failed sanity check: Couldn\'t read length')

        try:
            if n_features is not None:
                n_features_bi = txn.get('n_features'.encode('utf-8'))
                n_features = int.from_bytes(n_features_bi, 'big')

                print('Number of features per sample', n_features)
        except:
            print('Failed sanity check: Couldn\'t read num features-')

        try:
            print('Testing loading of content')
            for sample_idx in tqdm(range(length)):

                if n_features is not None:
                    for f_idx in range(n_features):
                        features_bi = txn.get(f'y-{sample_idx}-{f_idx}'.encode('utf-8'))
                        x = np.frombuffer(features_bi, dtype=dtype).reshape(shape).copy()
                else:
                    x_bi = txn.get(f'z-{sample_idx}'.encode('utf-8'))
                    x = np.frombuffer(x_bi, dtype=dtype).reshape(shape).copy() 
        except Exception:
            print('Failed sanity check: Couldn\'t read content vector')
            raise Exception


def main():
    # dataset place
    lmdb_dataset_root = 'data/MM_CelebA_HQ/MM_CelebA_HQ_lmdb'
    os.makedirs(lmdb_dataset_root, exist_ok=True)
    
    # feature
    feature_npz = 'data/MM_CelebA_HQ/clip_encoded_text.npz'
    feature_name = 'clip'
    feature_shape = (512)
    feature_dtype = np.float32
    n_features = 10
    
    # dataset
    dataset = 'data/MM_CelebA_HQ/images/faces'
    n_samples = 30_000
    

    # vae
    vae_weights = 'assets/stable_diffusion/autoencoder_kl.pth'
    vae = get_model(vae_weights)
    z_shape = (4, 32, 32)
    z_dtype = np.float32
    batch_size = 64
    device = 'cuda'
    
    # store feature vectors
    env_feature = lmdb.open(f'{lmdb_dataset_root}/{feature_name}', map_size=1_260_000_000)
    put_feature(feature_npz, n_samples, env_feature)
    put_single(env_feature, 'length', n_samples)
    put_single(env_feature, 'n_features', n_features)

    sanity_check(env_feature, feature_shape, feature_dtype, n_features=n_features)


    # store vae image encodings
    env_z = lmdb.open(f'{lmdb_dataset_root}/z', map_size=1_260_000_000)
    put_z(vae, env_z, dataset, batch_size)
    put_single(env_z, 'length', n_samples)

    sanity_check(env_z, z_shape, z_dtype)
    




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