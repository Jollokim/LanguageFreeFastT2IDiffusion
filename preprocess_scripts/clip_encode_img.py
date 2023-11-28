import torch
import clip
import os
from tqdm import tqdm
import numpy as np
from PIL import Image


def test(**kwargs):
    print(kwargs)


def main():
    # test(**{'x': 1, 'y': 2})

    img_dir = 'data/MM_CelebA_HQ/images/faces'
    out_npz = 'data/MM_CelebA_HQ/clip_encoded_img.npz'

    img_files = os.listdir(img_dir)
    img_files = sorted([int(file[:file.index('.jpg')]) for file in img_files])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    data = {}

    count = 0
    for file in tqdm(img_files):

        image = preprocess(Image.open(f"{img_dir}/{file}.jpg")).unsqueeze(0).to(device)
        features = model.encode_image(image)

        np_feature = features.detach().cpu().numpy().astype(np.float32)

        data[str(file)] = np_feature
        


    np.savez(out_npz, **data)

    npz = np.load(out_npz)

    print(npz.files)
    print(npz['0'].shape)
    print(npz['0'].dtype)









if __name__ == '__main__':
    main()