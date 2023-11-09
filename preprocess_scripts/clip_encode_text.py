import torch
import clip
import os
from tqdm import tqdm
import numpy as np


def test(**kwargs):
    print(kwargs)


def main():
    # test(**{'x': 1, 'y': 2})

    descriptions_dir = 'data/MM_CelebA_HQ/descriptions'
    out_npz = 'data/MM_CelebA_HQ/clip_encoded_text.npz'

    description_files = os.listdir(descriptions_dir)
    description_files = sorted([int(file[:file.index('.txt')]) for file in description_files])
    

    # print(clip.available_models())
    # return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    data = {}

    count = 0
    for file in tqdm(description_files):
        lines = get_lines_from_file(f'{descriptions_dir}/{file}.txt')

        text_tokenized = clip.tokenize(lines).to(device)

        text_features = model.encode_text(text_tokenized)

        np_features = text_features.detach().cpu().numpy()

        data[str(file)] = np_features


    np.savez(out_npz, **data)

    npz = np.load(out_npz)

    print(npz.files)
    print(npz['0'].shape)


        



        



def get_lines_from_file(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for i in range(len(lines)):
            lines[i] = lines[i][:lines[i].index('\n')]

        return lines








if __name__ == '__main__':
    main()