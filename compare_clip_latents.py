import torch
import torch.nn.functional as F
import clip
import os
from PIL import Image
from tqdm import tqdm
import numpy as np


def encode_text(text, model):
    # Load the pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    # Preprocess the text
    text = clip.tokenize(text).to("cuda")

    # Encode the text
    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features

def encode_list_of_images(images_dir, model, preprocess):
    # Preprocess the images
    images = torch.stack([preprocess(Image.open(f'{images_dir}/{image_path}')) for image_path in os.listdir(images_dir)]).to("cuda")
    
    # Encode the images
    with torch.no_grad():
        image_features = model.encode_image(images)

    return image_features


def compare_cosine_similarity(text_feature, image_features):
    text_feature = text_feature.unsqueeze(0).expand(image_features.shape[0], -1)
    # Calculate the cosine similarity between the text feature and the image features
    similarities = F.cosine_similarity(text_feature, image_features, dim=1)

    return similarities

def main():
    descriptions = [
        "She is attractive and has bushy eyebrows, brown hair, mouth slightly open, high cheekbones, pointy nose, and arched eyebrows. The picture is taken by the sea.",
        "This man is attractive and has bushy eyebrows, and black hair. The background is a forest.",
        "The woman is of Japanese origin. She has deep blue colored hair.",
        "This male is about 70 years old.",
        "This person is about 10 years young. She has blond hair and is happy.",
        "Man with oval face, dark eyes, glasses, neat black hair, and professional attire. He is of Indian origin.",
        "A girl from Nepal wearing traditional head wear.",
        "A man originating from Norway. He has brown hair and a beard. He is young.",
        "This woman is of Latin America origin. She is beautiful with black hair.",
        "This person is of African origin. He is wearing sunglasses and a cap."
    ]

    base_dir = 'samples_t2f/72'

    # Load the pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    enc_descriptions = encode_text(descriptions, model)

    enc_images = []

    for i in tqdm(range(len(descriptions))):
        images_dir = f'{base_dir}/description_{i}'
        enc_image = encode_list_of_images(images_dir, model, preprocess)
        enc_images.append(enc_image)

    similarities_per_description = {}
    global_similarities = []

    for i in tqdm(range(len(descriptions))):
        similarities = compare_cosine_similarity(enc_descriptions[i], enc_images[i])

        similarities_per_description['description_' + str(i)] = similarities
        
        for i in range(len(similarities)):
            global_similarities.append(similarities[i])

    
    for key, value in similarities_per_description.items():
        print(f'mean: {key}: {torch.mean(value)}')
        print(f'max: {key}: {torch.max(value)}, {torch.argmax(value)}')
        print(f'min: {key}: {torch.min(value)}, {torch.argmin(value)}')
        print()

    mean_similarity = sum([torch.sum(value) for value in similarities_per_description.values()]) / (len(descriptions) * len(os.listdir(images_dir)))
    print('base dir', base_dir)
    print('Mean of means similarity: ', torch.mean(torch.tensor(global_similarities)))



if __name__ == '__main__':
    main()