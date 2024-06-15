import os
import json
import torch
from tqdm import tqdm
import PIL.Image
from models.maskdit import EDMPrecond
import clip
from sample import edm_sampler, ablation_sampler
from utils import StackedRandomGenerator
import autoencoder

from train_utils.datasets import norm_by_l2



class T2FConfig:
    def create_model(self):
        model = EDMPrecond(
            img_resolution=self.in_size,
            img_channels=self.in_channels,
            num_classes=self.num_classes,
            model_type=self.model_type,
            use_decoder=self.use_decoder,
            mae_loss_coef=self.mae_loss_coef,
            pad_cls_token=self.pad_cls_token,
        )

        return model

    def load_model_weights(self):
        ckpt = torch.load(self.ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt['ema'], strict=True)

        self.model = self.model.to(self.device)


    def set_seed(self):
        torch.manual_seed(self.seed)


    def clip_encode_text(self, normalize=False):
        text = clip.tokenize(self.text).to(self.device)
        features = self.model.encode_text(text).float()

        if normalize:
            norm_features = torch.zeros_like(features)
            for i in range(len(features)):
                feature = features[i].detach().cpu().numpy()

                feature = norm_by_l2(feature)
                norm_features[i] = torch.tensor(feature)

            return norm_features.to(self.device).float()


        return features
    
    
    def clip_encode_image(self, normalize=True):
        images_path = self.images
        images = []
        for path in images_path:
            image = PIL.Image.open(path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            images.append(image)
        images = torch.cat(images, dim=0)
        features = self.model.encode_image(images).float()

        if normalize:
            norm_features = torch.zeros_like(features)
            for i in range(len(features)):
                feature = features[i].detach().cpu().numpy()

                feature = norm_by_l2(feature)
                norm_features[i] = torch.tensor(feature)

            return norm_features.to(self.device).float()

        return features
    

    def create_new_sampling_dir(self, root: str):
        dirs = os.listdir(root)
        dirs = [int(dir) for dir in dirs]

        if len(dirs) == 0:
            outdir = f'{root}/0'
            os.makedirs(outdir)
        else:
            mx = max(dirs)
            outdir = f'{root}/{mx+1}'
            os.makedirs(outdir)

        return outdir


    def create_config_note(self):
        # Extract attributes from the object's __dict__
        obj_dict = {attr: value for attr, value in self.__dict__.items() if not callable(value)}

        # Write the dictionary to the file in JSON format
        with open(f'{self.outdir}/config.json', "w") as json_file:
            json.dump(obj_dict, json_file, indent=2)



    def __init__(self) -> None:
        self.outdir = self.create_new_sampling_dir('samples_t2f')

        # CUDA device related
        self.device = 'cuda'

        # model
        self.in_size = 32
        self.in_channels = 4
        self.num_classes = 512 # match the feature dimension
        self.model_type = 'DiT-XL/2'
        self.use_decoder = True
        self.mae_loss_coef = 0.1
        self.pad_cls_token = False

        # path to the model checkpoint
        self.ckpt = 'results/language_free/celeb/masked_finetuning_exp/DiT-XL-2-edm-MM-CelebA-HQ-clsdrop0.1-m0.5-de1-mae0.1-bs-128-lr5e-05-gaussian1.2trunc0-0mean0-languagefree_from_ffhq/checkpoints/0139000.pt'

        # sampling algorithm
        self.num_steps = 40
        self.S_churn = 0
        self.solver = None
        self.discretization = None
        self.schedule = None
        self.scaling = None

        # classifier free guidance / tends to saturate colors in image
        self.cfg_scale = 4

        # VAE
        self.pretrained_path = 'assets/stable_diffusion/autoencoder_kl.pth'

        # seed
        self.seed = 0
        self.set_seed()

        # descriptions for t2i
        self.text = [
            "She is attractive and has bushy eyebrows. She has a round face and a small nose. The picture is taken by the sea.",
        ]



        # recreate images
        self.images = [
            # 'data/friends_HQ/friend1.png',
        ]

        # create note
        self.create_config_note()

        # sampling size per description
        self.batch_size = 64 # number of images per description
        self.seeds = [i for i in range(self.seed, (self.seed+self.batch_size))]

        # CLIP model
        self.model, self.transform = clip.load("ViT-B/32", device=self.device)

        

        if len(self.text) > 0:
            self.clip_text = self.clip_encode_text(normalize=True)
        else:
            self.clip_text = None

        if len(self.images) > 0:
            self.clip_img = self.clip_encode_image(normalize=True)
        else:
            self.clip_img = None

        del self.model # conserve memory

        self.model = self.create_model()
        self.load_model_weights()




def main():
    args = T2FConfig()
    generate_text_to_face(args, args.model, args.device)



@torch.no_grad()
def generate_text_to_face(args: T2FConfig, net, device, vae=None):
    seeds = args.seeds
    batch_size = args.batch_size

    net.eval()

    # Setup sampler
    sampler_kwargs = dict(num_steps=args.num_steps, S_churn=args.S_churn,
                          solver=args.solver, discretization=args.discretization,
                          schedule=args.schedule, scaling=args.scaling)
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
    sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
    print(f"sampler_kwargs: {sampler_kwargs}, \nsampler fn: {sampler_fn.__name__}")
    # Setup autoencoder (if not present in parameters)
    if vae is None:
        vae = autoencoder.get_model(args.pretrained_path).to(device)

    if args.clip_text is not None:
        # generate images
        print(f'Generating {len(seeds)} images for {args.clip_text.shape[0]} descriptions to "{args.outdir}"...')
        for i in tqdm(range(args.clip_text.shape[0]), unit='description'):
            
            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, seeds)
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            
            class_labels = args.clip_text[i].reshape(1, args.num_classes).expand(batch_size, -1)

            # Generate images.
            def recur_decode(z):
                try:
                    return vae.decode(z)
                except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
                    assert z.shape[2] % 2 == 0
                    z1, z2 = z.tensor_split(2)
                    return torch.cat([recur_decode(z1), recur_decode(z2)])

            with torch.no_grad():
                z = sampler_fn(net, latents.float(), class_labels.float(), randn_like=rnd.randn_like,
                            cfg_scale=args.cfg_scale, **sampler_kwargs).float()
                images = recur_decode(z)
                
            # Save images.
            images_np = images.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            # images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(seeds, images_np):
                image_dir = os.path.join(args.outdir, f'description_{i}')
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    
    if args.clip_img is not None:
        # generate images
        print(f'Generating {len(seeds)} images for {args.clip_img.shape[0]} images to "{args.outdir}"...')
        for i in tqdm(range(args.clip_img.shape[0]), unit='image'):
            
            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, seeds)
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            
            class_labels = args.clip_img[i].reshape(1, args.num_classes).expand(batch_size, -1)

            # Generate images.
            def recur_decode(z):
                try:
                    return vae.decode(z)
                except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
                    assert z.shape[2] % 2 == 0
                    z1, z2 = z.tensor_split(2)
                    return torch.cat([recur_decode(z1), recur_decode(z2)])

            with torch.no_grad():
                z = sampler_fn(net, latents.float(), class_labels.float(), randn_like=rnd.randn_like,
                            cfg_scale=args.cfg_scale, **sampler_kwargs).float()
                images = recur_decode(z)
                
            # Save images.
            images_np = images.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            # images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(seeds, images_np):
                image_dir = os.path.join(args.outdir, f'image_{i}')
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)


if __name__ == '__main__':
    main()

