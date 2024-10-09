# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os
import pickle
import clip
torch.cuda.set_device(1)
from Image_path import ImagePaths
from torchvision import transforms
from PIL import Image
import  numpy as np
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])
# dataset = ImagePaths('/home/yesenmao/disk/dataset/coco/coco_sent/captions_train2014.json',  transform)

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    with open('text_clip2.pkl','rb') as f:
            clip_fea = pickle.load(f).float()
    model_clip, preprocess = clip.load("ViT-B/32", device="cuda")
    text_raw=["A police man on a motorcycle is idle in front of a bush.",
"Some red and green flower in a room.",
"Assorted electronic devices sitting together in a photo",#this bird is a lime green with greyish wings and long legs.",
"A man riding a wave on top of a surfboard."]
#     text_raw=text_raw+["This bird is white from crown to belly, with gray wingbars and retrices.",
# "A small bird with blue-grey wings, rust colored sides and white collar.",
# "This bird is mainly grey, it has brown on the feathers and back of the tail.",
# "A bird with blue head, white belly and breast, and the bill is pointed."]
# text_raw = ["this bird is yellow",
# "this bird has a red head.",
# "this bird is white.",
# "a bird has brown feathers.",
# "this bird is yellow",
# "this bird has a red head.",
# "this bird is white.",
# "a bird has brown feathers."]
    #prompt = "a picture of flower"
    #prompt = "no description"
    #prompt = "a picture"
    prompt = "null"
    #prompt = "sjkldf sajfsdjf; slkfss ;slfl"
    # prompt = "we don't know what's in this picture"

    text = clip.tokenize(text_raw+[
# prompt,
# prompt,
# prompt,
# prompt,
prompt,
prompt,
prompt,
prompt]).to(device)
    text_features = model_clip.encode_text(text)
    # Labels to condition the model with (feel free to change):
    class_labels = [34, 79, 5, 30, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(text_raw)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    y  = text_features.float()
    # Setup classifier-free guidance:
    z = torch.cat([z,z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    #y = torch.cat([y, y], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.23126199671607967).sample
    #samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    print('rrr')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--ckpt", type=str, default='/disk/yesenmao/ldm/Dit_coco/results/006-DiT-L-2/checkpoints/0200000.pt',
    parser.add_argument("--ckpt", type=str, default='/disk/yesenmao/ldm/DiT_coco_extra/results/004-DiT-L-2/checkpoints/4450000.pt',
                help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
