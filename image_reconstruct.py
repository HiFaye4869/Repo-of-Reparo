# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:34:35 2023

@author: Christine
"""

import os
import argparse
import torch
from torchvision import utils as vutils
from vqgan import VQGAN
from utils import load_data
from utils import ImagePaths


'''
def generate_images(args):
    vqgan = VQGAN(args).to(device=args.device)
    checkpoint = torch.load(args.checkpoint_path)
    vqgan.load_state_dict(checkpoint['vqgan_state'])
    vqgan.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = load_data(args)

    with torch.no_grad():
        for i, imgs in enumerate(test_dataset):
            imgs = imgs.to(device=args.device)
            decoded_images, _, _ = vqgan(imgs)

            for j, image in enumerate(decoded_images):
                image_path = os.path.join(args.output_dir, f"image_{i}_{j}.jpg")

                vutils.save_image(image, image_path)

            if i >= args.num_images - 1:
                break

    print(f"{args.num_images} images have been generated and saved to {args.output_dir}.")

'''
def generate_images(args):
    vqgan = VQGAN(args).to(device=args.device)
    checkpoint = torch.load(args.checkpoint_path)
    vqgan.load_state_dict(checkpoint['vqgan_state'])
    vqgan.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = ImagePaths(args.dataset_path, size=args.image_size)
    test_dataset.images.sort(key = lambda x: int(x.replace('wave/','')[:-4]))  # Sort the image paths; replace the non-digit parts with '' to sort them

    with torch.no_grad():
        for i, img_path in enumerate(test_dataset.images):
            img = test_dataset.preprocess_image(img_path)
            img = torch.from_numpy(img).unsqueeze(0).to(device=args.device)
            
            decoded_images, _, _ = vqgan(img)

            for j, image in enumerate(decoded_images):
                image_path = os.path.join(args.output_dir, f"{i}_{j}.jpg")
                vutils.save_image(image, image_path)

            if i >= args.num_images - 1:
                break

    print(f"{args.num_images} images have been generated and saved to {args.output_dir}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Images using Trained VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the generation is on')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for generating images (default: 20)')
    
    parser.add_argument('--num-images', type=int, default=10, help='Number of images to generate (default: 10)')
    parser.add_argument('--dataset-path', type=str, default='./input', help='Path to the folder of input images')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/vqgan_epoch_0.pt',
                        help='Path to the checkpoint file (default: checkpoints/vqgan_epoch_0.pt)')
    parser.add_argument('--output-dir', type=str, default='generated_images',
                        help='Path to the directory for saving generated images (default: generated_images)')

    args = parser.parse_args()

    generate_images(args)