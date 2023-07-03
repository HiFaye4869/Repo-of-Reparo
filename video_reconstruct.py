# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:24:10 2023

@author: Christine
"""

import os
import argparse
import torch
from torchvision import utils as vutils
from vqgan import VQGAN
from utils import load_data
import np
from utils import ImagePaths

import cv2
'''
def generate_video(args):
    vqgan = VQGAN(args).to(device=args.device)
    checkpoint = torch.load(args.checkpoint_path)
    vqgan.load_state_dict(checkpoint['vqgan_state'])
    vqgan.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = load_data(args)
    #test_dataset = sorted(test_dataset.images)

    video_path = os.path.join(args.output_dir, 'output_video.mp4')
    video_writer = None

    with torch.no_grad():
        for i, imgs in enumerate(test_dataset):
            imgs = imgs.to(device=args.device)
            decoded_images, _, _ = vqgan(imgs)

            for j, image in enumerate(decoded_images):
                image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                if video_writer is None:
                    height, width, _ = image_bgr.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

                video_writer.write(image_bgr)

            if i >= args.num_images - 1:
                break

    video_writer.release()
    print(f"A video has been generated and saved to {video_path}.")
'''

def generate_image(args):
    vqgan = VQGAN(args).to(device=args.device)
    checkpoint = torch.load(args.checkpoint_path)
    vqgan.load_state_dict(checkpoint['vqgan_state'])
    vqgan.eval()

    os.makedirs(args.image_output, exist_ok=True)

    test_dataset = ImagePaths(args.dataset_path, size=args.image_size)
    test_dataset.images.sort(key = lambda x: int(x.replace('wave/','')[:-4]))  # Sort the image paths; replace the non-digit parts with '' to sort them

    with torch.no_grad():
        for i, img_path in enumerate(test_dataset.images):
            img = test_dataset.preprocess_image(img_path)
            img = torch.from_numpy(img).unsqueeze(0).to(device=args.device)
            decoded_images, _, _ = vqgan(img)

            for j, image in enumerate(decoded_images):
                image_path = os.path.join(args.image_output, f"{i}_{j}.jpg")
                vutils.save_image(image, image_path)

            if i >= args.num_images - 1:
                break

    print(f"{args.num_images} images have been generated and saved to {args.image_output}.")
    
def image_to_video(args):
     
    convert_image_path = args.image_output

    size = (256,256)
    videoWriter = cv2.VideoWriter(os.path.join(args.video_output,'testvideo2.avi'),cv2.VideoWriter_fourcc('I','4','2','0'),
                                  24,size)
     
    path_list = os.listdir(convert_image_path)

    path_list.sort(key=lambda x:int(x.split('_0.jpg')[0]))  #sort the frames in the folder 

    for img in path_list :
        img = os.path.join(convert_image_path, img)
        read_img = cv2.imread(img)
        videoWriter.write(read_img)
    videoWriter.release()   
    print(f"The reconstruct-video has been generated and saved to {args.video_output}")

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
    parser.add_argument('--image-output', type=str, default='generated_images',
                        help='Path to the directory for saving generated images (default: generated_images)')
    parser.add_argument('--video-output',type=str, default='reconstruct_results',
                        help='Path to the directory for saving output video (default: reconstruct_results)')

    args = parser.parse_args()

    generate_image(args)
    
    image_to_video(args)

    
