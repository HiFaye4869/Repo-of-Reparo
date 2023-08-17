# This file displays the function of smart_self_drop.
# The output image will be without the dropped trivial patches.
import numpy as np
import argparse
import itertools
import random
import os
import torchvision.transforms.functional as TF
from PIL import Image
import torch

def find_largest_bounding_rectangle(box):
    if len(box) == 0:
        return None
    
    # Extract x1, y1, x2, y2 values from the input tensor
    x1_values = box[:, 0]
    y1_values = box[:, 1]
    x2_values = box[:, 2]
    y2_values = box[:, 3]
    
    # Find the minimum and maximum values for x1, y1, x2, and y2
    min_x1 = max(np.min(x1_values), 0)
    min_y1 = max(np.min(y1_values), 0)
    max_x2 = max(np.max(x2_values), 0)
    max_y2 = max(np.max(y2_values), 0)
    
    s = (max_x2 - min_x1) * (max_y2 - min_y1)
    num_patch = s // 6400
    # Output the coordinates of the largest bounding rectangle
    largest_bounding_rectangle = [min_x1, min_y1, max_x2, max_y2]
    print(largest_bounding_rectangle)
    return largest_bounding_rectangle, num_patch

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : list
        Format: [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : list
        Format: [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

class PatchPicker:
    def __init__(self, self_drop):
        self.self_drop = self_drop

    def calculate_iou(self, box1, box2):
        # Calculate the Intersection over Union (IoU) between two bounding boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def index_patch_transfer(self, index):
        i = index % 4
        j = index // 4
        patch = [i * 80, j * 80, (i + 1) * 80, (j + 1) * 80]
        return patch

    def pick_patches(self, box):
        # Divide the image into 16 patches
        patch_width = 320 // 4
        patches = [(i * patch_width, j * patch_width, (i + 1) * patch_width, (j + 1) * patch_width) 
                   for i in range(4) for j in range(4)]

        # Pick self_drop number of patches that maximize IoU
        #selected_patches = []
        #remaining_patches = patches.copy()
        combines = itertools.combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], self.self_drop)
        combines = list(combines)
        
        best_iou = -1
        best_com = None
        for com in combines:
            iou = 0
            for i in com:
                # iou = iou + self.calculate_iou(box, self.index_patch_transfer(i))
                a = get_iou(box, self.index_patch_transfer(i))
                #print(i, a)
                iou = iou + a
            if iou > best_iou:
                best_iou = iou
                best_com = com

        print(best_com)
        #print(len(combines))
        return best_com
    
def patch_to_image(output_dir, image_size, output_back, drop_list):
    # Directory path where the patches are stored
    patches_directory = output_dir
    
    # Parameters for patch size and stride
    num_rows, num_cols = 4,4
    patch_size = image_size // 4
    stride = patch_size
    
    # Create a dictionary to store patches for each image
    image_patches = {}
    
    # List all files in the patches directory
    patch_filenames = os.listdir(patches_directory)
    
    for patch_filename in patch_filenames:
        # Load the patch image
        patch_image = Image.open(os.path.join(patches_directory, patch_filename))
    
        # Convert the patch image to a PyTorch tensor
        patch = TF.to_tensor(patch_image)
    
        # Extract the indices from the patch filename
        indices = patch_filename.split('.')[0].split('_')

        image_idx = int(indices[0])
        patch_idx = int(indices[1])
    
        # Check if the image index is already in the dictionary
        if image_idx in image_patches:
            # Append the patch to the existing image's patch list
            image_patches[image_idx].append((patch, patch_idx))
        else:
            # Create a new patch list for the image index
            image_patches[image_idx] = [(patch, patch_idx)]
    
    # Combine patches for each image into complete images
    for image_idx, patches in image_patches.items():
        # Create an empty tensor to store the combined image
        combined_image = torch.zeros(1, 3, num_rows * patch_size, num_cols * patch_size)
        
    
        # Loop through each patch and place it in the original location in the image tensor
        for patch, patch_idx in patches:
            if patch_idx in drop_list:
                continue
            # Calculate the row and column indices of the patch in the image tensor
            row_idx = patch_idx // num_cols
            col_idx = patch_idx % num_cols
    
            # Calculate the coordinates of the patch in the image tensor
            start_h = row_idx * patch_size
            start_w = col_idx * patch_size
            end_h = start_h + patch_size
            end_w = start_w + patch_size
            
            #print(patch.shape)
            # Place the patch in the original location in the image tensor
            combined_image[:, :, start_h:end_h, start_w:end_w] = patch
    
        # Convert the combined image tensor to a PIL image
        combined_image = TF.to_pil_image(combined_image.squeeze())
        
        os.makedirs(output_back, exist_ok=True)
        image_path = os.path.join(output_back, f"{image_idx}.jpg")
        combined_image.save(f'{image_path}')
    print(f"Images have been reconstructed and saved to {output_back}.")

def is_neighbor(patch1, patch2):
    # Check if two patches are neighbors
    row1, col1 = patch1 // 4, patch1 % 4
    row2, col2 = patch2 // 4, patch2 % 4
    return (abs(row1 - row2) <= 1 and col1 == col2) or (abs(col1 - col2) <= 1 and row1 == row2)

def valid_combination(combination):
    
    for patch1, patch2 in itertools.combinations(combination, 2):
        if is_neighbor(patch1, patch2):
            return False
    return True


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="reconstruct images using trained VQGAN")
    parser.add_argument('--output-patch', type=str, default='output_patch', help='the output directory of the output reconstruction images')
    parser.add_argument('--output-back', type=str, default='output_back', help='the output directory of the output reconstruction images')
    parser.add_argument('--image-size', type=int, default=320, help='the size of the input image')
    parser.add_argument("--self-drop", type=int, required=True,
                    help="Number of patches to be picked")
    parser.add_argument("--smart", action = 'store_true', help='do the smart_self_drop')
    args = parser.parse_args()

    # box = np.array([[ 5.58311e-01,  5.68663e+01,  1.71975e+02,  3.04615e+02],[ 2.25888e+01, -8.45184e-02,  1.71352e+02,  1.71610e+02]])
    box = np.array([[ 25.,   0., 175., 105.], [  0.,  29., 176., 282.]])
    '''
    [[ 26.,   0., 173., 120.], [  1., 128., 113., 291.]]
    '''
    if args.smart == True:
        box, num_patch = find_largest_bounding_rectangle(box)
        box = np.array(box)
        num_patch = int(num_patch)

        
        patch_picker = PatchPicker(num_patch)
        selected_patches = patch_picker.pick_patches(box)  # selected_patches: important patches

        all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        rest = set(all) - set(selected_patches)            # trivial patches that are save to be dropped
        combines = itertools.combinations(list(rest), args.self_drop)
        combines_list = list(combines)                     # get a list-version backup of iterator `combines`

        #valid_combinations = [c for c in combines if valid_combination(c)]
        a = []
        for c in combines_list:
            if valid_combination(c):
                a.append(c)
        #print(type(valid_combinations))

        if len(selected_patches) <= 16 - args.self_drop:
            if a:
                choice = random.choice(a)
                print(choice)
            else:
                print("Can only provide self-drop choices with neighbors!")
                choice = random.choice(combines_list)
                print(choice)
        else:
            print("Self-drop rate too high! May drop some important patches.")
            force_seleceted = itertools.combinations(selected_patches, 16 - num_patch - args.self_drop)
            choice = set(all) - set(selected_patches) + set(force_seleceted)
            choice = list(choice)
    else:
        combines = itertools.combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], args.self_drop)
        choice = random.choice(list(combines))

    patch_to_image(args.output_patch, args.image_size, args.output_back, choice)
