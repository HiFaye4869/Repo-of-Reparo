import numpy as np
import argparse
import itertools
import random
import os

def find_largest_bounding_rectangle(box):
    if len(box) == 0:
        return None
    
    # Extract x1, y1, x2, y2 values from the input tensor
    x1_values = box[:, 0]
    y1_values = box[:, 1]
    x2_values = box[:, 2]
    y2_values = box[:, 3]
    
    # Find the minimum and maximum values for x1, y1, x2, and y2
    min_x1 = np.min(x1_values)
    min_y1 = np.min(y1_values)
    max_x2 = np.max(x2_values)
    max_y2 = np.max(y2_values)
    
    s = (max_x2 - min_x1) * (max_y2 - min_y1)
    num_patch = s // 6400
    # Output the coordinates of the largest bounding rectangle
    largest_bounding_rectangle = [min_x1, min_y1, max_x2, max_y2]
    return largest_bounding_rectangle, num_patch

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
        i = index // 4
        j = index % 4
        patch = [i * 90, j * 90, (i + 1) * 90, (j + 1) * 90]
        return patch

    def pick_patches(self, box):
        # Divide the image into 16 patches
        patch_width = 360 // 4
        patches = [(i * patch_width, j * patch_width, (i + 1) * patch_width, (j + 1) * patch_width) 
                   for i in range(4) for j in range(4)]

        # Pick self_drop number of patches that maximize IoU
        #selected_patches = []
        #remaining_patches = patches.copy()
        combines = itertools.combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], self.self_drop)
        combines = list(combines)
        '''
        for _ in range(self.self_drop):
            best_iou = -1
            best_patch = None

            for patch in remaining_patches:
                iou = self.calculate_iou(box, patch)
                if iou > best_iou:
                    best_iou = iou
                    best_patch = patch

            if best_patch is not None:
                selected_patches.append(remaining_patches.index(best_patch))
                remaining_patches.remove(best_patch)
        '''
        best_iou = -1
        best_com = None
        for com in combines:
            iou = 0
            for i in range(self.self_drop):
                iou = iou + self.calculate_iou(box, self.index_patch_transfer(com[i]))
            if iou > best_iou:
                best_iou = iou
                best_com = com

            
        print(best_com)
        #print(len(combines))
        return best_com, combines

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Picker")
    parser.add_argument("--self-drop", type=int, required=True,
                        help="Number of patches to be picked")
    args = parser.parse_args()

    box = np.array([[ 5.58311e-01,  5.68663e+01,  1.71975e+02,  3.04615e+02],[ 2.25888e+01, -8.45184e-02,  1.71352e+02,  1.71610e+02]])

    #box = np.array([0.558311, -0.0845184, 171.975, 304.615])  # Example bounding box [x1, y1, x2, y2]
    box, num_patch = find_largest_bounding_rectangle(box)
    box = np.array(box)

    patch_picker = PatchPicker(args.self_drop)
    selected_patches, combines = patch_picker.pick_patches(box)
    rest = set(combines) - set(selected_patches)
    choice = random.choice(list(combines))
    print("Important patches indexes:", selected_patches)
    print("Dropped patches:", choice)

#[0.558311, -0.0845184, 171.975, 304.615]
