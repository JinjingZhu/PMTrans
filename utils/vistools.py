from copy import deepcopy
import string
import numpy as np
import cv2
import torch

def patch_plt(img, out,num_patch = (8, 8), space=(2, 2)):
    if isinstance(img, str):
        img = cv2.imread(img)
    H, W, C = img.shape
    patch_size = (H//8, W//8)
    H_, W_ = np.array(num_patch)*np.array(patch_size)
    img_ = cv2.resize(img, (W_, H_))
    # f, axes = plt.subplots(num_patch[0], num_patch[1], figsize=(10, 10), sharey=True, sharex=True)
    
    # feed up sapce 
    blank = np.ones((H_+(num_patch[0]-1)*space[0], W_+(num_patch[1]-1)*space[1], C), np.uint8)*255
    for i in range(num_patch[0]):
        for j in range(num_patch[1]):
            h = i*(patch_size[0]+space[0])
            w = j*(patch_size[1]+space[1])
            blank[h:h+patch_size[0], w: w+patch_size[1]]=img_[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]]
    cv2.imwrite(out, blank)
    
def patch_aug(img, weights, out=None,num_patch = (8, 8)):
    assert(len(weights) == num_patch[0]*num_patch[1])
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    if isinstance(img, str):
        img = cv2.imread(img)
    else:
        img = deepcopy(img)
    H, W, C = img.shape
    patch_size = (H//8, W//8)
    H_, W_ = np.array(num_patch)*np.array(patch_size)
    for i in range(num_patch[0]):
        for j in range(num_patch[1]):
            h = i*patch_size[0]
            w = j*patch_size[1]
            block = img[h:h+patch_size[0], w: w+patch_size[1]]
            block = block*weights[i*num_patch[0]+j]
            img[h:h+patch_size[0], w: w+patch_size[1]] = block
    if out:
        cv2.imwrite(out, img.astype(np.uint8))
    else:
        return img