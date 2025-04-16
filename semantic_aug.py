import torch
import torch.nn.functional
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from functions import *
from strategy import *
import random 

def truncated_normal(mean=0.0, std=1, low=-2, high=2):
    while True:
        val = torch.normal(mean=float(mean), std=float(std), size=(1,))
        if low <= val <= high:
            return float(val)
        
def rotate_fill_prior(delta, semantic_kd, args, random_batch, device='cpu', interpolation=transforms.InterpolationMode.BILINEAR):
    ori = delta.clone().to(device)
    ori_kd = semantic_kd.clone().to(device)
    angle = truncated_normal(mean=0.0, std= args.angle, low= -args.angle, high= args.angle)
    # for delta rotate
    rotated_delta = TF.rotate(delta, angle, fill=0, interpolation=interpolation)
    mask = (rotated_delta == 0).float()
    final_delta = rotated_delta * (1 - mask) + ori.to(device) * mask
    
    #for semantic sample rotate
    rotated_kd = TF.rotate(semantic_kd, angle, fill=0, interpolation=interpolation)
    mask = (rotated_kd == 0).float()
    final_kd = rotated_kd * (1 - mask) + ori_kd.to(device) * mask
    
    
    return final_delta, final_kd

def scaling_transform(delta, semantic_kd, args, device):
    ratio = torch.empty(1).to(device)
    ratio = ratio.uniform_(args.scale_t_low, args.scale_t_high).item()
    final_delta = delta * ratio 
    final_kd = semantic_kd * ratio
    return final_delta, final_kd

def get_length(length, num_block):
    length = int(length)
    rand = np.random.uniform(size=num_block)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm < 1] = 1
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)
    
def shuffle_only(x, semantic_kd, num_block):
    _, h, w = x.shape
    width_length, height_length = get_length(w, num_block=num_block), get_length(h, num_block=num_block)
    width_perm = np.random.permutation(np.arange(num_block))
    height_perm = np.random.permutation(np.arange(num_block))
    
    x_split_w = torch.split(x, width_length, dim=2)
    x_split_h_l = [torch.split(x_split_w[i], height_length, dim=1) for i in width_perm]
    
    shuffled_strips = []
    
    for strip in x_split_h_l:
        shuffled_height_sections = [strip[i] for i in height_perm]
        shuffled_strip = torch.cat(shuffled_height_sections, dim=1)
        shuffled_strips.append(shuffled_strip)
    x_h_perm = torch.cat(shuffled_strips, dim=2)
    
    kd_split_w = torch.split(semantic_kd, width_length, dim=2)
    kd_split_h_l = [torch.split(kd_split_w[i], height_length, dim=1) for i in width_perm]
    
    shuffled_strips = []
    
    for strip in kd_split_h_l:
        shuffled_height_sections = [strip[i] for i in height_perm]
        shuffled_strip = torch.cat(shuffled_height_sections, dim=1)
        shuffled_strips.append(shuffled_strip)
    kd_h_perm = torch.cat(shuffled_strips, dim=2)

    return x_h_perm, kd_h_perm

def random_crop(X, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
    C, H, W = X.shape
    image_area = H * W

    for attempt in range(10):
        target_scale = random.uniform(scale[0], scale[1])
        target_ratio = random.uniform(ratio[0], ratio[1])

        target_area = image_area * target_scale
        target_width = int(round(math.sqrt(target_area * target_ratio)))
        target_height = int(round(math.sqrt(target_area / target_ratio)))

        if target_height <= H and target_width <= W:
            top = random.randint(0, H - target_height)
            left = random.randint(0, W - target_width)
            crop_x = X[:, top:top + target_height, left:left + target_width]
            return crop_x, target_height, target_width, top, left

    in_ratio = float(W) / float(H)
    if in_ratio < min(ratio):
        target_width = W
        target_height = int(round(W / min(ratio)))
    elif in_ratio > max(ratio):
        target_height = H
        target_width = int(round(H * max(ratio)))
    else:
        target_height = H
        target_width = W

    target_height = min(target_height, H)
    target_width = min(target_width, W)

    top = (H - target_height) // 2
    left = (W - target_width) // 2
    crop_x = X[:, top:top + target_height, left:left + target_width]
    return crop_x, target_height, target_width, top, left

def random_crop_and_resize(
    X, 
    prior,
    scale_crop=(0.2, 1.0), 
    ratio_crop=(3./4., 4./3.),
):
    B, C, H, W = X.shape
    original_size = (H, W)
    resized_images = []

    for i in range(B):
        image = X[i]
        crop_x, crop_h, crop_w, crop_top, crop_left = random_crop(
            image, scale=scale_crop, ratio=ratio_crop
        )
        resized_crop_x = transforms.Resize(original_size)(crop_x.unsqueeze(0)).squeeze(0)
        resized_images.append(resized_crop_x)
        
    resized_images = torch.stack(resized_images)

    return resized_images