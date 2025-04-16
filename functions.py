import numpy as np
import cv2
import random
import math
import torch.utils
from skimage.transform import resize
from imageio import imread
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

TORCH_HUB_DIR = './TorchHub'

def normalize(x):
    """
    Normalizes a batch of images with size (batch_size, 3, height, width)
    by mean and std dev expected by PyTorch models
    """
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (x - mean.type_as(x)[None,:,None,None]) / std.type_as(x)[None,:,None,None]

def validate_arguments(test_model):
    # models = ['vgg16', 'vgg19', 'googlenet', 'alexnet', 'resnet152','resnet18','resnet34', 'inception_v3']
    models = ['vgg16', 'vgg19', 'resnet50', 'resnet152', 'alexnet', 'densenet121', 'mobilenet_v3_large', 'inception_v3', 'vgg16', 'googlenet', 'vit_b_16', 'convnext_large']
    if not (test_model in models):
        print ('Argument Error: invalid network')
        exit(-1)

def get_uap(path,device):
    uap = np.load(path)
    uap = torch.tensor(uap, device=device)
    return uap

def prepare_for_model(args,model_name,device,initialize=True):
    '''
    Return a pretrained model on device.
    '''
    if initialize == True:
        if args.val_dataset_name == 'imagenet':
            model = eval("torchvision.models.{}(pretrained=True)".format(model_name)).to(device)
        args.delta_size = 224
        if 'inception' in model_name:
            args.delta_size = 299
    if args.val_dataset_name == 'imagenet':
        model = eval("torchvision.models.{}(pretrained=True)".format(model_name)).to(device)
    return model

def get_data_loader(dataset_name, data_path, surrogate_model=None, batch_size=64, shuffle=False,analyze=False):
    """
    Returns a dataLoader with validation or test images for dataset name.
    """
    if 'inception' in surrogate_model:
        center_crop = 299
        transform = transforms.Compose([
        transforms.Resize(center_crop),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        ])
        
    else:
        center_crop =224
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
        ])

    if dataset_name == 'imagenet':
        val_dataset =  datasets.ImageFolder(
                data_path,
                transform=transform
                )
        if analyze == True:
            dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=64)
            return dataset
        train_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [1000, 49000])

        test_dataset, _=torch.utils.data.random_split(test_dataset, [1000, 48000])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=64)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=64)

        return train_loader,test_loader

def flip(I, flip_p):
    if flip_p > 0.5:
        return I[:, ::-1, :]
    else:
        return I

def blur(img_temp, blur_p, blur_val):
    if blur_p > 0.5:
        return cv2.GaussianBlur(img_temp, (blur_val, blur_val), 1)
    else:
        return img_temp

def rotate(img_temp, rot, rot_p):
    if(rot_p > 0.5):
        rows, cols, ind = img_temp.shape
        h_pad = int(rows*abs(math.cos(rot/180.0*math.pi)) +
                    cols*abs(math.sin(rot/180.0*math.pi)))
        w_pad = int(cols*abs(math.cos(rot/180.0*math.pi)) +
                    rows*abs(math.sin(rot/180.0*math.pi)))
        final_img = np.zeros((h_pad, w_pad, 3))
        final_img[(h_pad-rows)//2:(h_pad+rows)//2, (w_pad-cols) //
                  2:(w_pad+cols)//2, :] = np.copy(img_temp)
        M = cv2.getRotationMatrix2D((w_pad//2, h_pad//2), rot, 1)
        final_img = cv2.warpAffine(
            final_img, M, (w_pad, h_pad), flags=cv2.INTER_NEAREST)
        part_denom = (math.cos(2*rot/180.0*math.pi))
        w_inside = int((cols*abs(math.cos(rot/180.0*math.pi)) -
                        rows*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        h_inside = int((rows*abs(math.cos(rot/180.0*math.pi)) -
                        cols*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        final_img = final_img[(h_pad-h_inside)//2:(h_pad+h_inside)//2,
                              (w_pad - w_inside)//2:(w_pad + w_inside)//2, :].astype('uint8')
        return final_img
    else:
        return img_temp

def rand_crop(img_temp, dim=224):
    h = img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h = trig_w = False
    if(h > dim):
        h_p = int(random.uniform(0, 1)*(h-dim))
        img_temp = img_temp[h_p:h_p+dim, :, :]
    elif(h < dim):
        trig_h = True
    if(w > dim):
        w_p = int(random.uniform(0, 1)*(w-dim))
        img_temp = img_temp[:, w_p:w_p+dim, :]
    elif(w < dim):
        trig_w = True
    if(trig_h or trig_w):
        pad = np.zeros((dim, dim, 3), dtype='uint8')
        pad[:, :, 0] += 127
        pad[:, :, 1] += 127
        pad[:, :, 2] += 127
        pad[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
        return pad
    else:
        return img_temp

def randomizer(img_temp):
    dim = 224
    flip_p = random.uniform(0, 1)
    scale_p = random.uniform(0, 1)
    blur_p = random.uniform(0, 1)
    blur_val = random.choice([3, 5, 7, 9])
    rot_p = np.random.uniform(0, 1)
    rot = random.choice([-10, -7, -5, -3, 3, 5, 7, 10])
    if(scale_p > .5):
        scale = random.uniform(0.75, 1.5)
    else:
        scale = 1
    if(img_temp.shape[0] < img_temp.shape[1]):
        ratio = dim*scale/float(img_temp.shape[0])
    else:
        ratio = dim*scale/float(img_temp.shape[1])
    img_temp = cv2.resize(
        img_temp, (int(img_temp.shape[1]*ratio), int(img_temp.shape[0]*ratio)))
    img_temp = flip(img_temp, flip_p)
    img_temp = rotate(img_temp, rot, rot_p)
    img_temp = blur(img_temp, blur_p, blur_val)
    img_temp = rand_crop(img_temp)
    return img_temp

def img_preprocess(im,img_path=None, size=224, augment=False, model_name='1'):
    '''
    A generic preprocessor for the range prior
    '''
    mean = [127.5,127.5,127.5]
    #print(im.shape)
    if img_path == None:
        img = im
    else:
        img = imread(img_path)
    if augment:
        img = randomizer(img)
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    if 'inception' in model_name:
        resFac = 299.0/min(img.shape[:2])
    else:
        resFac = 256.0/min(img.shape[:2])
    newSize = list(map(int, (img.shape[0]*resFac, img.shape[1]*resFac)))
    img = resize(img, newSize, mode='constant', preserve_range=True)
    offset = [newSize[0]/2.0 -
              np.floor(size/2.0), newSize[1]/2.0-np.floor(size/2.0)]
    #print(img.shape)
    img = img[int(offset[0]):int(offset[0])+size,
              int(offset[1]):int(offset[1])+size, :]
    #print(img.shape)
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img

def downsample(inp):
    return np.reshape(inp[1:-2, 1:-2, :], [1, 224, 224, 3])

def upsample(inp):
    out = np.zeros([227, 227, 3])
    out[1:-2, 1:-2, :] = inp
    out[0, 1:-2, :] = inp[0, :, :]
    out[-2, 1:-2, :] = inp[-1, :, :]
    out[-1, 1:-2, :] = inp[-1, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, -2, :] = out[:, -3, :]
    out[:, -1, :] = out[:, -3, :]
    return np.reshape(out, [1, 227, 227, 3])

def make_some_noise_gauss(std,size):
    '''
    The range prior for input with gauss noise
    '''
    mean = [127.5,127.5,127.5]
    sd = [std,std+10,std+20]
    im = np.zeros((size, size, 3))

    for i in range(3):
        im[:, :, i] = np.random.normal(
            loc=mean[i], scale=sd[i], size=(size, size))

    im = np.clip(im, 0, 255)
    return im

def get_ranlist(num,min=0,max=225):
    list = [min]
    for i in range(num):
        x = random.randrange(min,max,step=1)
        list.append(x)
    return sorted(list)

def shuffle(img,wide=5,high=7,min=0,max=256,bound=224):
    #assert mode in [0, 1], 'check shuffle mode'

    wide_list = get_ranlist(wide,max=bound+1)
    high_list = get_ranlist(high,max=bound+1)
    for i in range(len(wide_list)):
        w_start = wide_list[i]
        if i < len(wide_list)-1:
            w_end = wide_list[i + 1]
        else:
            w_end = bound
        for j in range(len(high_list)):
            h_start = high_list[j]
            if j <len(high_list)-1:
                h_end = high_list[j+1]
            else:
                h_end = bound
            img[0, w_start:w_end, h_start:h_end] = random.randrange(min,max,1)
            img[1, w_start:w_end, h_start:h_end] = random.randrange(min,max,1)
            img[2, w_start:w_end, h_start:h_end] = random.randrange(min,max,1)
    return img



