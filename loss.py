import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import torch.storage
from functions import *
from skimage import filters
from skimage.morphology import disk
from strategy import *
from semantic_aug import *
from torchvision.models.inception import InceptionOutputs
debug = False

def get_conv_layers(model):
    '''
    Get all the convolution layers in the network.
    '''
    return [module for module in model.modules() if type(module) == nn.Conv2d]

def l2_layer_loss(model, delta,args,device):
    '''
    Compute the loss of TRM
    '''
    loss = torch.tensor(0.)
    activations = []
    p_activations = []
    remove_handles = []
    def check_zero(tensor):
        if tensor.equal(torch.zeros_like(tensor)):
            return False
        else:
            return True
    def activation_recorder_hook(self, input, output):
        activations.append(output)
        return None
    for conv_layer in get_conv_layers(model):
        handle = conv_layer.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)

    model.eval()
    model.zero_grad()
    model(delta)

    for handle in remove_handles:
        handle.remove()
        
    if args.p_active == True:
        truncate = int(len(activations)* args.p_rate)
        if truncate <=0 and args.p_rate != 0.0:
            truncate += 1
        
        for i in range(truncate):
            ac_tensor = activations[i].reshape(-1)
            ac_tensor = torch.where(ac_tensor > 0, ac_tensor, torch.zeros_like(ac_tensor))
            p_activations.append(ac_tensor)
        

    else:
        for i in range(len(activations)):
            activations[i] = torch.where(activations[i] > 0, activations[i], torch.zeros_like(activations[i])).to(device)
    if args.p_active == True:
        p_loss = sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2+ 1e-9), p_activations)))
        loss = -p_loss
        return loss

    else:
        loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2 + 1e-9), activations)))

        return loss
    
def l2_layer_loss_weight(model, delta, semantic_prior, args,num_copies, device):
    '''
    Compute the loss of TRM
    '''
    loss = torch.tensor(0.)
    activations = []
    p_activations = []
    remove_handles = []
    T = args.temper
    
    with torch.no_grad():
        output_semantic = model(semantic_prior)
        
        
    def check_zero(tensor):
        if tensor.equal(torch.zeros_like(tensor)):
            return False
        else:
            return True
    def activation_recorder_hook(self, input, output):
        activations.append(output)
        return None

    for conv_layer in get_conv_layers(model):
        handle = conv_layer.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)
    
    model.eval()
    model.zero_grad()
    output_delta = model(delta)
        
    weights = kd_loss(output_delta, output_semantic, T=args.temper)
    
    weights = 1/weights
    weights = torch.sqrt(weights)
    
    weights = weights.squeeze(-1)

    for handle in remove_handles:
        handle.remove()
        
    if args.p_active == True:
        truncate = int(len(activations)* args.p_rate)
        if truncate <=0 and args.p_rate != 0.0:
            truncate += 1
        
        
        for i in range(truncate):
            layer_activations = activations[i]
            ac_tensor = layer_activations.reshape(num_copies*args.prior_batch, -1)
            weighted_activations = ac_tensor * weights.detach().view(num_copies*args.prior_batch, 1)
            weighted_activations = weighted_activations.view(-1)
            p_activations.append(weighted_activations)
            

    else:
        for i in range(len(activations)):
            activations[i] = torch.where(activations[i] > 0, activations[i], torch.zeros_like(activations[i])).to(device)
            
    if args.p_active == True:
        p_loss = sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2+ 1e-9), p_activations)))
        loss = -p_loss
        
        return loss

    else:
        loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2 + 1e-9), activations)))

        return loss
    
def kd_loss(student, teacher, T):
    if isinstance(teacher, InceptionOutputs):
        teacher = teacher.logits
    log_s = torch.log_softmax(student / T, dim=-1)
    soft_t = torch.softmax(teacher / T, dim=-1)
    loss = torch.nn.functional.kl_div(log_s, soft_t, reduction='none')*(T**2)
    loss = loss.sum(dim=-1, keepdim=True)
    return loss

    
def get_fooling_rate(model, delta, data_loader, device):
    """
    Computes the fooling rate of the UAP on the dataset.
    """
    flipped = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(normalize(images))
            _, predicted = torch.max(outputs.data, 1)

            adv_images = torch.add(delta, images).clamp(0, 1)
            adv_outputs = model(normalize(adv_images))

            _, adv_predicted = torch.max(adv_outputs.data, 1)

            total += images.size(0)
            flipped += (predicted != adv_predicted).sum().item()

    return flipped / total


def get_rate_of_saturation(delta, xi):
    """
    Returns the proportion of pixels in delta
    that have reached the max-norm limit xi
    """
    return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)


def get_gauss_prior(args):
    '''
    The Gaussian noise is used
    as range-prior to simulate the real image.
    '''

    for i in range(args.prior_batch):
        im = None
        if args.prior == 'gauss':
            im = make_some_noise_gauss(args.std,args.delta_size)
        else:
            return None
        prior = img_preprocess(im = im,size=args.delta_size,augment=True, model_name=args.surrogate_model)
        prior = np.moveaxis(prior, -1, 1)/255
        prior = torch.Tensor(prior)
        if i == 0:
            prior_batch = prior
        else:
            prior_batch = torch.cat([prior_batch, prior], dim=0)


    return prior_batch

def get_jigsaw(img,args,min=0,max=256,filter=False):
    img_shape = torch.zeros_like(img.cpu().detach()).squeeze(0)
    img_batch = torch.zeros_like(img.cpu().detach()).squeeze(0)

    for j in range(args.prior_batch):
        if args.surrogate_model == 'googlenet' or 'inception_v3':
            ximg = shuffle(img_shape, args.fre+2, args.fre, min,max)
        else:
            ximg = shuffle(img_shape, args.fre, args.fre,min,max)

        if filter == True:
            ximg = ximg.numpy()
            for i in range(len(ximg)):
                ximg[i] = filters.median(ximg[i], disk(5))
            ximg = torch.Tensor(ximg)
        ximg = ximg.unsqueeze(0)
        ximg = ximg / 255
        if j == 0:
            img_batch = ximg
        else:
            img_batch = torch.cat([img_batch, ximg], dim=0)
    return img_batch