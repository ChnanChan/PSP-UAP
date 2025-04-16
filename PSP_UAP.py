import torch
import torch.nn.functional
import torch.optim as optim
import numpy as np
import torch.storage
from tqdm import tqdm
from functions import *
from strategy import *
from semantic_aug import *
from loss import *

debug = False

def psp_uap(model, args, device, prior=False):
    max_iter = 10000
    size = args.delta_size

    sat_threshold = 0.00001
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min = 0.5
    sat_should_rescale = False
    num_copise = args.num_copies
    
    iter_since_last_fooling = 0
    iter_since_last_best = 0
    best_fooling_rate = 0

    xi_min = -args.epsilon
    xi_max = args.epsilon
    iter_num = 0
    max_angle = args.angle
    
    delta = (xi_min - xi_max) * torch.rand((1, 3, size, size), device=device) + xi_max
    delta.requires_grad = True

    print(f"Initial norm: {torch.norm(delta, p=np.inf)}")

    optimizer = optim.Adam([delta], lr=args.uap_lr)

    val_loader,_ = get_data_loader(args.val_dataset_name, args.data_path, args.surrogate_model, batch_size=args.batch_size)
    for i in tqdm(range(max_iter)):
        optimizer.zero_grad()
        iter_num +=1
        iter_since_last_fooling += 1
        
        
        if prior != None:
            if prior == 'gauss':
                args = curriculum_strategy_gauss(iter_num,args)
                random_batch = get_gauss_prior(args=args)
            elif prior == 'jigsaw':
                if args.surrogate_model == 'resnet152':
                    args = curriculum_strategy_jigsaw_resnet152(iter_num,args)
                elif (args.surrogate_model == 'googlenet') or (args.surrogate_model == 'inception_v3'):
                    args = curriculum_strategy_jigsaw_googlenet(iter_num,args)
                else:
                    args = curriculum_strategy_jigsaw(iter_num,args)
                random_batch = get_jigsaw(delta,args,filter=True)
            if random_batch!=None:
                example_prior = delta + random_batch.to(device)
            else:
                example_prior = delta
        else:
            example_prior = delta    
            
        semantic_priors = example_prior.repeat(num_copise, 1, 1, 1).to(device)
        for j in range(semantic_priors.size(0)):            
                semantic_priors[j] = random_crop_and_resize(semantic_priors[j].unsqueeze(0), 
                                                                        prior = random_batch,
                                                                        scale_crop=(0.08, 1),  
                                                                        ratio_crop=(3./4., 4./3.)).squeeze(0)
        
        semantic_delta = semantic_priors.detach() + delta.repeat(num_copise*args.prior_batch, 1, 1, 1)
        semantic_priors_kd = semantic_priors.clone().detach()
        if args.input_transform:
            for j in range(semantic_delta.size(0)):
                prob = torch.rand(1)
                if prob.item() < 1/3:
                    semantic_delta[j], semantic_priors_kd[j] = rotate_fill_prior(semantic_delta[j], semantic_priors_kd[j], args, random_batch, device=device)
                
                elif 1/3 <= prob.item() < 2/3:
                    semantic_delta[j],  semantic_priors_kd[j] = scaling_transform(semantic_delta[j], semantic_priors_kd[j], args, device=device)
                
                else:
                    semantic_delta[j],  semantic_priors_kd[j] = shuffle_only(semantic_delta[j], semantic_priors_kd[j], num_block=args.shuffle_block)
                
        model.zero_grad()
        if args.re_weight != True:
            loss = l2_layer_loss(model, semantic_delta, args, device)
        else:
            loss = l2_layer_loss_weight(model, semantic_delta, semantic_priors_kd, args, num_copise, device)
        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(xi_min, xi_max)

        sat_prev = np.copy(sat)
        sat = get_rate_of_saturation(delta.cpu().detach().numpy(), xi_max)
        sat_change = np.abs(sat - sat_prev)

        if sat_change < sat_threshold and sat > sat_min:
            if debug:
                print(f"Saturated delta in iter {i} with {sat} > {sat_min}\nChange in saturation: {sat_change} < {sat_threshold}\n")
            sat_should_rescale = True

        if iter_since_last_fooling > 400 or (sat_should_rescale and iter_since_last_fooling > 200):
            iter_since_last_fooling = 0

            print("\nGetting latest fooling rate...")

            current_fooling_rate = get_fooling_rate(model, torch.clamp(delta,xi_min,xi_max), val_loader, device)
            print(f"\nLatest fooling rate: {current_fooling_rate}")

            if current_fooling_rate > best_fooling_rate:
                print(f"Best fooling rate thus far: {current_fooling_rate}")
                best_fooling_rate = current_fooling_rate
                best_uap = delta
            else:
                iter_since_last_best += 1
            
            
            if iter_since_last_best >= args.patience_interval:
                break

        if sat_should_rescale:
            if iter_since_last_best < args.patience_interval-1:
                with torch.no_grad():
                    delta.data = delta.data/2
            else:
                with torch.no_grad():
                    delta.data = delta.data*0.8           
            sat_should_rescale = False
    
    return best_uap
