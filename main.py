import argparse
from torchvision import models
import torch
from functions import validate_arguments
from functions import *
import numpy as np
import os
from PSP_UAP import psp_uap, get_fooling_rate

download_path = 'TorchHub/'
torch.hub.set_dir(download_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./',
                        help='Imagenet data path')
    parser.add_argument('--surrogate_model', default='googlenet',
                        help='The substitute network eg. vgg19')
    parser.add_argument('--target_model', default='vgg16',
                        help='The target model eg. vgg19')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='The batch size to use for testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--val_dataset_name', default='imagenet',choices=['imagenet'],
                        help='The dataset to be used as test')

    parser.add_argument('--p_active', action="store_true",
                        help='maximize the positive activation the conv layer')
    parser.add_argument('--p_rate', default=0.65, type=float,
                        help='positive proportion of conv layer used')
    
    
    parser.add_argument('--semantic_prior', action='store_true',
                        help='using pseudo-semantic prior')
    parser.add_argument('--input_transform', action='store_true',
                        help='using input transformation')
    
    
    parser.add_argument('--angle', default=6, type=float,
                        help='value of the angle')
    parser.add_argument('--scale_t_low', default=0.8, type=float,
                        help='minimum value of scaling factor')
    parser.add_argument('--scale_t_high', default=4, type=float,
                        help='maximum value of scaling factor')
    parser.add_argument('--shuffle_block', default=2, type=int,
                        help='the number of blocks for shuffling')
    parser.add_argument('--temper', default=0.7, type=float,
                        help='temperature parameter')
    parser.add_argument('--num_copies', default=10, type=int,
                        help='the number of semantic samples')
    parser.add_argument('--re_weight', action="store_true",
                        help='chose sample rewighting')
    parser.add_argument('--additional_cnn', action="store_true",
                        help='chose additional_cnn models')



    parser.add_argument('--seed', default=126, type=int,
                        help='random seed')
    parser.add_argument('--epsilon', default=10/255, type=float,
                        help='the infinite norm limitation of UAP')
    parser.add_argument('--delta_size', default=224, type=int,
                        help='the size of delta')
    parser.add_argument('--uap_lr', default=0.1, type=float,
                        help='the leraning rate of UAP')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='the leraning rate of UAP')
    

    parser.add_argument('--prior', default='gauss',choices=['gauss','jigsaw','None'], type=str,
                        help='the range prior of perturbations')
    parser.add_argument('--prior_batch', default=1, type=int,
                        help='the batch size of prior')
    parser.add_argument('--std', default=10, type=int,
                        help='initialize the standard deviation of gaussian noise')
    parser.add_argument('--fre', default=1, type=int,
                        help='initialize the frequency of jigsaw image')
    parser.add_argument('--uap_path', default=None, type=str,
                        help='the path of UAP')
    parser.add_argument('--gauss_t0', default=400, type=int,
                        help='the threshold to adjust the increasing rate of standard deviation(gauss)')
    parser.add_argument('--gauss_gamma', default=10, type=int,
                        help='the step size(gauss)')
    parser.add_argument('--jigsaw_t0', default=600, type=int,
                        help='the threshold to adjust the increasing rate of standard deviation(jigsaw)')
    parser.add_argument('--jigsaw_gamma', default=1, type=int,
                        help='the step size(jigsaw)')
    parser.add_argument('--jigsaw_end_iter', default=4200, type=int,
                        help='the iterations which stop the increment of frequency(jigsaw)')

    
    args = parser.parse_args()
    validate_arguments(args.surrogate_model)
    if args.additional_cnn:
        target_models = [models.resnet50, models.densenet121, models.mobilenet_v3_large, models.inception_v3]
    else:
        target_models = [models.alexnet, models.vgg16, models.vgg19, models.resnet152, models.googlenet]
    
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = prepare_for_model(args,args.surrogate_model,device,initialize=True)

    if args.uap_path == None:
        if args.re_weight:
            filename = f"results/additional/uap_reweight_{args.temper}_{args.angle}_{args.num_copies}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}_{args.input_transform}"
            logname = f"results/additional/uap_reweight_{args.temper}_{args.angle}_{args.num_copies}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}_{args.input_transform}.txt"
                    
            if args.input_transform is None:
                filename = f"results/supple/uap_reweight_{args.temper}_{args.num_copies}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}"
                logname = f"results/supple/uap_reweight_{args.temper}_{args.num_copies}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}.txt"
                    
        else:
            filename = f"results/aux_trans/uap_{args.num_copies}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}"
            logname = f"results/aux_trans/uap_{args.num_copies}_{args.prior_batch}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}.txt"
            if args.input_transform is not None:
                filename = f"results/supple/uap_T_{args.num_copies}_{args.input_transform}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}"
                logname = f"results/supple/uap_T_{args.num_copies}_{args.input_transform}_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                    f"_seed={args.seed}_prior={args.prior}.txt"

        os.makedirs(os.path.dirname(filename),exist_ok=True)
        print("start:", filename)
        
        uap = psp_uap(model, args, device,prior=args.prior)
        
        print("saved on ", filename)
        np.save(filename, uap.cpu().detach().numpy())
        print(f'the UAP of surrogate model {args.surrogate_model} is crfted.')
    else:
        uap = get_uap(args.uap_path,device)
        logname = os.path.splitext(args.uap_path)[0] + '.txt'

    with open(logname, 'a') as log_file:
        for t in target_models:
            loader = get_data_loader(args.val_dataset_name, args.data_path,surrogate_model=t.__name__,batch_size=args.batch_size,shuffle=True,analyze=True)
            noise = uap
            if 'inception' in t.__name__:
                if uap.shape[-1] != 299:
                    noise = torch.nn.functional.interpolate(uap, (299,299))
            else:
                if uap.shape[-1] == 299:
                    noise = uap[...,37:261,37:261]
                    
            target_model = t(pretrained=True).to(device)
            target_model.eval()
            final_fooling_rate = get_fooling_rate(target_model,torch.clamp(noise,-args.epsilon,args.epsilon),loader,device)
            print(f'the FR of UAP ({args.surrogate_model}) on ({t.__name__}) is {final_fooling_rate*100}')
            log_file.write(f'the FR of UAP ({args.surrogate_model}) on ({t.__name__}) is {final_fooling_rate}\n')

    print('finish')


if __name__ == '__main__':
    main()