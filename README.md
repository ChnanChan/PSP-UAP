# Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior

## Introduction
We propose a novel data-free universal adversarial attack method that leverages semantic samples drawn from a pseudo-semantic prior.  
The implementation is based on PyTorch.

## Preparation
* Our PSP-UAP method has been tested on the following environment:
  - **Operating System**: Ubuntu 20.04  
  - **Python**: 3.8  
  - **PyTorch**: 1.11.0  
  - **Torchvision**: 0.12.0  
  - **CUDA**: 11.3  
* To install the required Python packages, run the following commands:
```bash
conda create -y -n PSP python=3.8
conda activate PSP
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install matplotlib tqdm opencv-python scikit-image imageio
```

* Download the ImageNet validation set from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
* Pre-trained models will be loaded automatically from Torchvision.

​
## Usage
### 1. Training
The commands below initiate training to craft a UAP using a surrogate model with optimal hyperparameters.
After training, the generated UAP will be evaluated for attack performance on predefined target models using the ImageNet validation set.

```bash
python main.py --surrogate_model alexnet --data_path DATA_PATH --p_rate 1.0 --p_active --prior gauss --re_weight --temper 1 --input_transform

python main.py --surrogate_model vgg16 --data_path DATA_PATH --p_rate 1.0 --p_active --prior gauss --re_weight --temper 5 --input_transform

python main.py --surrogate_model vgg19 --data_path DATA_PATH --p_rate 1.0 --p_active --prior gauss --re_weight --temper 5 --input_transform

python main.py --surrogate_model resnet152 --data_path DATA_PATH --p_rate 0.65 --p_active --prior jigsaw --re_weight --temper 3 --input_transform

python main.py --surrogate_model googlenet --data_path DATA_PATH --p_rate 0.55 --p_active --prior jigsaw --re_weight --temper 5 --input_transform

python main.py --surrogate_model resnet50 --data_path DATA_PATH --p_rate 0.7 --p_active --prior jigsaw --re_weight --temper 3 --input_transform --additional_cnn

python main.py --surrogate_model densenet121 --data_path DATA_PATH --p_rate 0.9 --p_active --prior jigsaw --re_weight --temper 10 --input_transform --additional_cnn

python main.py --surrogate_model mobilenet_v3_large --data_path DATA_PATH --p_rate 0.9 --p_active --prior gauss --re_weight --temper 2 --input_transform --additional_cnn

python main.py --surrogate_model inception_v3 --data_path DATA_PATH --p_rate 0.2 --p_active --prior jigsaw --re_weight --temper 3 --input_transform --additional_cnn --delta_size 299
```


### 2. Testing
To evaluate a pre-generated UAP, use the following command:

```bash
python main.py --data_path DATA_PATH --uap_path UAP_PATH
```

Alternatively, to evaluate the UAP on additional CNN models, use the following command:

```bash
python main.py --data_path DATA_PATH --uap_path UAP_PATH --additional_cnn
```
