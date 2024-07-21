# TransUNet
This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## Structure


```bash
.
├── FLA-TransUNet
│   ├──datasets
│   │       └── dataset_*.py
│   ├──lists
│   │       └──...
│   ├──networks
│   │       └──...
│   ├──train.py
│   ├──test.py
│   ├──dataset_preprocess.py
│   └──...
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
└── data
    └──ACDC
        ├── test_npz
        │   ├── patient101_slice000.npz
        │   └── *.npy
        └── train_npz
            ├── patient001_slice000.npz
            └── *.npz
```

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Use [ACDC](https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17) dataset. And please use [dataset_preprocess](FLA-TransUNet/dataset_preprocess) to process the train and test data into npz format and write the names of all the train and test data into train.txt and test.txt respectively.

### 3. Environment

Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
# Softmax Attention(TranUNet)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset ACDC --vit_name R50-ViT-B_16 --max_epochs 50 --n_gpu 1 --attention Attention

# or Linear Attention(FLA-TransUNet)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset ACDC --vit_name R50-ViT-B_16 --max_epochs 50 --n_gpu 1
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
# Softmax Attention(TranUNet)
python test.py --dataset ACDC --vit_name R50-ViT-B_16 --is_savenii --max_epochs 50 --attention Attention --pretrain_path your_path

# or Linear Attention(FLA-TransUNet)
python test.py --dataset ACDC --vit_name R50-ViT-B_16 --is_savenii --max_epochs 50 --pretrain_path your_path
```

## Our models
* [Get our models in this link](https://drive.google.com/drive/folders/15BLoKhiJCowPlYBgbJMsPUa0OEd5wzf5?usp=sharing)

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
