# TransUNet
TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to "./datasets/README.md" for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data.

### 3. Train/Test

- run the train script on synapse dataset

```bash
python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- run the test script on synapse dataset

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)

## Citations

```bibtex
@article{xxx,
  title={TransUNet},
  author={xxx},
  journal={arXiv preprint arXiv:xxx},
  year={2021}
}
```