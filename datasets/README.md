# Data Preparing

1. Access to the synapse multi-organ dataset by signing up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the Abdomen dataset.  
   This can also be done by using the `dowload_data.py` script in the `datasets` directory: 
   ```
      python download_data.py [Synapse username] [Synapse password] [SynapseID of the dataset] [directory where the file should be stored]
   ```
   Using `--help` also displays the SynapseIds of the Abdomen dataset.
   (You probably want to download either `Reg-Training-Testing` or `Reg-Training-Training`.)
2. Preprocess the data:                  
   1. Use the `preprocess_data.py` script in the `datasets` directory. 
      It will clip the image data within the range [-125, 275], normalize it to [0, 1], extract 2D slices from the 3D and save it in the appropriate file formats.
      ```
         python preprocess_data.py [Location of the unzipped abdomen dataset]
      ```
      By default, the data in the target directory won't be overwritten unless the `--overwrite` parameter is passed.  
      For an overview of additional arguments use the `--help` option.
   2.  You can also send an Email directly to jienengchen01 AT gmail.com to request the preprocessed data for reproduction.
2. The directory structure of the whole project is as follows:

```bash
.
├── TransUNet
│   ├──datasets
│   │       ├── dataset_*.py
│   │       ├── download_data.py
│   │       └── preprocess_data.py
│   ├──train.py
│   ├──test.py
│   └──...
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
└── data
    └──Synapse
        ├── test_vol_h5
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train_npz
            ├── case0005_slice000.npz
            └── *.npz
```
