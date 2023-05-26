# PMTrans
Patch-Mix Transformer for Unsupervised Domain Adaptation: A Game Perspective

### CVPR 2023 Highlight

This is a rough version, I will continue to polish it.

### Pretrained Swin-B

- Download [swin_base_patch4_window7_224_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) and put it into `pretrained_models`

### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n swin python=3.7 -y
conda activate swin
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
pip install tensorboard 
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
https://github.com/NVIDIA/apex/issues/1227
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

### Datasets:

- Download the `Office31, Office Home, VisDA and Domainnet` Make a file recording the path and label of image like txt files in `datasets/office_home/`

 ```bash
  $ tree data
  datasets
  ├── ofice_home
  │   ├── Art.txt
  │   ├── Clipart.txt
  │   ├── Product.txt
  │   ├── Real_World.txt
  └── ...
  ```   

### Training:

bash dist_train.sh
