# Official pytorch implementation of our paper entitled: Data Extrapolation for Text-to-image Generation on Small Datasets
# Recurrent-Affine-Transformation-for-Text-to-image-Synthesis

Official Pytorch implementation for our paper [Data Extrapolation for Text-to-image Generation on Small Datasets]([https://arxiv.org/abs/2204.10482](https://arxiv.org/abs/2410.01638)) 

![image](https://github.com/user-attachments/assets/605ed437-89dd-4ca8-901f-01ccd9771689)

### Examples

![image](https://github.com/user-attachments/assets/c4abaf29-ef06-4b8f-a139-0bf6a55f6152)

---
### Requirements
- python 3.12.3
- Pytorch 2.2.2
- RTX 3090 or stronger GPUs

### Installation

Clone this repo.
```
git clone https://github.com/senmaoy/RAT-Diffusion.git
cd RAT-Diffusion
conda env create -f environment.yml
conda activate RAT
```

### Datasets Preparation
1. Download the preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`.Raw text data of CUB dataset is avaiable [here](https://drive.google.com/file/d/1KyTQVo67izP4NEAAZBRnqrGG3yRh3azD/view?usp=sharing)
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
4. Download [flower](https://drive.google.com/file/d/1cL0F5Q3AYLfwWY7OrUaV1YmTx4zJXgNG/view?usp=sharing) dataset and extract the images to `data/flower/`.	Raw text data of flower dataset is avaiable [here](https://drive.google.com/file/d/1G4QRcRZ_s57giew6wgnxemwWRDb-3h5P/view?usp=sharing)




---
### Training on extrapolated data

**Train RAT-Diffusion models:**
nproc_per_node means the number of GPUs
  - : `torchrun --nnodes=1 --nproc_per_node=2 --master_port=9904 --data-path='path to extra data' train.py`

### Fine-tuning on the original dataset

**Train RAT-Diffusion models:**
nproc_per_node means the number of GPUs
  - : `torchrun --nnodes=1 --nproc_per_node=2 --master_port=9903 --data-path='path to dataset' --ckpt='path to model trained on extra data' train.py`

### Pre-trained models
1. Download the [pre-trained checkpoint](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V) for CUB and save it to `../bird/`
2. Download the [pre-trained checkpoint](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ) for coco and save it to `../bird/`
3. Download the [pre-trained checkpoint](https://drive.google.com/file/d/1Gb5jRhSN9QGgmACNnZvwJMbDLDuVqffp/view?usp=sharing) for flower and save it to `../bird/`

### Sampling

**Dwonload Pretrained Model**
nproc_per_node means the number of GPUs
  - : `python sample_simple.py`


### Evaluating

**Dwonload Pretrained Model**
nproc_per_node means the number of GPUs
  - : `torchrun --nnodes=1 --nproc_per_node=2 --master_port=9902 --data-path='path to dataset' sample_ddp.py`



**Evaluate RAT-GAN models:**

- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).
- We compute FID for CUB and coco using (https://github.com/senmaoy/Inception-Score-FID-on-CUB-and-OXford.git). 

---
### Citing RAT-Diffusion

If you find RAT-Diffusion useful in your research, please consider citing our paper:

```
@article{ye2022recurrent,
  title={Recurrent Affine Transformation for Text-to-image Synthesis},
  author={Ye, Senmao and Liu, Fei and Tan, Minkui},
  journal={arXiv preprint arXiv:2204.10482},
  year={2022}
}
```
If you are interseted, join us on Wechat group where a dozen of t2i partners are waiting for you! If the QR code is expired, you can add this wechat: Unsupervised2020

![image](https://github.com/user-attachments/assets/e08a15f8-0692-4c6a-959f-67f5b03061e4)

The code is released for academic research use only. Please contact me through senmaoy@gmail.com

**Reference**
- [RAT-GAN: Recurrent Affine Transformation for Text-to-Image Synthesis](https://arxiv.org/abs/2204.10482) [[code]](https://github.com/tobran/DF-GAN.git)
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) [[code]](https://github.com/facebookresearch/DiT)
- [SMM: Score Mismatching for Generative Modeling](https://arxiv.org/abs/2309.11043) [[code]](https://github.com/senmaoy/Score-Mismatching)
