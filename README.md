# Data Extrapolation for Text-to-image Generation on Small Datasets

Official Pytorch implementation for our paper [Data Extrapolation for Text-to-image Generation on Small Datasets](https://arxiv.org/abs/2410.01638) 

![image](https://github.com/user-attachments/assets/a6ff845f-5a98-4bf6-b8e6-2843ca34a00f)

### Examples

![image](https://github.com/user-attachments/assets/59b31d2e-7281-4486-8ab8-814fc4fa3b31)

---
### Requirements
- python 3.12.3
- Pytorch 2.2.2
- RTX 3090 or stronger GPUs(~7 days on CUB and Oxford with 2 3090 TI, ~30 days on COCO with 2 3090 TI)

### Installation

Clone this repo.
```
git clone https://github.com/senmaoy/RAT-Diffusion.git
cd RAT-Diffusion
conda env create -f environment.yml
conda activate RAT
```
If the cuda version doesn't match your GPUs, see https://pytorch.org/get-started/locally/ for a suitable one
### Datasets Preparation
1. Download the preprocessed metadata for [birds_dataset](https://drive.google.com/file/d/1s-R4dDrfry6W8jFv0KFe3Q8_gtCtFzSG/view?usp=drive_link), [birds_extra](https://drive.google.com/file/d/13o3HM7KacIciqJOtIBZco4IRzOebSB5Y/view?usp=drive_link), [flower_dataset](https://drive.google.com/file/d/1nmVmS2dPpHnSFfA1_3WQadtrXzvr-AbH/view?usp=drive_link), [flower_extra](https://drive.google.com/file/d/1o_Qwh0PV6ddbCjCNgUmTWFz2nFulkDBY/view?usp=drive_link),[coco_dataset](https://drive.google.com/file/d/17DvuQ6xeuXYyUboOsp3AIQh8JtbvAUKV/view?usp=drive_link),[coco_extra](https://drive.google.com/file/d/17aubtONziNoHe66hFgrpQsmKOpUtnV2h/view?usp=drive_link) and save them to `dataset/`
2. Download the [bird_dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/) image data and extract them to `dataset/bird/dataset`,Download the [bird_extra](https://drive.google.com/file/d/1oHz3sUPZ_dKDjNOIxZSMRXq-yX2EytXR/view?usp=drive_link) image data and extract them to `dataset/bird/extra`,
3. Download the [flower_dataset](https://drive.google.com/file/d/1cL0F5Q3AYLfwWY7OrUaV1YmTx4zJXgNG/view?usp=sharing) image data and extract them to `dataset/flower/dataset`,Download the [flower_extra](https://drive.google.com/file/d/1e7FdY2Lgfqhg_5R11F6jQzTeqPrPhzlm/view?usp=drive_link) image data and extract them to `dataset/flower/extra`,
4. Download the [coco_dataset](http://cocodataset.org/#download) image data and extract them to `dataset/coco/dataset`,Download the [coco_extra](https://drive.google.com/file/d/1dpFbdQely3MvgS9OFEgtQeqvOieI8a2Y/view?usp=drive_link) image data and extract them to `dataset/coco/extra`,




---
### Training on extrapolated data

**Train RAT-Diffusion models:**
nproc_per_node means the number of GPUs
  - : `torchrun --nnodes=1 --nproc_per_node=2 --master_port=9904 --data-path='path to extra data' train.py`

### Fine-tuning on the original dataset

**Train RAT-Diffusion models:**
nproc_per_node means the number of GPUs
  - : `torchrun --nnodes=1 --nproc_per_node=2 --master_port=9903 --data-path='path to dataset' --ckpt='path to model trained on extra data' train.py`

  - The above is for general purpose 

### Pre-trained models
1. Download the [pre-trained checkpoint](https://drive.google.com/file/d/1kHYKzdNn9n4qu-pNDpB_nYjP8Prne6d4/view?usp=drive_link) for CUB and save it to `./result/`
2. Download the [pre-trained checkpoint](https://drive.google.com/file/d/1pGxht2W1vlvJyXoiEPmAJ1o779-8kevi/view?usp=drive_link) for flower and save it to `./result/`
3. Download the [pre-trained checkpoint](https://drive.google.com/file/d/1Y2NamGpCZFOmZLVAGDh3PRzeG7J-p0Gd/view?usp=drive_link) for coco and save it to `./result/`

### Sampling

**Download Pretrained Model**
nproc_per_node means the number of GPUs
  - : `python sample_simple.py`


### Evaluating

**Download Pretrained Model**
nproc_per_node means the number of GPUs
  - : `torchrun --nnodes=1 --nproc_per_node=2 --master_port=9902 --data-path='path to dataset' sample_ddp.py`



**Evaluate RAT-Diffusion models:**

- We compute FID for CUB and coco using (https://github.com/senmaoy/Inception-Score-FID-on-CUB-and-OXford.git).

**RAW Image Data:**
1. Download the raw bird data for [birds_extra](https://drive.google.com/file/d/1gcXFLC6HurSCAmCpXpaN8l7KDYiH7cYA/view?usp=drive_link) (~50G)
2. Download the raw flower data for [flower_extra](https://drive.google.com/file/d/1VdrIoBcCQa5RrfMNSZVsSjKOSXWIno33/view?usp=drive_link) (~30G)
3. Download the raw coco data for [coco_extra](https://drive.google.com/file/d/1zJart0zaMuPMNgjHl8Mzd-6_y8SjYLsu/view?usp=drive_link)(~900G)
- Note that! We only provide the urls of raw images because they are too big. It will spend 2~3 days to download these images. I will upload more scripts if needed.
---
### Ask for help
I'm strugglling for a job concerning image generation. Any recommendations are welcome!快毕业了，真找不到对口的工作了哥！ 
- Email:  senmaoy@gmail.com
- Wechat: Unsupervised2020
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
