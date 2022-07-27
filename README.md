# PowerVQE: An Open Framework for Quality Enhancement of Compressed Videos

## 0. Content

- [PowerVQE: An Open Framework for Quality Enhancement of Compressed Videos](#powervqe-an-open-framework-for-quality-enhancement-of-compressed-videos)
  - [0. Content](#0-content)
  - [1. Introduction](#1-introduction)
  - [2. Performance](#2-performance)
  - [3. Environment](#3-environment)
  - [4. Datasets](#4-datasets)
    - [4.1 LDVv2 Dataset](#41-ldvv2-dataset)
    - [4.2 MFQEv2 Dataset](#42-mfqev2-dataset)
    - [4.3 Create a Symbolic Link](#43-create-a-symbolic-link)
  - [5. Training](#5-training)
    - [5.1 Special Case of the MFQEv2 models](#51-special-case-of-the-mfqev2-models)
  - [6. Test](#6-test)
    - [6.1 Special Case of the MFQEv2 models](#61-special-case-of-the-mfqev2-models)
    - [6.2 Special Case of BasicVSR++ over the MFQEv2 Dataset](#62-special-case-of-basicvsr-over-the-mfqev2-dataset)
    - [6.3 Special Cases of DCAD and DnCNN](#63-special-cases-of-dcad-and-dncnn)
    - [6.4 PSNR Calculation](#64-psnr-calculation)
  - [7. Q&A](#7-qa)
    - [7.1 Main Differences from the Original Papers](#71-main-differences-from-the-original-papers)
    - [7.2 How to Use the Latest MMEditing](#72-how-to-use-the-latest-mmediting)
    - [7.3 Support for Image Datasets](#73-support-for-image-datasets)
    - [7.4 Support for LMDB](#74-support-for-lmdb)
    - [7.5 Use Pre-commit Hook to Polish Code](#75-use-pre-commit-hook-to-polish-code)
  - [8. Licenses](#8-licenses)

## 1. Introduction

We implement some widely-used quality enhancement approaches for compressed videos based on the powerful [MMEditing](https://github.com/open-mmlab/mmediting) project. These approaches are commonly used for comparison in this field are as follows:

- [STDF (AAAI 2020)](https://github.com/ryanxingql/stdf-pytorch): Enhancing compressed videos with feature-wise deformable convolutions, instead of frame-wise motion estimation and compensation.
- [MFQEv2 (TPAMI 2019)](https://github.com/ryanxingql/mfqev2.0): Enhancing frames in compressed videos by taking advantage of neighboring good-quality frames.
- [DCAD (DCC 2017)](https://ieeexplore.ieee.org/abstract/document/7923714/): The first approach to post-process the decoded videos with a deep-learning-based method. It also sets up good experimental settings.
- [DnCNN (TIP 2017)](https://arxiv.org/abs/1608.03981): Widely used approach for decompression and denoising.

We also implement some SR baseline models for quality enhancement as follows:

- [BasicVSR++ (CVPR 2022)](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr_plusplus/README.md): Winner of the NTIRE 2021 VSR challenge.
- [EDVR (CVPRW 2019)](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/edvr/README.md): Winner of the NTIRE 2019 VSR challenge.

Furthermore, we incorporate some image-oriented models into PowerVQE, which are detailed in [this section](#73-support-for-image-datasets):

- [RBQE (ECCV 2020)](https://arxiv.org/abs/2006.16581)
- [CBDNet (CVPR 2019)](https://arxiv.org/abs/1807.04686)
- [RDN (CVPR 2018)](https://arxiv.org/abs/1802.08797)
- [AR-CNN (ICCV 2015)](https://arxiv.org/abs/1504.06993)

## 2. Performance

[\[TensorBoard\]](https://github.com/ryanxingql/powervqe/wiki/TensorBoard) [\[Pre-trained models\]](https://github.com/ryanxingql/powervqe/releases/)

RGB-PSNR results on the test set of the [LDVv2 dataset](https://arxiv.org/abs/2204.09314) for the NTIRE 2022 video quality enhancement challenge are as follows:

<details>
<summary>[Table Here]</summary>

| Index | Video name | Width | Height | Frames | Frame rate | LQ     | DCAD   | DnCNN  | STDF   | MFQEv2 | EDVR   | BasicVSR++ |
| ----- | ---------- | ----- | ------ | ------ | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ---------- |
| 1     | 001        | 960   | 536    | 250    | 25         | 32.433 | 32.872 | 32.901 | 32.971 | 33.039 | 33.163 | 33.342     |
| 2     | 002        | 960   | 536    | 300    | 30         | 30.182 | 30.587 | 30.624 | 30.719 | 30.824 | 30.978 | 31.686     |
| 3     | 003        | 960   | 536    | 300    | 30         | 29.047 | 29.670 | 29.700 | 29.678 | 29.981 | 29.890 | 30.488     |
| 4     | 004        | 960   | 536    | 250    | 25         | 34.313 | 35.105 | 35.209 | 35.324 | 35.448 | 35.774 | 36.196     |
| 5     | 005        | 960   | 536    | 600    | 60         | 30.492 | 30.919 | 30.953 | 31.048 | 31.108 | 31.335 | 31.693     |
| 6     | 006        | 960   | 536    | 300    | 30         | 28.994 | 29.450 | 29.503 | 29.484 | 29.588 | 29.785 | 30.117     |
| 7     | 007        | 960   | 536    | 240    | 24         | 28.845 | 29.359 | 29.373 | 29.472 | 29.581 | 29.696 | 29.865     |
| 8     | 008        | 960   | 504    | 240    | 24         | 31.798 | 32.640 | 32.706 | 32.831 | 32.835 | 33.119 | 33.389     |
| 9     | 009        | 960   | 536    | 250    | 25         | 29.857 | 30.628 | 30.656 | 30.751 | 30.842 | 30.999 | 31.140     |
| 10    | 010        | 960   | 504    | 600    | 60         | 34.646 | 35.531 | 35.598 | 35.621 | 35.591 | 35.666 | 35.634     |
| 11    | 011        | 960   | 536    | 600    | 60         | 26.300 | 26.662 | 26.735 | 26.718 | 26.666 | 26.804 | 26.926     |
| 12    | 012        | 960   | 536    | 600    | 60         | 22.867 | 23.334 | 23.324 | 23.467 | 23.437 | 23.319 | 23.391     |
| 13    | 013        | 960   | 536    | 300    | 30         | 28.424 | 28.787 | 28.793 | 28.886 | 29.222 | 29.190 | 29.805     |
| 14    | 014        | 960   | 536    | 300    | 30         | 31.214 | 31.820 | 31.853 | 31.909 | 32.022 | 32.209 | 32.451     |
| 15    | 015        | 960   | 536    | 300    | 30         | 29.032 | 29.495 | 29.527 | 29.497 | 29.601 | 29.681 | 30.461     |
|       | Ave.       |       |        |        |            | 29.896 | 30.457 | 30.497 | 30.558 | 30.652 | 30.774 | 31.106     |
|       | Delta PSNR |       |        |        |            |        | 0.561  | 0.601  | 0.662  | 0.756  | 0.878  | 1.209      |

</details>

Y-PSNR results on the test set (QP=37) of the [MFQEv2 dataset](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset) are as follows:

<details>
<summary>[Table Here]</summary>

| Index | Video Name       | Width | Height | Frames | Frame rate | LQ     | DCAD   | DnCNN  | STDF   | STDF-Y | MFQEv2 | EDVR   | BasicVSR++ |
| ----- | ---------------- | ----- | ------ | ------ | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---------- |
| 1     | Kimono           | 1920  | 1080   | 240    | 24         | 34.397 | 34.661 | 34.686 | 34.737 | 34.908 | 35.149 | 35.299 |            |
| 2     | Park Scene       | 1920  | 1080   | 240    | 24         | 31.629 | 31.800 | 31.809 | 31.836 | 32.002 | 32.146 | 32.137 |            |
| 3     | Cactus           | 1920  | 1080   | 500    | 50         | 32.486 | 32.765 | 32.779 | 32.835 | 33.023 | 33.069 | 33.172 |            |
| 4     | BQ Terrace       | 1920  | 1080   | 600    | 60         | 31.289 | 31.653 | 31.656 | 31.676 | 31.764 | 31.766 | 31.812 |            |
| 5     | Basketball Drive | 1920  | 1080   | 500    | 50         | 33.382 | 33.723 | 33.751 | 33.796 | 33.929 | 34.004 | 34.143 |            |
| 6     | Race Horses      | 832   | 480    | 300    | 30         | 30.161 | 30.475 | 30.496 | 30.495 | 30.574 | 30.718 | 30.723 |            |
| 7     | BQ Mall          | 832   | 480    | 600    | 60         | 31.353 | 31.777 | 31.803 | 31.842 | 32.100 | 32.143 | 32.225 |            |
| 8     | Party Scene      | 832   | 480    | 500    | 50         | 27.925 | 28.259 | 28.271 | 28.387 | 28.552 | 28.480 | 28.498 |            |
| 9     | Basketball Drill | 832   | 480    | 500    | 50         | 31.646 | 32.140 | 32.171 | 32.231 | 32.384 | 32.340 | 32.482 |            |
| 10    | Race Horses      | 416   | 240    | 300    | 30         | 29.375 | 29.750 | 29.767 | 29.792 | 29.941 | 30.130 | 30.138 |            |
| 11    | BQ Square        | 416   | 240    | 600    | 60         | 28.365 | 28.947 | 28.957 | 29.154 | 29.338 | 29.239 | 29.198 |            |
| 12    | Blowing Bubbles  | 416   | 240    | 500    | 50         | 27.876 | 28.188 | 28.202 | 28.342 | 28.509 | 28.486 | 28.479 |            |
| 13    | Basketball Pass  | 416   | 240    | 500    | 50         | 30.551 | 30.960 | 30.986 | 31.065 | 31.299 | 31.367 | 31.496 |            |
| 14    | Four People      | 1280  | 720    | 600    | 60         | 34.673 | 35.231 | 35.265 | 35.297 | 35.516 | 35.387 | 35.573 |            |
| 15    | Johnny           | 1280  | 720    | 600    | 60         | 36.412 | 36.917 | 36.960 | 36.983 | 37.210 | 36.990 | 37.228 |            |
| 16    | Kristen And Sara | 1280  | 720    | 600    | 60         | 35.949 | 36.530 | 36.553 | 36.620 | 36.875 | 36.695 | 36.864 |            |
|       | Ave.             |       |        |        |            | 31.717 | 32.111 | 32.132 | 32.193 | 32.370 | 32.382 | 32.467 |            |
|       | Delta PSNR       |       |        |        |            |        | 0.394  | 0.415  | 0.476  | 0.653  | 0.665  | 0.750  |            |

</details>

Note:

1. For simplicity, all models except for the STDF-Y model are trained with RGB data. The Y-PSNR results are obtained from the RGB data.

2. The STDF model trained on RGB data performs worse than the STDF-Y model. I guess the increasing channel number (Y->RGB) also increases the learning difficulty of DCN (which learns offset and mask for each channel separately).

3. (22-07-28) I am waiting for a 32GB-memory GPU to test BasicVSR++ over the MFQEv2 dataset.

## 3. Environment

PowerVQE depends on [PyTorch](https://pytorch.org), [MMCV](https://github.com/open-mmlab/mmcv) and some other packages. Here is my code:

```bash
git clone https://github.com/ryanxingql/powervqe.git --depth=1
cd powervqe/mmediting/

conda create -n powervqe python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y && \
conda activate powervqe

pip3 install openmim && mim install mmcv-full
pip3 install -e .

pip3 install scipy tqdm lmdb
#pip3 install setuptools==59.5.0
```

Layout:

```text
powervqe/
|-- mmediting/
`-- toolbox_data/
```

## 4. Datasets

We provide the built FFmpeg 5.0.1 for converting MKV to PNG. You can download it on [Dropbox](https://www.dropbox.com/s/dr4pxs5fjzri6z1/ffmpeg-release-amd64-static.tar.xz?dl=0) or [Baidu Pan](https://pan.baidu.com/s/1Zw_ozBf6Dd1r2K2Q-4kIaw?pwd=13l0).

```bash
cd <your-ffmpeg-dir>

# download ffmpeg-release-amd64-static.tar.xz

tar -xvf ffmpeg-release-amd64-static.tar.xz
```

You can also use your own FFmpeg.

### 4.1 LDVv2 Dataset

We use the [LDVv2 dataset](https://arxiv.org/abs/2204.09314) for training, validation, and testing. This dataset is also used for the NTIRE 2022 challenge on video quality enhancement.

The LDVv2 dataset includes 240 videos for training, 15 videos for validation, and 15 videos for testing. Each raw video is compressed by HM 16.20 with the LDP, QP=37 setting.

You can first download this dataset on [Dropbox](https://www.dropbox.com/sh/3enteycl9439gzq/AABgfXApw4yQaZFAB5gyAmQha?dl=0) or [Baidu Pan](https://pan.baidu.com/s/1eJn-O7gkwykNxU8qJ9emaw?pwd=iuil).

To convert MKV to PNG, the suggested commands are as follows:

```bash
cd <your-data-dir>

# download ldv_v2/

cd ldv_v2/
chmod +x ./run.sh

# suppose the ffmpeg is located at ldv_v2/../ffmpeg-5.0.1-amd64-static/ffmpeg
# then you should run:
#
#./run.sh ../
#
./run.sh <your-ffmpeg-dir>
```

If you want to train DCAD or DnCNN, you should select some frames from each video to form the training and validation sets. Here are the suggested commands:

```bash
cd toolbox_data/
python generate_data_dir_for_dcad.py -label train -src-dir <your-data-dir>/ldv_v2/
python generate_data_dir_for_dcad.py -label valid -src-dir <your-data-dir>/ldv_v2/
```

### 4.2 MFQEv2 Dataset

We use the [MFQEv2 dataset](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset) for testing in addition to the LDVv2 test set.

The test set includes 16 videos. Each raw video is compressed by HM 16.20 with the LDP, QP=37 setting.

You can first download this dataset on [Dropbox](https://www.dropbox.com/sh/0wry4mned9djfhv/AACg--320WhAWQPVHUOVP7ala?dl=0) or [Baidu Pan](https://pan.baidu.com/s/1ASPHbeFmJf5HVMfa-bkiJw?pwd=cc2s).

To convert MKV to PNG, the suggested commands are as follows:

```bash
cd <your-data-dir>

# download mfqe_v2/

cd mfqe_v2/
chmod +x ./run.sh

# suppose the ffmpeg is located at mfqe_v2/../ffmpeg-5.0.1-amd64-static/ffmpeg
# then you should run:
#./run.sh ../
./run.sh <your-ffmpeg-dir>
```

Note: The MFQEv2 dataset originally has 18 test videos. Among them, two videos with 2K resolution, i.e., *PeopleOnStreet* and *Traffic*, are abandoned, since most approaches cannot test them with a GPU with 16 GB memory.

### 4.3 Create a Symbolic Link

```bash
cd mmediting/

# suppose your data dir is /mnt/usr/data
# then you should run:
#
#ln -s /mnt/usr/data ./
#
ln -s <your-data-dir> ./
```

Note that the `<your-data-dir>` should be an absolute path.

Layout:

```text
powervqe/
`-- mmediting/
    `-- data/
        |-- ldv_v2/
        |   |-- train_gt/
        |   |   |-- 001/
        |   |   |   |-- f001.png
        |   |   |   `-- ...
        |   |   `-- ...
        |   |-- train_lq/
        |   |-- valid_gt/
        |   |-- valid_lq/
        |   |-- test_gt/
        |   `-- test_lq/
        `-- mfqe_v2
            |-- test_gt/
            |   |-- BasketballDrill_832x480_500/
            |   |   |-- f001.png
            |   |   `-- ...
            |   `-- ...
            `-- test_lq/
```

## 5. Training

The suggested commands are the same as those in MMEditing.

Change the `data['train_dataloader']['samples_per_gpu']` in the config file according to your GPU number (It is strongly suggested to copy the config file and rename it according to your GPU number), then run:

```bash
cd mmediting/

chmod +x ./tools/dist_train.sh

# suppose your config file is located at ./configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py
# and the gpu number is 4
# then you should run:
#
#conda activate powervqe && \
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#PORT=29500 \
#./tools/dist_train.sh ./configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py 4
#
conda activate powervqe && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_train.sh <config-path> <gpu-number>
```

### 5.1 Special Case of the MFQEv2 models

To train the MFQEv2 models, you should first train the non-PQF model and then the PQF model:

```bash
cd mmediting/

# non-PQF
conda activate powervqe && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_train.sh ./configs/restorers/mfqev2/ldv_v2_non_pqf_4gpus.py 4

# PQF
conda activate powervqe && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_train.sh ./configs/restorers/mfqev2/ldv_v2_pqf_4gpus.py 4
```

## 6. Test

You can download the pre-trained models at the latest [Releases](https://github.com/ryanxingql/powervqe/releases/).

The suggested commands are the same as those in MMEditing.

Change the `data['test']['lq_folder']` and `data['test']['gt_folder']` in the config file, then run:

```bash
cd mmediting/

chmod +x ./tools/dist_test.sh

# suppose:
# your config file is located at:
# ./configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py
# your pre-trained model is located at:
# ./work_dirs/basicvsrpp_ldv_v2/iter_500000.pth
# you want to use 4 gpus
# you want to save images at ./data/enhanced/basicvsrpp_ldv_v2/500k/ldv
# then you should run:
#
#conda activate powervqe && \
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#PORT=29500 \
#./tools/dist_test.sh \
#./configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py \
#./work_dirs/basicvsrpp_ldv_v2/iter_500000.pth \
#4 \
#--save-path ./data/enhanced/basicvsrpp_ldv_v2/500k/ldv_v2
#
conda activate powervqe && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29510 \
./tools/dist_test.sh \
<config-path> \
<model-path> \
<gpu-number> \
--save-path <img-save-path>
```

### 6.1 Special Case of the MFQEv2 models

To test the MFQEv2 models, you should test the non-PQF and PQF models separately and save the enhanced frames to the same dir. Take testing over the LDVv2 dataset as an example:

```bash
# test non-PQFs over the LDVv2 dataset
conda activate powervqe && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_test.sh \
./configs/restorers/mfqev2/ldv_v2_non_pqf_4gpus.py \
./work_dirs/mfqev2_ldv_v2_non_pqf/iter_600000.pth \
4 \
--save-path ./data/enhanced/mfqev2_ldv_v2/600k/ldv_v2

# test PQFs over the LDVv2 dataset
conda activate powervqe && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_test.sh \
./configs/restorers/mfqev2/ldv_v2_pqf_4gpus.py \
./work_dirs/mfqev2_ldv_v2_pqf/iter_600000.pth \
4 \
--save-path ./data/enhanced/mfqev2_ldv_v2/600k/ldv_v2
```

### 6.2 Special Case of BasicVSR++ over the MFQEv2 Dataset

To test over the MFQEv2 dataset for BasicVSR++, a 32GB-memory GPU is needed. Besides, we use the following script:

```bash
cd mmediting/toolbox_test/

conda activate powervqe && \
python test.py -gpu 0 \
-inp-dir '../data/mfqe_v2/test_lq' \
-out-dir '../data/enhanced/basicvsrpp_ldv_v2/300k/mfqe_v2/' \
-config-path '../configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py' \
-model-path '../work_dirs/basicvsrpp_ldv_v2/iter_300000.pth'
```

### 6.3 Special Cases of DCAD and DnCNN

To test each video subfolder for DCAD or DnCNN, the demo pipeline is more recommended than the test pipeline. Take DCAD as an example:

```bash
cd mmediting/toolbox_test/

# test over the LDVv2 dataset
conda activate powervqe && \
python test.py -gpu 0 \
-inp-dir '../data/ldv_v2/test_lq' \
-out-dir '../data/enhanced/dcad_ldv_v2/500k/ldv_v2/' \
-config-path '../configs/restorers/dcad/ldv_v2_4gpus.py' \
-model-path '../work_dirs/dcad_ldv_v2/iter_500000.pth' \
-if-img

# test over the MFQEv2 dataset
conda activate powervqe && \
python test.py -gpu 0 \
-inp-dir '../data/mfqe_v2/test_lq' \
-out-dir '../data/enhanced/dcad_ldv_v2/500k/mfqe_v2/' \
-config-path '../configs/restorers/dcad/ldv_v2_4gpus.py' \
-model-path '../work_dirs/dcad_ldv_v2/iter_500000.pth' \
-if-img
```

### 6.4 PSNR Calculation

Finally, we can get the PSNR results. Take DCAD as an example:

```bash
cd toolbox_data/

# RGB-PSNR over the LDVv2 dataset

conda activate powervqe && \
python cal_rgb_psnr.py \
-gt-dir '../mmediting/data/ldv_v2/test_gt' \
-enh-dir '../mmediting/data/enhanced/dcad_ldv_v2/500k/ldv_v2' \
-ignored-frms '{"002":[0]}' \
-save-dir './log/dcad_ldv_v2/500k/ldv_v2'

# Y-PSNR over the MFQEv2 dataset

conda activate powervqe && \
python cal_y_psnr.py \
-gt-dir '../mmediting/data/mfqe_v2/test_gt' \
-enh-dir '../mmediting/data/enhanced/dcad_ldv_v2/500k/mfqe_v2' \
-save-dir './log/dcad_ldv_v2/500k/mfqe_v2' \
-order
```

Note: We ignore the PSNR of the first frame of video *002* in the LDVv2 dataset since it is a black frame and the PSNR is `inf`.

For the STDF-Y model:

```bash
conda activate powervqe && \
python cal_y_psnr_stdf_y.py \
-gt-dir '../mmediting/data/mfqe_v2/test_gt' \
-enh-dir '../mmediting/data/enhanced/stdf_y_ldv_v2/1m/mfqe_v2' \
-save-dir './log/stdf_y_ldv_v2/1m/mfqe_v2' \
-order
```

## 7. Q&A

### 7.1 Main Differences from the Original Papers

To improve the performance of DCAD,

1. The training patch size is changed from 38 to 128.
2. The LR is changed from 1 to 1e-4.
3. The optimizer is changed from AdaDelta to Adam.

To improve the performance of DnCNN,

1. The training patch size is changed from 40 to 128.
2. The LR is changed from 0.1 to 1e-4.
3. The optimizer is changed from SGD to Adam.
4. Different from [PowerQE](https://github.com/ryanxingql/powerqe), the batch normalization is turned on. It benefits the convergence of DnCNN.

To simplify the training of EDVR,

1. The input frames are 4x downsampled by strided convolutions. Downsampling can result in lower GPU consumption and faster training speed. Besides, we can use an SR model for quality enhancement this way.
2. The scheduler is changed from the multi-step CosineRestart to a single-step CosineRestart.

To simplify the training of MFQEv2,

1. Instead of conducting PQF detection, we assume that PQFs are located at the first, fifth, ninth, and... frames.
2. Instead of training a ME-MC subnet from scratch, we use a pre-trained SpyNet.

### 7.2 How to Use the Latest MMEditing

Here are some important files to run our codes. You can simply copy these files to the latest MMEditing repo.

- `mmediting/toolbox_test`
- `mmediting/demo/restoration_video_demo_basicvsrpp.py`
- `mmediting/configs/restorers/<your-interested-configs>.py`
- `mmediting/mmedit/apis/restoration_video_inference.py`
- `mmediting/mmedit/datasets/pipelines/augmentation.py`
- `mmediting/mmedit/datasets/<your-interested-datasets>.py`
- `mmediting/mmedit/models/backbones/sr_backbones/<your-interested-backbones>.py`
- `mmediting/mmedit/models/restorers/<your-interested-restorers>.py`

### 7.3 Support for Image Datasets

Prepare your image dataset. Take the DIV2K dataset as an example. Layout:

```text
powervqe/
`-- mmediting/
    `-- data/
        `-- div2k/
            |-- train_hq/
            |   |-- 0001.png
            |   |-- ...
            |   `-- 0800.png
            |-- train_lq/
            |   |-- qp27
            |   |   |-- 0001.png
            |   |   |-- ...
            |   |   `-- 0800.png
            |   |-- qp32
            |   |-- qp37
            |   |-- qp42
            |   |-- qf20
            |   |   |-- 0001.jpg
            |   |   |-- ...
            |   |   `-- 0800.jpg
            |   |-- qf30
            |   |-- qf40
            |   `-- qf50
            |-- valid_hq/
            |   |-- 0801.png
            |   |-- ...
            |   `-- 0900.png
            `-- valid_lq/
                |-- qp27
                |   |-- 0801.png
                |   |-- ...
                |   `-- 0900.png
                |-- qp32
                |-- qp37
                |-- qp42
                |-- qf20
                |   |-- 0801.jpg
                |   |-- ...
                |   `-- 0900.jpg
                |-- qf30
                |-- qf40
                `-- qf50
```

Example config files are presented in `mmediting/configs/` for some approaches as follows,

- [RBQE (ECCV 2020)](https://arxiv.org/abs/2006.16581): To improve the performance, it is trained in a non-blind manner.
- [CBDNet (CVPR 2019)](https://arxiv.org/abs/1807.04686): To improve the performance, it is trained in an end-to-end manner without the TV loss. Also, it is trained in a non-blind manner.
- [RDN (CVPR 2018)](https://arxiv.org/abs/1802.08797): To avoid memory out during the testing, the input frames are 4x downsampled by strided convolutions.
- [DCAD (DCC 2017)](https://ieeexplore.ieee.org/abstract/document/7923714)
- [DnCNN (TIP 2017)](https://arxiv.org/abs/1608.03981)
- [AR-CNN (ICCV 2015)](https://arxiv.org/abs/1504.06993): To improve the performance, the training patch size is changed from 32 to 128.

You can download the pre-trained models at the latest [Releases](https://github.com/ryanxingql/powervqe/releases/).

Note that for simplicity, we first train the `QP=37` and `QF=50` models, and then fine-tune them to get other models.

### 7.4 Support for LMDB

We can use LMDB to accerlate the IO. Specifically, we can store training patches and test images (optional) into LMDB files.

Pros:

- Fast IO speed.
- We can combine a large number of image patches into a few big LMDB files.
- Patches are prepared for training.
  - There is no need to randomly crop the patches during the training.
  - We can decide how to crop the patches (e.g., frames, patch size, cropping stride, etc.) in advance of the training.
- All images (PNG, JPG, etc.) can be stored as PNG.

Cons:

- We have to prepare the LMDB files with extra time, computation and storage.
- Once the LMDB file is generated, the training patches cannot be changed.
- The data pipeline should be modified for LMDB IO.

Take the DIV2K dataset as an example.

```bash
cd mmediting/

# train
conda activate powervqe && \
python tools/data/super-resolution/div2k/preprocess_div2k_dataset_powervqe.py \
--n-thread 16 \
--data-root ./data/div2k --data-type train \
--if-hq \
--if-lq --lqs qp27 qp32 qp37 qp42 qf20 qf30 qf40 qf50 \
--extract-patches --crop-size 128 --step 64 \
--make-lmdb

# valid
conda activate powervqe && \
python tools/data/super-resolution/div2k/preprocess_div2k_dataset_powervqe.py \
--n-thread 16 \
--data-root ./data/div2k --data-type valid \
--if-hq \
--if-lq --lqs qp27 qp32 qp37 qp42 qf20 qf30 qf40 qf50 \
--make-lmdb
```

After preparing the LMDB files, you should change the data path and pipeline in your config. Please refer to `mmediting/configs/restorers/cbdnet/` for examples.

### 7.5 Use Pre-commit Hook to Polish Code

1. Install pre-commit hook: `pip install -U pre-commit`
2. Config pre-commit hook based on `powervqe/.pre-commit-config.yaml`: `cd powervqe && pre-commit install`
3. Polish code before each commit and PR: `cd powervqe && pre-commit run --all`.

## 8. Licenses

We adopt Apache License 2.0. For other licenses, see MMEditing.

Enjoy this repo. Star it if you like it ^ ^
