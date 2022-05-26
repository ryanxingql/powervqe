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
    - [5.1 Special Cases](#51-special-cases)
  - [6. Test](#6-test)
    - [6.1 Special Cases](#61-special-cases)
    - [6.2 PSNR Calculation](#62-psnr-calculation)
  - [7. Q&A](#7-qa)
    - [7.1 Main Differences from the Original Papers](#71-main-differences-from-the-original-papers)
    - [7.2 How to Use the Latest MMEditing](#72-how-to-use-the-latest-mmediting)
    - [7.3 Supports for Image Datasets](#73-supports-for-image-datasets)
  - [8. Licenses](#8-licenses)

## 1. Introduction

We implement some widely-used quality enhancement approaches for compressed videos based on the powerful [MMEditing](https://github.com/open-mmlab/mmediting) project. These approaches are commonly used for comparison in this field are as follows:

- [STDF (AAAI 2020)](https://github.com/ryanxingql/stdf-pytorch): Enhancing compressed videos with feature-wise deformable convolutions, instead of frame-wise motion estimation and compensation.
- [MFQEv2 (TPAMI 2019)](https://github.com/ryanxingql/mfqev2.0): Enhancing frames in compressed videos taking advantage of neighboring good-quality frames.
- [DCAD (DCC 2017)](https://ieeexplore.ieee.org/abstract/document/7923714/): The first approach to post-process the decoded videos with a deep-learning-based method. It also sets up good experimental settings.
- [DnCNN (TIP 2017)](https://arxiv.org/abs/1608.03981): Widely used approach for decompression and denoising.

We also implement some SR baseline models for quality enhancement as follows:

- [BasicVSR++](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr_plusplus/README.md): Winner of the NTIRE 2021 VSR challenge.
- [EDVR](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/edvr/README.md): Winner of the NTIRE 2019 VSR challenge.

## 2. Performance

RGB-PSNR results on the test set of the [LDVv2 dataset](https://arxiv.org/abs/2204.09314) for the NTIRE 2022 video quality enhancement challenge are as follows:

| Index | Video name | Width | Height | Frames | Frame rate | LQ     | DCAD   | DnCNN  | STDF   | MFQEv2 | EDVR   | BasicVSR++ |
| ----- | ---------- | ----- | ------ | ------ | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ---------- |
| 1     | 001        | 960   | 536    | 250    | 25         | 32.433 | 32.872 | 32.901 | 32.915 | 33.039 | 33.163 | 33.342     |
| 2     | 002        | 960   | 536    | 300    | 30         | 30.182 | 30.587 | 30.624 | 30.690 | 30.824 | 30.978 | 31.686     |
| 3     | 003        | 960   | 536    | 300    | 30         | 29.047 | 29.670 | 29.700 | 29.659 | 29.981 | 29.890 | 30.488     |
| 4     | 004        | 960   | 536    | 250    | 25         | 34.313 | 35.105 | 35.209 | 35.258 | 35.448 | 35.774 | 36.196     |
| 5     | 005        | 960   | 536    | 600    | 60         | 30.492 | 30.919 | 30.953 | 31.023 | 31.108 | 31.335 | 31.693     |
| 6     | 006        | 960   | 536    | 300    | 30         | 28.994 | 29.450 | 29.503 | 29.490 | 29.588 | 29.785 | 30.117     |
| 7     | 007        | 960   | 536    | 240    | 24         | 28.845 | 29.359 | 29.373 | 29.445 | 29.581 | 29.696 | 29.865     |
| 8     | 008        | 960   | 504    | 240    | 24         | 31.798 | 32.640 | 32.706 | 32.790 | 32.835 | 33.119 | 33.389     |
| 9     | 009        | 960   | 536    | 250    | 25         | 29.857 | 30.628 | 30.656 | 30.718 | 30.842 | 30.999 | 31.140     |
| 10    | 010        | 960   | 504    | 600    | 60         | 34.646 | 35.531 | 35.598 | 35.578 | 35.591 | 35.666 | 35.634     |
| 11    | 011        | 960   | 536    | 600    | 60         | 26.300 | 26.662 | 26.735 | 26.689 | 26.666 | 26.804 | 26.926     |
| 12    | 012        | 960   | 536    | 600    | 60         | 22.867 | 23.334 | 23.324 | 23.455 | 23.437 | 23.319 | 23.391     |
| 13    | 013        | 960   | 536    | 300    | 30         | 28.424 | 28.787 | 28.793 | 28.863 | 29.222 | 29.190 | 29.805     |
| 14    | 014        | 960   | 536    | 300    | 30         | 31.214 | 31.820 | 31.853 | 31.875 | 32.022 | 32.209 | 32.451     |
| 15    | 015        | 960   | 536    | 300    | 30         | 29.032 | 29.495 | 29.527 | 29.475 | 29.601 | 29.681 | 30.461     |
|       | Ave.       |       |        |        |            | 29.896 | 30.457 | 30.497 | 30.528 | 30.652 | 30.774 | 31.106     |
|       | Delta PSNR |       |        |        |            |        | 0.561  | 0.601  | 0.632  | 0.756  | 0.878  | 1.209      |

Y-PSNR results on the test set (QP=37) of the [MFQEv2 dataset](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset) are as follows:

| Index | Video Name       | Width | Height | Frames | Frame rate | LQ     | DCAD   | DnCNN  | STDF   | MFQEv2 | EDVR   | BasicVSR++ |
| ----- | ---------------- | ----- | ------ | ------ | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ---------- |
| 1     | Kimono           | 1920  | 1080   | 240    | 24         | 33.076 | 33.345 | 33.370 | 33.419 | 33.837 | 33.986 | 34.560     |
| 2     | Park Scene       | 1920  | 1080   | 240    | 24         | 30.305 | 30.479 | 30.488 | 30.505 | 30.826 | 30.819 | 31.315     |
| 3     | Cactus           | 1920  | 1080   | 500    | 50         | 31.167 | 31.448 | 31.463 | 31.504 | 31.752 | 31.855 | 32.182     |
| 4     | BQ Terrace       | 1920  | 1080   | 600    | 60         | 29.962 | 30.330 | 30.332 | 30.341 | 30.443 | 30.488 | 30.730     |
| 5     | Basketball Drive | 1920  | 1080   | 500    | 50         | 32.066 | 32.408 | 32.436 | 32.475 | 32.689 | 32.827 | 33.247     |
| 6     | Race Horses      | 832   | 480    | 300    | 30         | 28.838 | 29.154 | 29.176 | 29.169 | 29.398 | 29.403 | 29.761     |
| 7     | BQ Mall          | 832   | 480    | 600    | 60         | 30.034 | 30.459 | 30.485 | 30.520 | 30.825 | 30.907 | 31.419     |
| 8     | Party Scene      | 832   | 480    | 500    | 50         | 26.604 | 26.937 | 26.950 | 27.050 | 27.159 | 27.176 | 27.446     |
| 9     | Basketball Drill | 832   | 480    | 500    | 50         | 30.334 | 30.829 | 30.861 | 30.911 | 31.028 | 31.171 | 31.367     |
| 10    | Race Horses      | 416   | 240    | 300    | 30         | 28.052 | 28.429 | 28.446 | 28.464 | 28.810 | 28.818 | 29.196     |
| 11    | BQ Square        | 416   | 240    | 600    | 60         | 27.040 | 27.623 | 27.633 | 27.804 | 27.916 | 27.873 | 28.174     |
| 12    | Blowing Bubbles  | 416   | 240    | 500    | 50         | 26.556 | 26.868 | 26.882 | 27.007 | 27.166 | 27.159 | 27.521     |
| 13    | Basketball Pass  | 416   | 240    | 500    | 50         | 29.232 | 29.642 | 29.668 | 29.736 | 30.048 | 30.176 | 30.562     |
| 14    | Four People      | 1280  | 720    | 600    | 60         | 33.349 | 33.912 | 33.946 | 33.962 | 34.068 | 34.254 | 34.466     |
| 15    | Johnny           | 1280  | 720    | 600    | 60         | 35.052 | 35.572 | 35.624 | 35.631 | 35.649 | 35.893 | 36.086     |
| 16    | Kristen And Sara | 1280  | 720    | 600    | 60         | 34.609 | 35.205 | 35.227 | 35.272 | 35.370 | 35.534 | 35.857     |
|       | Ave.             |       |        |        |            | 30.392 | 30.790 | 30.812 | 30.861 | 31.061 | 31.146 | 31.493     |
|       | Delta PSNR       |       |        |        |            |        | 0.398  | 0.420  | 0.468  | 0.669  | 0.754  | 1.101      |

Note: The performance of STDF (trained with RGB data) is slightly worse than that in this [repo](https://github.com/ryanxingql/stdf-pytorch) (trained with Y data).

## 3. Environment

Ubuntu; Four V100 GPUs with 16 GB memory.

```bash
git clone https://github.com/ryanxingql/powervqe.git --depth=1
cd powervqe/

conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y && conda activate open-mmlab
#pip install pip -U
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install openmim && mim install mmcv-full

#git clone https://github.com/open-mmlab/mmediting.git
cd mmediting/
pip3 install -e .

pip3 install scipy tqdm
pip3 install setuptools==59.5.0
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
#./run.sh ../
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
#ln -s /mnt/usr/data ./

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
#conda activate open-mmlab && \
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#PORT=29500 \
#./tools/dist_train.sh ./configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py 4
conda activate open-mmlab && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_train.sh <config-path> <gpu-number>
```

### 5.1 Special Cases

To train the MFQEv2 models, you should first train the non-PQF model and then the PQF model:

```bash
cd mmediting/

# non-PQF
conda activate open-mmlab && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_train.sh ./configs/restorers/mfqev2/ldv_v2_non_pqf_4gpus.py 4

# PQF
conda activate open-mmlab && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_train.sh ./configs/restorers/mfqev2/ldv_v2_pqf_4gpus.py 4
```

## 6. Test

You can download the pre-trained models at the latest Releases.

The suggested commands are the same as those in MMEditing.

Change the `data['test']['lq_folder']` and `data['test']['gt_folder']` in the config file, then run:

```bash
cd mmediting/

chmod +x ./tools/dist_test.sh

# suppose your config file is located at ./configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py
# your pre-trained model is located at ./work_dirs/basicvsrpp_ldv_v2/iter_500000.pth
# you want to use 4 gpus
# and you want to save images at ./work_dirs/basicvsrpp_ldv_v2/500k/ldv
# then you should run:
#conda activate open-mmlab && \
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#PORT=29500 \
#./tools/dist_test.sh \
#./configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py \
#./work_dirs/basicvsrpp_ldv_v2/iter_500000.pth \
#4 \
#--save-path ./work_dirs/basicvsrpp_ldv_v2/500k/ldv_v2
conda activate open-mmlab && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29510 \
./tools/dist_test.sh <config-path> <model-path> <gpu-number> \
--save-path <img-save-path>
```

### 6.1 Special Cases

To test the MFQEv2 dataset for BasicVSR++, the following commands are necessary, since the original BasicVSR++ cannot process frames with sizes smaller than 4*64:

```bash
cd mmediting/toolbox_test/

conda activate open-mmlab && \
python test.py -gpu 0 \
-inp-dir '../data/mfqe_v2/test_lq' \
-out-dir '../work_dirs/basicvsrpp_ldv_v2/300k/mfqe_v2/' \
-config-path '../configs/restorers/basicvsr_plusplus/ldv_v2_4gpus.py' \
-model-path '../work_dirs/basicvsrpp_ldv_v2/iter_300000.pth'
```

To test each video subfolder for DCAD or DnCNN, the demo pipeline is more recommended than the test pipeline:

```bash
cd mmediting/toolbox_test/

# test LDVv2 dataset
conda activate open-mmlab && \
python test.py -gpu 0 \
-inp-dir '../data/ldv_v2/test_lq' \
-out-dir '../work_dirs/dcad_ldv_v2/500k/ldv_v2/' \
-config-path '../configs/restorers/dcad/ldv_v2_4gpus.py' \
-model-path '../work_dirs/dcad_ldv_v2/iter_500000.pth' \
-if-img

# test MFQEv2 dataset
conda activate open-mmlab && \
python test.py -gpu 0 \
-inp-dir '../data/mfqe_v2/test_lq' \
-out-dir '../work_dirs/dcad_ldv_v2/500k/mfqe_v2/' \
-config-path '../configs/restorers/dcad/ldv_v2_4gpus.py' \
-model-path '../work_dirs/dcad_ldv_v2/iter_500000.pth' \
-if-img
```

To test the MFQEv2 models, you should test the non-PQF and PQF models separately and save the enhanced frames to the same dir.

```bash
# test non-PQFs
conda activate open-mmlab && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_test.sh \
./configs/restorers/mfqev2/ldv_v2_non_pqf_4gpus.py \
./work_dirs/mfqev2_ldv_v2_non_pqf/iter_600000.pth \
4 \
--save-path ./work_dirs/mfqev2_ldv_v2/ldv_v2

# test PQFs
conda activate open-mmlab && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 \
./tools/dist_test.sh \
./configs/restorers/mfqev2/ldv_v2_pqf_4gpus.py \
./work_dirs/mfqev2_ldv_v2_pqf/iter_600000.pth \
4 \
--save-path ./work_dirs/mfqev2_ldv_v2/ldv_v2
```

### 6.2 PSNR Calculation

Finally, we can get the PSNR results. Take DCAD as an example:

```bash
cd toolbox_data/

# RGB-PSNR for LDVv2

conda activate open-mmlab && \
python cal_rgb_psnr.py \
-gt-dir '../mmediting/data/ldv_v2/test_gt' \
-enh-dir '../mmediting/work_dirs/dcad_ldv_v2/500k/ldv_v2/' \
-ignored-frms '{"002":[0]}' \
-save-dir './log/dcad_ldv_v2/500k/ldv_v2/'

# Y-PSNR for MFQEv2

conda activate open-mmlab && \
python cal_y_psnr.py \
-gt-dir '../mmediting/data/mfqe_v2/test_gt' \
-enh-dir '../mmediting/work_dirs/dcad_ldv_v2/500k/mfqe_v2/' \
-save-dir './log/dcad_ldv_v2/500k/mfqe_v2/' \
-order
```

Note: We ignore the PSNR of the first frame of video *002* in the LDVv2 dataset since it is a black frame and the PSNR is `inf`.

## 7. Q&A

### 7.1 Main Differences from the Original Papers

DCAD:

1. Change the training patch size from 38 to 128.
2. Change the LR from 1 to 1e-4.
3. Change AdaDelta to Adam.

DnCNN:

1. Change the training patch size from 40 to 128.
2. Change the LR from 0.1 to 1e-4.
3. Change SGD to Adam.
4. Different from [PowerQE](https://github.com/ryanxingql/powerqe), we turn on batch normalization for DnCNN. It benefits the convergence of DnCNN.

EDVR:

1. Conducts 4x downsampling to the input frames (by strided convolutions) first. Downsampling can result in lower GPU consumption and faster training speed. Besides, we can use an SR model for quality enhancement this way.
2. Change the multi-step CosineRestart to a single-step CosineRestart.

MFQEv2:

1. Instead of conducting PQF detection, we assume that PQFs are located at the first, 5-th, 9-th, ... frames.
2. Instead of training a ME-MC subnet from scratch, we use a pre-trained SpyNet.

### 7.2 How to Use the Latest MMEditing

Here are some important files to run our codes. You can simply copy these files to the latest MMEditing repo.

- `mmediting/configs/restorers/{basicvsr_plusplus,dcad,dncnn,edvr,stdf}/ldv_v2_4gpus.py`
- `mmediting/configs/restorers/mfqev2/ldv_v2_{non_pqf,pqf}_4gpus.py`
- `mmediting/mmedit/models/backbones/sr_backbones/{basicvsr_pp_no_mirror,dcad,dncnn,edvr_net,mfqev2,stdf}.py`
- `mmediting/mmedit/models/restorers/{basicvsr,mfqev2_restorer,stdf}.py`
- `mmediting/mmedit/datasets/pipelines/augmentation.py`
- `mmediting/mmedit/datasets/{sr_ldv_dataset,ldp_dataset}.py`
- `mmediting/demo/restoration_video_demo.py`
- `mmediting/mmedit/apis/restoration_video_inference.py`
- `mmediting/toolbox_test`

### 7.3 Supports for Image Datasets

First, prepare your image dataset. Take the DIV2K dataset as an example. Layout:

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

Example config files are also presented in `mmediting/configs/` for some approaches, e.g., DCAD and DnCNN.

Note that for simplicity, we first train the `QP=37` and `QF=50` models, and then fine-tune them to get other models.

## 8. Licenses

We adopt Apache License 2.0. For other licenses, see MMEditing.

Enjoy this repo. Star it if you like it ^ ^
