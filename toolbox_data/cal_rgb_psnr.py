import argparse
import json
import os
import os.path as osp
from glob import glob
from math import log10

import numpy as np
import pandas as pd
from cv2 import cv2
from tqdm import tqdm

max_pixel_square = 255 * 255


def cal_psnr(original, compressed):
    mse = np.mean((original - compressed)**2)
    if mse == 0:
        return np.inf
    psnr = 10 * log10(max_pixel_square / mse)
    return psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt-dir', default='../mmediting/data/ldv_v2/test_gt')
    parser.add_argument('-enh-dir', default='../mmediting/data/ldv_v2/test_lq')
    parser.add_argument('-ignored-frms',
                        type=json.loads,
                        default='{"002":[0]}',
                        help='{"002":[0,]} will leads to error!')
    parser.add_argument('-save-dir', default='log')
    args = parser.parse_args()

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    vid_list = glob(osp.join(args.gt_dir, '*/'))
    vid_name_list = sorted([vid_path.split('/')[-2] for vid_path in vid_list])

    for vid_name in vid_name_list:
        gt_vid_dir = osp.join(args.gt_dir, vid_name)
        enh_vid_dir = osp.join(args.enh_dir, vid_name)

        img_list = glob(osp.join(gt_vid_dir, '*.png'))
        img_name_list = sorted(
            [img_path.split('/')[-1] for img_path in img_list])

        psnr_list = []
        bar_ = tqdm(total=len(img_name_list))
        for img_name in img_name_list:
            img_gt = cv2.imread(osp.join(gt_vid_dir, img_name)).astype(float)
            img_enh = cv2.imread(osp.join(enh_vid_dir, img_name)).astype(float)

            psnr = cal_psnr(img_enh, img_gt)
            psnr_list.append(psnr)

            bar_.update(1)
        bar_.close()

        df = pd.DataFrame(dict(psnr=psnr_list, ))
        csv_path = osp.join(args.save_dir, f'{vid_name}.csv')
        df.to_csv(csv_path)
        print(f'saved to: {csv_path}')

    ave_psnr_list = []
    inf_num_list = []
    ignore_num_list = []
    for vid_name in vid_name_list:
        df = pd.read_csv(osp.join(args.save_dir, f'{vid_name}.csv'))

        valid_psnr_list = []
        inf_num = 0
        ignore_num = 0
        for idx_psnr, psnr in enumerate(df['psnr']):
            if psnr == np.inf:
                inf_num += 1

            if (vid_name in args.ignored_frms) and (
                    idx_psnr in args.ignored_frms[vid_name]):
                ignore_num += 1
            else:
                valid_psnr_list.append(psnr)

        inf_num_list.append(inf_num)
        ignore_num_list.append(ignore_num)
        ave_psnr_list.append(np.mean(valid_psnr_list))

    df = pd.DataFrame(
        dict(
            vid_name=vid_name_list,
            psnr=ave_psnr_list,
            inf_num=inf_num_list,
            ignore_num=ignore_num_list,
        ))
    csv_path = osp.join(args.save_dir, 'ave.csv')
    df.to_csv(csv_path)
    print(f'saved to: {csv_path}')


if __name__ == '__main__':
    main()
