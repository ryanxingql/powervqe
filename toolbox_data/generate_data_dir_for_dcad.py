import argparse
import os
import os.path as osp
import random
import shutil
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-src-dir', default='../mmediting/data/ldv_v2')
parser.add_argument('-label', default='train')
parser.add_argument('-nfrms', default=10, type=int)
args = parser.parse_args()

assert osp.exists(args.src_dir)

src_dir_gt = osp.join(args.src_dir, f'{args.label}_gt')
src_dir_lq = osp.join(args.src_dir, f'{args.label}_lq')

tar_dir_gt = osp.join(args.src_dir,
                      f'{args.label}_gt_{args.nfrms}_frms_max_per_vid')
tar_dir_lq = osp.join(args.src_dir,
                      f'{args.label}_lq_{args.nfrms}_frms_max_per_vid')
os.mkdir(tar_dir_gt)
os.mkdir(tar_dir_lq)

src_dir_gt_vid_list = sorted(glob(osp.join(src_dir_gt, '*/')))
count = 0
for src_dir_gt_vid in src_dir_gt_vid_list:
    src_dir_gt_imgs = sorted(glob(osp.join(src_dir_gt_vid, '*.png')))
    random.seed(7)
    random.shuffle(src_dir_gt_imgs)

    vid_name = src_dir_gt_vid.split('/')[-2]
    src_dir_lq_vid = osp.join(src_dir_lq, vid_name)

    for src_dir_gt_img in src_dir_gt_imgs[:args.nfrms]:
        img_name = src_dir_gt_img.split('/')[-1]
        src_dir_lq_img = osp.join(src_dir_lq_vid, img_name)

        new_img_name = f'{vid_name}_{img_name}'
        tar_dir_gt_img = osp.join(tar_dir_gt, new_img_name)
        tar_dir_lq_img = osp.join(tar_dir_lq, new_img_name)

        shutil.copy(src_dir_gt_img, tar_dir_gt_img)
        shutil.copy(src_dir_lq_img, tar_dir_lq_img)

        print(f'{src_dir_gt_img}\n-> {tar_dir_gt_img}')
        print(f'{src_dir_lq_img}\n-> {tar_dir_lq_img}')
        count += 1

print(f'> {len(src_dir_gt_vid_list)} videos were found.')
print(f'> {count} images were copied.')
