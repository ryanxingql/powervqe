"""
only use one gpu
"""
import subprocess
import argparse
import os.path as osp
import os
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-inp-dir', type=str, default='../data/ldv/test_lq')
parser.add_argument('-out-dir', type=str, default='../work_dirs/basicvsrpp_ldv_v2/latest')

parser.add_argument('-config-path', type=str, default='../configs/restorers/basicvsr_plusplus/ldv_v2.py')
parser.add_argument('-model-path', type=str, default='../work_dirs/basicvsrpp_ldv_v2/latest.pth')

parser.add_argument('-if-rmd', action='store_true')

parser.add_argument('-if-class', action='store_true')
parser.add_argument('-vid-class', type=str, default='A')

parser.add_argument('-if-indicate', action='store_true')
parser.add_argument('-vid-names', nargs='*',)

parser.add_argument('-if-img', action='store_true', help='Test image restoration methods')
args = parser.parse_args()

if not osp.exists(args.out_dir):
    os.makedirs(args.out_dir)

vid_list = glob(osp.join(args.inp_dir, '*/'))
vid_name_list = sorted([vid_path.split('/')[-2] for vid_path in vid_list])

for vid_name in vid_name_list:
    if args.if_class:
        width = int(vid_name.split('_')[1].split('x')[0])
        if args.vid_class == 'A' and width != 2560:
            continue
        if args.vid_class == 'B' and width != 1920:
            continue
        if args.vid_class == 'C' and width != 832:
            continue
        if args.vid_class == 'D' and width != 416:
            continue
        if args.vid_class == 'E' and width != 1280:
            continue
    elif args.if_indicate:
        if not (vid_name in args.vid_names):
            continue

    vid_subdir = osp.join(args.inp_dir, vid_name)
    save_subdir = osp.join(args.out_dir, vid_name)
    if not osp.exists(save_subdir):
        os.mkdir(save_subdir)

    print(f'> test: {vid_name}')
    cmd_ = f'CUDA_VISIBLE_DEVICES={args.gpu}'\
           f' python ../demo/restoration_video_demo.py {args.config_path} {args.model_path} {vid_subdir} {save_subdir}'\
           ' --start-idx 1 --filename-tmpl f{:03d}.png'
    if args.if_rmd:
        cmd_ += ' --if-rmd'
    if args.if_img:
        cmd_ += ' --if-img'
    p = subprocess.call(args=cmd_, shell=True)
