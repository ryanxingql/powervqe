# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from glob import glob

import cv2
import mmcv
import numpy as np
import torch
from tqdm import tqdm

from mmedit.apis import (init_model, restoration_inference,
                         restoration_video_inference)
from mmedit.core import tensor2img
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template of the file names')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')

    parser.add_argument('--if-rmd', action='store_true')

    parser.add_argument(
        '--if-img', action='store_true', help='Test frame by frame')
    args = parser.parse_args()
    return args


def main():
    """ Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir'
    is set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    model = init_model(
        args.config,
        args.checkpoint,
        device=torch.device('cuda', args.device),
    )

    if args.if_img:
        img_list = sorted(glob(osp.join(args.input_dir, '*.png')))
        pbar = tqdm(img_list)
        for img in pbar:
            img_name = img.split('/')[-1]
            output = restoration_inference(model, img)
            output = tensor2img(output)

            save_path = osp.join(args.output_dir, img_name)
            mmcv.imwrite(output, save_path)
        pbar.close()
    else:
        output = restoration_video_inference(
            model,
            args.input_dir,
            args.window_size,
            args.start_idx,
            args.filename_tmpl,
            args.max_seq_len,
            if_rmd=args.if_rmd,
        )
        file_extension = os.path.splitext(args.output_dir)[1]
        if file_extension in VIDEO_EXTENSIONS:  # save as video
            h, w = output.shape[-2:]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.output_dir, fourcc, 25, (w, h))
            for i in range(0, output.size(1)):
                img = tensor2img(output[:, i, :, :, :])
                video_writer.write(img.astype(np.uint8))
            cv2.destroyAllWindows()
            video_writer.release()
        else:
            for i in range(args.start_idx, args.start_idx + output.size(1)):
                output_i = output[:, i - args.start_idx, :, :, :]
                output_i = tensor2img(output_i)
                save_path_i = (f'{args.output_dir}'
                               '/{args.filename_tmpl.format(i)}')

                mmcv.imwrite(output_i, save_path_i)


if __name__ == '__main__':
    main()
