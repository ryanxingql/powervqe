# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import re
from functools import reduce

import mmcv
import numpy as np
import torch
import torch.nn.functional as nnF

from mmedit.datasets.pipelines import Compose

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def remove_duplicated_frames(data):
    nfrms = data.shape[1]

    # detect
    dup_list = []
    for idx_frm in range(1, nfrms):
        pre_frm = data[0, idx_frm-1, ...]
        curr_frm = data[0, idx_frm, ...]
        if np.equal(pre_frm, curr_frm).all():
            dup_list.append(idx_frm)

    # remove
    data = np.delete(data, dup_list, axis=1)
    return data, dup_list


def restore_duplicated_frames(data, dup_list):
    b, t, c, h, w = data.shape
    out_t = len(dup_list) + t
    data_out = torch.zeros(size=(b, out_t, c, h, w))
    accm_idx = 0
    for idx_t in range(out_t):
        if idx_t in dup_list:
            data_out[:, idx_t, ...] = data_out[:, idx_t-1, ...]
        else:
            data_out[:, idx_t, ...] = data[:, accm_idx, ...]
            accm_idx += 1
    return data_out


step_w = 428
step_h = 240
patch_w = step_w * 3
patch_h = step_h * 3


def cut_frames(data):
    b, t, c, h, w = data.shape
    assert b == 1
    if (w == 1920 and h == 1080 and t > 240) or (w > 1920) or (h > 1080):
        if_cut = True

        """
        # 1-D image
        |*|*|*|
        
        # pad on both sides
        |-|*|*|*|-|
        
        # patch
        |-|*|*|
          |*|*|*|
            |*|*|-|
        """

        n_w = int(np.ceil(w / step_w))
        n_h = int(np.ceil(h / step_h))

        w_after_pad = (n_w + 2) * step_w
        h_after_pad = (n_h + 2) * step_h

        w_gap = w_after_pad - w
        h_gap = h_after_pad - h

        pad_info = (step_w, w_gap-step_w, step_h, h_gap-step_h)
        print(f'> here we cut the {w}x{h} input frames into patches with the shape {patch_w}x{patch_h}!')
        data = nnF.pad(input=data.view(t, c, h, w), pad=pad_info, mode='reflect')  # t, c, h_after_pad, w_after_pad

        out = torch.zeros((n_w*n_h, t, c, patch_h, patch_w))  # each patch has no relation to others; thus as dim B
        for idx_w in range(n_w):
            for idx_h in range(n_h):
                idx_patch = idx_w * n_h + idx_h
                out[idx_patch, ...] = data[..., idx_h*step_h:(idx_h*step_h+patch_h), idx_w*step_w:(idx_w*step_w+patch_w)]
        data = out
        cut_info = (n_w, n_h, h, w)
    else:
        if_cut = False
        cut_info = None
    return data, if_cut, cut_info


def merge_patches(data, if_cut, cut_info):
    if if_cut:
        n_w, n_h, h, w = cut_info
        _, t, c, patch_h, patch_w = data.shape
        out = torch.zeros((1, t, c, n_h*step_h, n_w*step_w))
        for idx_w in range(n_w):
            for idx_h in range(n_h):
                idx_patch = idx_w * n_h + idx_h
                center_patch = data[idx_patch, :, :, step_h:step_h*2, step_w:step_w*2]  # center 1/3x1/3
                out[..., idx_h*step_h:(idx_h+1)*step_h, idx_w*step_w:(idx_w+1)*step_w] = center_patch
        data = out[..., :h, :w]
    return data


def pad_frames(data):
    b, t, c, h, w = data.shape

    h_gap = 64 * 4 - h
    h_gap = h_gap if h_gap > 0 else 0
    w_gap = 64 * 4 - w
    w_gap = w_gap if w_gap > 0 else 0
    if h_gap != 0 or w_gap != 0:
        print('> here we pad the input frames!')

    pad_info = (w_gap//2, w_gap-w_gap//2, h_gap//2, h_gap-h_gap//2)
    data = nnF.pad(input=data.view(-1, c, h, w), pad=pad_info, mode='reflect')
    data = data.view(b, t, c, h+h_gap, w+w_gap)
    return data, pad_info


def restore_resolution(data, pad_info):
    w_l, w_r, h_l, h_r = pad_info
    if w_r == 0 and (h_r != 0):
        data = data[..., h_l:-h_r, w_l:]  # w_l:-0 will cause empty data!
    elif w_r != 0 and (h_r == 0):
        data = data[..., h_l:, w_l:w_r]
    elif w_r == 0 and (h_r == 0):
        data = data[..., h_l:, w_l:]
    else:
        data = data[..., h_l:-h_r, w_l:-w_r]
    return data


def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None,
                                if_rmd=False,):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.

    Returns:
        Tensor: The predicted restoration result.
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # check if the input is a video
    file_extension = osp.splitext(img_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(img_dir)
        # load the images
        data = dict(lq=[], lq_path=None, key=img_dir)
        for frame in video_reader:
            data['lq'].append(np.flip(frame, axis=2))

        # remove the data loading pipeline
        tmp_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] not in [
                    'GenerateSegmentIndices', 'LoadImageFromFileList'
            ]:
                tmp_pipeline.append(pipeline)
        test_pipeline = tmp_pipeline
    else:
        # the first element in the pipeline must be 'GenerateSegmentIndices'
        if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
            raise TypeError('The first element in the pipeline must be '
                            f'"GenerateSegmentIndices", but got '
                            f'"{test_pipeline[0]["type"]}".')

        # specify start_idx and filename_tmpl
        test_pipeline[0]['start_idx'] = start_idx
        test_pipeline[0]['filename_tmpl'] = filename_tmpl

        # prepare data
        sequence_length = len(glob.glob(osp.join(img_dir, '*')))
        img_dir_split = re.split(r'[\\/]', img_dir)
        key = img_dir_split[-1]
        lq_folder = reduce(osp.join, img_dir_split[:-1])
        data = dict(
            lq_path=lq_folder,
            gt_path='',
            key=key,
            sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu  # btchw

    # remove duplicated frames if indicated
    if if_rmd:
        data, dup_list = remove_duplicated_frames(data)
        print(f'> {len(dup_list)} duplicated frames are found for video {key}!')

    # cut frames into patches
    data, if_cut, cut_info = cut_frames(data)

    # pad frames to make the size bigger than 256
    data, pad_info = pad_frames(data)

    # forward the model
    with torch.no_grad():
        result_batch_list = []
        nb = data.shape[0]
        data_ = data
        for ib in range(nb):
            if if_cut:
                print(f'patch: {ib+1} / {nb}')
                data = data_[ib:ib+1, ...]  # not [ib, ...]; to preserve the 0-th dim

            if window_size > 0:  # sliding window framework
                data = pad_sequence(data, window_size)
                result = []
                for i in range(0, data.size(1) - 2 * (window_size // 2)):
                    data_i = data[:, i:i + window_size].to(device)
                    result.append(model(lq=data_i, test_mode=True)['output'].cpu())
                result = torch.stack(result, dim=1)
            else:  # recurrent framework
                if max_seq_len is None:
                    result = model(
                        lq=data.to(device), test_mode=True)['output'].cpu()
                else:
                    result = []
                    for i in range(0, data.size(1), max_seq_len):
                        result.append(
                            model(
                                lq=data[:, i:i + max_seq_len].to(device),
                                test_mode=True)['output'].cpu())
                    result = torch.cat(result, dim=1)

            result_batch_list.append(result)
            torch.cuda.empty_cache()
        result = torch.cat(result_batch_list, dim=0)

    # restore resolution if pad above
    result = restore_resolution(result, pad_info)

    # merge patches if patch above
    result = merge_patches(result, if_cut, cut_info)

    # restore duplicated frames if remove above
    if if_rmd:
        result = restore_duplicated_frames(result, dup_list)

    return result
