# Copyright (c) ryanxingql. All rights reserved.
import os
import os.path as osp
from glob import glob

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS

import random


@DATASETS.register_module()
class SRLDVDataset(BaseSRDataset):
    """LDV dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    It reads REDS keys from the txt file.
    Each line contains:
    1. image name; 2, image shape, separated by a white space.
    Examples:

    ::

        000/00000000.png (720, 1280, 3)
        000/00000001.png (720, 1280, 3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.

    @ryanxingql
    """

    def __init__(
        self,
        lq_folder,
        gt_folder,
        num_input_frames,
        pipeline,
        scale,
        max_need_frms=100,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames

        self.test_mode = test_mode
        self.max_need_frms = max_need_frms

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for REDS dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        data_infos = []

        vid_list = sorted(glob(osp.join(self.lq_folder, '*/')))
        for vid_dir in vid_list:
            frm_list = sorted(glob(osp.join(vid_dir, '*.png')))
            vid_name = vid_dir.split('/')[-2]
            max_frm_num = len(frm_list)

            if self.test_mode and (max_frm_num > self.max_need_frms):
                random.shuffle(frm_list)
                frm_list = frm_list[:self.max_need_frms]

            for frm_path in frm_list:
                frm_name = frm_path.split('/')[-1].split('.')[0]
                clip_frm_name = f'{vid_name}/{frm_name}'
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        key=clip_frm_name,
                        max_frame_num=max_frm_num,
                        num_input_frames=self.num_input_frames))

        return data_infos
