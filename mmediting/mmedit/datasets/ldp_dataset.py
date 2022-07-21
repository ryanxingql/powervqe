# Copyright (c) ryanxingql. All rights reserved.
import os.path as osp
import random
from glob import glob

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class LDPPQFDataset(BaseSRDataset):
    """LDP PQF dataset for compressed video quality enhancement.

    The dataset loads three LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        i_frame_idx (int): Index of the I frame.
            Default: 0
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
        pipeline,
        scale,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        max_need_frms=100,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        assert filename_tmpl == 'f{:03d}'
        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.max_need_frms = max_need_frms

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, center_pqf, pqf_list):
        ord_center_pqf = pqf_list.index(center_pqf)
        if ord_center_pqf == 0:
            return pqf_list[0], pqf_list[1]
        elif ord_center_pqf == len(pqf_list) - 1:
            return pqf_list[-2], pqf_list[-1]
        else:
            return pqf_list[ord_center_pqf - 1], pqf_list[ord_center_pqf + 1]

    def load_annotations(self):
        """Load annotations.

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

            # for LDP, pqf is almost like this:
            # I frame (also PQF) NP NP NP PQF NP NP NP PQF NP ...
            # gap between two neighboring PQFs are 3
            pqf_list = list(
                range(self.i_frame_idx, self.i_frame_idx + max_frm_num, 4))

            for frm_path in frm_list:
                frm_name = frm_path.split('/')[-1].split('.')[0]

                # only collect PQFs
                frm_idx = int(frm_name[1:])
                if frm_idx not in pqf_list:
                    continue

                left_pqf_idx, right_pqf_idx = self.find_left_right_pqf(
                    frm_idx, pqf_list)
                clip_frm_name = f'{vid_name}/{frm_name}'
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        left_pqf_idx=left_pqf_idx,
                        right_pqf_idx=right_pqf_idx,
                        key=clip_frm_name,
                    ))

        return data_infos


@DATASETS.register_module()
class LDPNonPQFDataset(BaseSRDataset):
    """LDP non-PQF dataset for compressed video quality enhancement.

    The dataset loads three LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        i_frame_idx (int): Index of the I frame.
            Default: 0
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
        pipeline,
        scale,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        max_need_frms=100,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        assert filename_tmpl == 'f{:03d}'
        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.max_need_frms = max_need_frms

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, center_npqf, pqf_list):
        for ord_pqf_idx in range(len(pqf_list) - 1):
            if (pqf_list[ord_pqf_idx] <
                    center_npqf) and (pqf_list[ord_pqf_idx + 1] > center_npqf):
                return pqf_list[ord_pqf_idx], pqf_list[ord_pqf_idx + 1]

        # right pqf was not found; return itself as the right pqf
        return pqf_list[-1], center_npqf

    def load_annotations(self):
        """Load annotations.

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

            # for LDP, pqf is almost like this:
            # I frame (also PQF) NP NP NP PQF NP NP NP PQF NP ...
            # gap between two neighboring PQFs are 3
            pqf_list = list(
                range(self.i_frame_idx, self.i_frame_idx + max_frm_num, 4))

            for frm_path in frm_list:
                frm_name = frm_path.split('/')[-1].split('.')[0]

                # only collect non-PQFs
                frm_idx = int(frm_name[1:])
                if frm_idx in pqf_list:
                    continue

                left_pqf_idx, right_pqf_idx = self.find_left_right_pqf(
                    frm_idx, pqf_list)
                clip_frm_name = f'{vid_name}/{frm_name}'
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        left_pqf_idx=left_pqf_idx,
                        right_pqf_idx=right_pqf_idx,
                        key=clip_frm_name,
                    ))

        return data_infos
