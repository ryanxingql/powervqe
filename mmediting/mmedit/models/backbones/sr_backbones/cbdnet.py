# Copyright (c) ryanxingql. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from .unet import UNet

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class CBDNet(nn.Module):
    """CBDNet network structure.

    Args:
        in_channels (int): Channel number of inputs.
            Default: 3.
        mid_channels_1 (int): Channel number of the first intermediate features.
            Default: 64.
        mid_channels_2 (int): Channel number of the second intermediate features.
            Default: 32.
        mid_channels_3 (int): Channel number of the second intermediate features.
            Default: 16.
        out_channels (int): Channel number of outputs.
            Default: 3.
        in_kernel_size (int): Kernel size of the first convolution.
            Default: 9.
        mid_kernel_size (int): Kernel size of the first intermediate convolution.
            Default: 7.
        mid_kernel_size (int): Kernel size of the second intermediate convolution.
            Default: 1.
        out_kernel_size (int): Kernel size of the last convolution.
            Default: 5.
    """

    def __init__(self,
                 in_channels=3,
                 estimate_channels=32,
                 out_channels=3,
                 nlevel_denoise=3,
                 nf_base_denoise=64,
                 nf_gr_denoise=2,
                 nl_base_denoise=1,
                 nl_gr_denoise=2,
                 down_denoise='avepool2d',
                 up_denoise='transpose2d',
                 reduce_denoise='add'):
        super().__init__()

        self.in_channels = in_channels
        self.estimate_channels = estimate_channels
        self.out_channels = out_channels
        self.nlevel_denoise = nlevel_denoise
        self.nf_base_denoise = nf_base_denoise
        self.nf_gr_denoise = nf_gr_denoise
        self.nl_base_denoise = nl_base_denoise
        self.nl_gr_denoise = nl_gr_denoise
        self.down_denoise = down_denoise
        self.up_denoise = up_denoise
        self.reduce_denoise = reduce_denoise

        estimate_list = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=estimate_channels,
                kernel_size=3,
                padding=3 // 2),
            nn.ReLU(inplace=True)
        ] + [
            nn.Conv2d(
                in_channels=estimate_channels,
                out_channels=estimate_channels,
                kernel_size=3,
                padding=3 // 2),
            nn.ReLU(inplace=True),
        ] * 3 + [
            nn.Conv2d(estimate_channels, out_channels, 3, padding=3 // 2),
            nn.ReLU(inplace=True)
        ])
        self.estimate = nn.Sequential(*estimate_list)

        self.denoise = UNet(
            nf_in=in_channels * 2,
            nf_out=out_channels,
            nlevel=nlevel_denoise,
            nf_base=nf_base_denoise,
            nf_gr=nf_gr_denoise,
            nl_base=nl_base_denoise,
            nl_gr=nl_gr_denoise,
            down=down_denoise,
            up=up_denoise,
            reduce=reduce_denoise,
            residual=False,
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        estimated_noise_map = self.estimate(x)
        res = self.denoise(torch.cat([x, estimated_noise_map], dim=1))
        x = res + x

        return x

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
