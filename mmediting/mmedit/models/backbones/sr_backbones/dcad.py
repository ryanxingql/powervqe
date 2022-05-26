# Copyright (c) ryanxingql. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class DCAD(nn.Module):
    """DCAD network structure.

    Paper: https://ieeexplore.ieee.org/document/7923714

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=8):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, padding=1)

        body_ = []
        for _ in range(num_blocks):
            body_ += [
                nn.ReLU(),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
            ]
        self.body = nn.Sequential(*body_)

        self.conv_after_body = nn.Sequential(
            *[nn.ReLU(),
              nn.Conv2d(mid_channels, out_channels, 3, padding=1)])

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        res = self.conv_after_body(self.body(self.conv_first(x)))
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
