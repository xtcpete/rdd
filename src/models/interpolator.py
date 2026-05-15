"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bilinear', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        if self.align_corners:
            scale = torch.tensor([max(W - 1, 1), max(H - 1, 1)], device=x.device, dtype=x.dtype)
            return 2.0 * (x / scale) - 1.0

        scale = torch.tensor([W, H], device=x.device, dtype=x.dtype)
        return (2.0 * x + 1.0) / scale - 1.0

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = self.align_corners)
        return x.permute(0,2,3,1).squeeze(-2)
