import torch.nn as nn
from einops.einops import rearrange
import torch
import torch.nn.functional as F
from torchvision.models import resnet
from .detector import ResBlock

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([max_out, avg_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn
    
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = SEBlock(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        output = self.channel(x)
        output = self.spatial(output)
        return output + x  # Residual connection

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.conv1 = ResBlock(features, features)
        self.conv2 = ResBlock(features, features)

        self.norm1 = nn.BatchNorm2d(features)
        self.norm2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.attn = CBAM(features)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.attn(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)
    
class Upsampler(nn.Module):
    def __init__(self, hr_dim, lr_dim, depth, hidden_dim): 
        super().__init__()    
        self.input_proj_conv1 = resnet.conv1x1(hr_dim, hidden_dim)
        self.input_proj_conv2 =  resnet.conv1x1(lr_dim, hidden_dim)
        self.ResidualConvUnit = nn.ModuleList([ResidualConvUnit(hidden_dim, nn.ReLU(inplace=True)) for _ in range(depth)])
        self.final_conv = resnet.conv1x1(hidden_dim, hidden_dim)

    def forward(self, hr_feats, lr_feats):
        b, c, h, w = hr_feats.shape
        
        # Project query to the desired dimension
        hr_feats = self.input_proj_conv1(hr_feats)
        lr_feats = self.input_proj_conv2(lr_feats)

        # Interpolate low-resolution features to high-resolution size
        lr_feats = F.interpolate(lr_feats, size=(h, w), mode='bilinear', align_corners=False)

        # Residual connection
        for residual_conv in self.ResidualConvUnit:
            hr_feats = residual_conv(hr_feats + lr_feats)

        hr_feats = self.final_conv(hr_feats)

        return hr_feats # b, hidden_dim, h, w

def build_upsamplers(config):
    hr_dims = config['hr_dim']
    lr_dims = config['lr_dim']
    depth = config['depth']
    hidden_dims = config['upsamplers_hidden_dim']

    upsamplers = []
    for i in range(len(hr_dims)):
        upsamplers.append(
            Upsampler(
                hr_dim=hr_dims[i],
                lr_dim=lr_dims[i],
                depth=depth,
                hidden_dim=hidden_dims[i]
            )
        )
    upsamplers = nn.ModuleList(upsamplers)
    return upsamplers