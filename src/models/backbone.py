from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from .convnext import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    convnext_xlarge,
)
from .position_encoding import build_position_encoding
from ..utils.misc import NestedTensor
STRIDES = [4, 8, 16, 32]
MODEL_TO_NUM_CHANNELS: Dict[str, list[int]] = {
    "convnext_tiny": [96, 192, 384, 768],
    "convnext_small": [96, 192, 384, 768],
    "convnext_base": [128, 256, 512, 1024],
    "convnext_large": [192, 384, 768, 1536],
    "convnext_xlarge": [256, 512, 1024, 2048],
}
CONVNEXT_BUILDERS = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
    "convnext_xlarge": convnext_xlarge,
}
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class ConvNeXtFeatureExtractor(nn.Module):
    """Expose intermediate ConvNeXt feature maps."""
    def __init__(self, model: nn.Module, backbone_name: str) -> None:
        super().__init__()
        self.model = model
        self.backbone_name = backbone_name
        self.num_stages = len(self.model.stages)
        self.strides = STRIDES[: self.num_stages]
        self.num_channels = MODEL_TO_NUM_CHANNELS[backbone_name][: self.num_stages]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, NestedTensor]:
        outputs: Dict[str, NestedTensor] = {}
        feat = x
        for idx in range(self.num_stages):
            feat = self.model.downsample_layers[idx](feat)
            feat = self.model.stages[idx](feat)
            mask_i = None
            if mask is not None:
                mask_i = (
                    F.interpolate(mask[None].float(), size=feat.shape[-2:], mode="nearest")
                    .to(torch.bool)[0]
                )
            outputs[str(idx)] = NestedTensor(feat, mask_i)
        return outputs
    
class Backbone(nn.Module):
    def __init__(self, config, pretrain: bool = False, **_: Any) -> None:
        super().__init__()
        backbone_name = config["backbone"]
        use_22k = config.get("use_22k", True)
        if backbone_name not in CONVNEXT_BUILDERS:
            raise ValueError(f"Unsupported ConvNeXt backbone '{backbone_name}'.")
        builder = CONVNEXT_BUILDERS[backbone_name]
        model = builder(pretrained=pretrain, in_22k=use_22k)
        self.encoder = ConvNeXtFeatureExtractor(model, backbone_name)
        self.strides = self.encoder.strides
        self.num_channels = self.encoder.num_channels
        self.input_transform = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        x = tensor_list.tensors
        x = self.input_transform(x)
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x.shape[1]}")
        return self.encoder(x, tensor_list.mask)
    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for x in xs.values():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_backbone(config):
    position_embedding = build_position_encoding(config)
    pretrain = config['pretrain_backbone']
    backbone = Backbone(config, pretrain)
    model = Joiner(backbone, position_embedding)
    return model
