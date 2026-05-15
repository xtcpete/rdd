from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
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
    "resnet50": [256, 512, 1024, 2048],
}
CONVNEXT_BUILDERS = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
    "convnext_xlarge": convnext_xlarge,
}
RESNET_BUILDERS = {
    "resnet50": resnet50,
}
RESNET_WEIGHTS = {
    "resnet50": ResNet50_Weights.IMAGENET1K_V1,
}
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _canonical_backbone_name(backbone_name: str) -> str:
    normalized = backbone_name.lower().replace("-", "_")
    if normalized == "resnet_50":
        return "resnet50"
    return normalized


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed batch statistics and affine parameters."""

    def __init__(self, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


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


class ResNetFeatureExtractor(nn.Module):
    """Expose intermediate ResNet feature maps."""

    def __init__(self, model: nn.Module, backbone_name: str, num_feature_levels: int) -> None:
        super().__init__()
        if num_feature_levels >= 5:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = MODEL_TO_NUM_CHANNELS[backbone_name]
        elif num_feature_levels == 4:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = MODEL_TO_NUM_CHANNELS[backbone_name][1:]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [MODEL_TO_NUM_CHANNELS[backbone_name][-1]]
        self.body = IntermediateLayerGetter(model, return_layers=return_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, NestedTensor]:
        xs = self.body(x)
        outputs: Dict[str, NestedTensor] = {}
        for name, feat in xs.items():
            mask_i = None
            if mask is not None:
                mask_i = (
                    F.interpolate(mask[None].float(), size=feat.shape[-2:], mode="nearest")
                    .to(torch.bool)[0]
                )
            outputs[name] = NestedTensor(feat, mask_i)
        return outputs
    
class Backbone(nn.Module):
    def __init__(self, config, pretrain: bool = False, **_: Any) -> None:
        super().__init__()
        backbone_name = _canonical_backbone_name(config["backbone"])
        use_22k = config.get("use_22k", True)
        if backbone_name in CONVNEXT_BUILDERS:
            builder = CONVNEXT_BUILDERS[backbone_name]
            model = builder(pretrained=pretrain, in_22k=use_22k)
            self.encoder = ConvNeXtFeatureExtractor(model, backbone_name)
        elif backbone_name in RESNET_BUILDERS:
            builder = RESNET_BUILDERS[backbone_name]
            weights = RESNET_WEIGHTS[backbone_name] if pretrain else None
            model = builder(
                weights=weights,
                replace_stride_with_dilation=[False, False, False],
                norm_layer=FrozenBatchNorm2d,
            )
            self.encoder = ResNetFeatureExtractor(
                model,
                backbone_name,
                config.get("num_feature_levels", 5),
            )
        else:
            raise ValueError(f"Unsupported backbone '{config['backbone']}'.")
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
