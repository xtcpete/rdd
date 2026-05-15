import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .deformable_transformer import build_deformable_transformer

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = False),
									)

	def forward(self, x):
	  return self.layer(x)

class RDD_Descriptor(nn.Module):
    def __init__(self, backbone, hidden_dim, num_feature_levels, transformer=None):
        super().__init__()
        self.transformer = transformer
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.use_deformable_transformer = transformer is not None

        matchibility_hidden_dim = max(self.hidden_dim // 2, 64)
        matchibility_low_dim = max(matchibility_hidden_dim // 2, 32)
        
        self.matchibility_head = nn.Sequential(
										BasicLayer(self.hidden_dim, matchibility_hidden_dim, 1, padding=0),
										BasicLayer(matchibility_hidden_dim, matchibility_low_dim, 1, padding=0),
										nn.Conv2d(matchibility_low_dim, 1, 1),
										nn.Sigmoid()
									)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.backbone = backbone
        self.stride = backbone.strides[0]
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
            
    def forward(self, samples: NestedTensor):
        
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        if self.use_deformable_transformer:
            flatten_feats, spatial_shapes, level_start_index = self.transformer(srcs, masks, pos)
            feats = []
            level_start_index = torch.cat((
                level_start_index,
                torch.tensor([flatten_feats.shape[1]], device=level_start_index.device),
            ))
            for i, shape in enumerate(spatial_shapes):
                assert len(shape) == 2
                temp = flatten_feats[:, level_start_index[i]: level_start_index[i + 1], :]
                feats.append(temp.transpose(1, 2).view(-1, self.hidden_dim, *shape))
        else:
            # ConvNeXt-only baseline: use projected backbone features directly.
            feats = srcs

        final_feature = feats[0]
        for feat in feats[1:]:
            final_feature = final_feature + F.interpolate(feat, size=final_feature.shape[-2:], mode='bilinear', align_corners=False)
        
        matchibility = self.matchibility_head(final_feature)
        
        return final_feature, matchibility
    
    
def build_descriptor(config):
    backbone = build_backbone(config)
    transformer = None
    if config.get('use_deformable_transformer', True):
        transformer = build_deformable_transformer(config)
    return RDD_Descriptor(
        backbone,
        hidden_dim=config['d_model'],
        num_feature_levels=config['num_feature_levels'],
        transformer=transformer,
    )
