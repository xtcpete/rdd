"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    MegaDepth data handling was adapted from 
    LoFTR official code: https://github.com/zju3dv/LoFTR/blob/master/src/datasets/megadepth.py
"""

import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
from RDD.dataset.megadepth.utils import read_megadepth_color, read_megadepth_depth, fix_path_from_d2net, get_image_name
import numpy.random as rnd
from RDD.dataset.megadepth.utils import get_essential, get_fundamental
import pdb, tqdm, os
from RDD.dataset.megadepth.megadepth_warper import spvs_coarse
import cv2
from RDD.dataset.megadepth.utils import scale_intrinsics

class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 num_per_scene=200,
                 min_overlap_score = 0.01, #0.3,
                 max_overlap_score = 0.8, #1,
                 load_depth = True,
                 img_resize = (800,608), #or None
                 df=32,
                 img_padding=True,
                 depth_padding=True,
                 crop=False,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            detection_dir (str): root folder. This contains 2D detections of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('/')[-1].split('.')[0].split('_')[0]
        self.load_depth = load_depth
        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            min_overlap_score = 0
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score]
        if len(self.pair_infos) > num_per_scene:
            indices = np.random.choice(len(self.pair_infos), num_per_scene, replace=False)
            self.pair_infos = [self.pair_infos[idx] for idx in indices]
        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None #and img_padding and depth_padding
        self.crop = crop
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.


    def __len__(self):
        return len(self.pair_infos)

    def crop_scale(self, image, depth, intrinsic, centeral):
        three, h_org, w_org = image.shape
        image_size = min(h_org, w_org)
        if h_org > w_org:
            if centeral[1] - image_size // 2 < 0:
                h_start = 0
            elif centeral[1] + image_size // 2 > h_org:
                h_start = h_org - image_size
            else:
                h_start = int(centeral[1]) - image_size // 2
            w_start = 0
        else:
            if centeral[0] - image_size // 2 < 0:
                w_start = 0
            elif centeral[0] + image_size // 2 > w_org:
                w_start = w_org - image_size
            else:
                w_start = int(centeral[0]) - image_size // 2
            h_start = 0

        croped_image = image[:, h_start: h_start + image_size, w_start: w_start + image_size]
        croped_depth = depth[h_start: h_start + image_size, w_start: w_start + image_size]
        intrinsic[0, 2] = intrinsic[0, 2] - w_start
        intrinsic[1, 2] = intrinsic[1, 2] - h_start
        
        image = F.interpolate(croped_image[None], self.img_resize)[0]
        depth = F.interpolate(croped_depth[None][None], self.img_resize)[0][0]
        intrinsic = scale_intrinsics(intrinsic, (image_size / self.img_resize[0], image_size / self.img_resize[1]))
        intrinsic = torch.from_numpy(intrinsic).float()
        return image, depth, intrinsic

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx % len(self)]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        try:
            # TODO: Support augmentation & handle seeds for each worker correctly.
            image0, mask0, scale0 = read_megadepth_color(
                img_name0, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
            image1, mask1, scale1 = read_megadepth_color(
                img_name1, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        except Exception as e:
            self.__getitem__(torch.randint(0, len(self), (1,)).item())
            
        del mask0, mask1
        
        if self.load_depth:
            
            depth0_path = '/'.join([self.scene_info['depth_paths'][idx0].replace('phoenix/S6/zl548/MegaDepth_v1', 'depth_undistorted').split('/')[i] for i in [0, 1, -1]])
            depth1_path = '/'.join([self.scene_info['depth_paths'][idx1].replace('phoenix/S6/zl548/MegaDepth_v1', 'depth_undistorted').split('/')[i] for i in [0, 1, -1]])
            
            # read depth. shape: (h, w)
            if self.mode in ['train', 'val']:
                depth0 = read_megadepth_depth(
                    osp.join(self.root_dir, depth0_path), pad_to=self.depth_max_size)
                depth1 = read_megadepth_depth(
                    osp.join(self.root_dir, depth1_path), pad_to=self.depth_max_size)
            else:
                depth0 = depth1 = torch.tensor([])

            # read intrinsics of original size
            K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
            K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()
            # E = get_essential(T0, T1)
            # Fm = get_fundamental(E, K_0, K_1)
            
            
            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'T0': T0,
                'T1': T1,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }
            if self.crop:
                # warp image and get the ROI of the warped image
                
                image0, mask0, scale0 = read_megadepth_color(
                    img_name0, None, self.df, False, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
                image1, mask1, scale1 = read_megadepth_color(
                    img_name1, None, self.df, False, None)
                    # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
                
                downsample = 8
                
                data['image0'] = image0
                data['image1'] = image1
                try:
                    corrs = spvs_coarse(data, scale = downsample)
                    corrs = corrs[0]
                    central_match = corrs[0] * downsample
                except:
                    central_match = [0, 0, 0, 0]
                
                image0, depth0, K_0 = self.crop_scale(image0, depth0, K_0, central_match[:2])
                image1, depth1, K_1 = self.crop_scale(image1, depth1, K_1, central_match[2:])
                
                data = {
                    'image0': image0,  # (1, h, w)
                    'depth0': depth0,  # (h, w)
                    'image1': image1,
                    'depth1': depth1,
                    'T_0to1': T_0to1,  # (4, 4)
                    'T_1to0': T_1to0,
                    'T0': T0,
                    'T1': T1,
                    'K0': K_0,  # (3, 3)
                    'K1': K_1,
                    'scale0': scale0,  # [scale_w, scale_h]
                    'scale1': scale1,
                    'dataset_name': 'MegaDepth',
                    'scene_id': self.scene_id,
                    'pair_id': idx,
                    'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
                }

        else:
            
            # read intrinsics of original size
            K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
            K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()

            data = {
                'image0': image0,  # (1, h, w)
                'image1': image1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }

        return data
    
    