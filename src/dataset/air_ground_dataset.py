import os
import copy
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from .utils import scale_intrinsics, warp_depth, warp_points2d

class AirGroundDataset(Dataset):
    def __init__(
            self,
            root,
            npz_path,
            num_per_scene=100,
            image_size=800,
            min_overlap_score=0.0,
            max_overlap_score=1.0,
            gray=False,
            crop_or_scale='scale',  # crop, scale, crop_scale
            train=True,
            **kwargs
    ):
        self.data_path = Path(root)
        self.num_per_scene = num_per_scene
        self.train = train
        self.image_size = image_size
        self.gray = gray
        self.crop_or_scale = crop_or_scale
        self.scene_id = npz_path.split('/')[-1].split('_')[0]
        self.pair_infos = dict(np.load(npz_path, allow_pickle=True))['indices']
        self.pair_infos = np.array([item for item in self.pair_infos if min_overlap_score <= item['overlap_score'] <= max_overlap_score])   
        if len(self.pair_infos) > num_per_scene:
            idxs = np.random.choice([i for i in range(len(self.pair_infos))], num_per_scene, replace=False)
            self.pair_infos = self.pair_infos[idxs]
        else:
            self.pair_infos = self.pair_infos
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.ToTensor()])
            
    def __len__(self):
        return len(self.pair_infos)

    def recover_pair(self, idx):
        # read grayscale image and mask. (1, h, w) and (h, w)
        pair_names = self.pair_infos[idx]['pair_names']
        poses = self.pair_infos[idx]['poses']
        Ks = self.pair_infos[idx]['Ks']
        depths = self.pair_infos[idx]['depths']
        
        
        img_name0 = pair_names[0]
        img_name1 = pair_names[1]

        depth_path0 = self.data_path / self.scene_id / 'depths' / depths[0].replace('.jpg', '')
        depth0 = np.load(depth_path0)
        
        image_path0 = self.data_path / self.scene_id / 'images' / img_name0
        image0 = Image.open(image_path0)
        if image0.mode != 'RGB':
            image0 = image0.convert('RGB')
        image0 = np.array(image0)
        assert image0.shape[0] == depth0.shape[0] and image0.shape[1] == depth0.shape[1]
        
        intrinsics0 = Ks[0].copy().reshape(3, 3)
        pose0 = poses[0]

        depth_path1 = self.data_path / self.scene_id / 'depths' / depths[1].replace('.jpg', '')
        depth1 = np.load(depth_path1)
        
        image_path1 = self.data_path / self.scene_id / 'images' / img_name1
        image1 = Image.open(image_path1)
        if image1.mode != 'RGB': 
            image1 = image1.convert('RGB')
        image1 = np.array(image1)
        assert image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1]
        
        intrinsics1 = Ks[1].copy().reshape(3, 3)
        pose1 = poses[1]

        pose01 = pose1 @ np.linalg.inv(pose0)
        pose10 = np.linalg.inv(pose01)
        
        central_match = None

        if self.train:
            if "crop" in self.crop_or_scale:
                # ================================================= compute central_match
                DOWNSAMPLE = 10
                # resize to speed up
                depth0s = cv2.resize(depth0, (depth0.shape[1] // DOWNSAMPLE, depth0.shape[0] // DOWNSAMPLE))
                depth1s = cv2.resize(depth1, (depth1.shape[1] // DOWNSAMPLE, depth1.shape[0] // DOWNSAMPLE))
                intrinsic0s = scale_intrinsics(intrinsics0, (DOWNSAMPLE, DOWNSAMPLE))
                intrinsic1s = scale_intrinsics(intrinsics1, (DOWNSAMPLE, DOWNSAMPLE))

                # warp
                depth01s = warp_depth(depth0s, intrinsic0s, intrinsic1s, pose01, depth1s.shape)
                depth10s = warp_depth(depth1s, intrinsic1s, intrinsic0s, pose10, depth0s.shape)

                depth01s[depth01s < 0] = 0
                depth10s[depth10s < 0] = 0

                valid01s = np.logical_and(depth01s > 0, depth1s > 0)
                valid10s = np.logical_and(depth10s > 0, depth0s > 0)

                pos0 = np.array(valid10s.nonzero())
                try:
                    idx0_random = np.random.choice(np.arange(pos0.shape[1]), 1)
                    uv0s = pos0[:, idx0_random][[1, 0]].reshape(1, 2)
                    d0s = np.array(depth0s[uv0s[0, 1], uv0s[0, 0]]).reshape(1, 1)

                    uv01s, _ = warp_points2d(uv0s, d0s, intrinsic0s, intrinsic1s, pose01)

                    uv0 = uv0s[0] * DOWNSAMPLE
                    uv1 = uv01s[0] * DOWNSAMPLE
                except ValueError:
                    uv0 = [depth0.shape[1] / 2, depth0.shape[0] / 2]
                    uv1 = [depth1.shape[1] / 2, depth1.shape[0] / 2]

                central_match = [uv0[1], uv0[0], uv1[1], uv1[0]]
                # ================================================= compute central_match

            if self.crop_or_scale == 'crop':
                # =============== padding
                h0, w0, _ = image0.shape
                h1, w1, _ = image1.shape
                if h0 < self.image_size:
                    padding = np.zeros((self.image_size - h0, w0, 3))
                    image0 = np.concatenate([image0, padding], axis=0).astype(np.uint8)
                    depth0 = np.concatenate([depth0, padding[:, :, 0]], axis=0).astype(np.float32)
                    h0, w0, _ = image0.shape
                if w0 < self.image_size:
                    padding = np.zeros((h0, self.image_size - w0, 3))
                    image0 = np.concatenate([image0, padding], axis=1).astype(np.uint8)
                    depth0 = np.concatenate([depth0, padding[:, :, 0]], axis=1).astype(np.float32)
                if h1 < self.image_size:
                    padding = np.zeros((self.image_size - h1, w1, 3))
                    image1 = np.concatenate([image1, padding], axis=0).astype(np.uint8)
                    depth1 = np.concatenate([depth1, padding[:, :, 0]], axis=0).astype(np.float32)
                    h1, w1, _ = image1.shape
                if w1 < self.image_size:
                    padding = np.zeros((h1, self.image_size - w1, 3))
                    image1 = np.concatenate([image1, padding], axis=1).astype(np.uint8)
                    depth1 = np.concatenate([depth1, padding[:, :, 0]], axis=1).astype(np.float32)
                # =============== padding
                image0, bbox0, image1, bbox1 = self.crop(image0, image1, central_match)

                depth0 = depth0[bbox0[0]: bbox0[0] + self.image_size, bbox0[1]: bbox0[1] + self.image_size]
                depth1 = depth1[bbox1[0]: bbox1[0] + self.image_size, bbox1[1]: bbox1[1] + self.image_size]

            elif self.crop_or_scale == 'scale':
                image0, depth0, intrinsics0 = self.scale(image0, depth0, intrinsics0)
                image1, depth1, intrinsics1 = self.scale(image1, depth1, intrinsics1)
                bbox0 = bbox1 = np.array([0., 0.])
            elif self.crop_or_scale == 'crop_scale':
                bbox0 = bbox1 = np.array([0., 0.])
                image0, depth0, intrinsics0 = self.crop_scale(image0, depth0, intrinsics0, central_match[:2])
                image1, depth1, intrinsics1 = self.crop_scale(image1, depth1, intrinsics1, central_match[2:])
            else:
                raise RuntimeError(f"Unkown type {self.crop_or_scale}")
        else:
            image0, depth0, intrinsics0 = self.scale_keep_aspect(image0, depth0, intrinsics0)
            image1, depth1, intrinsics1 = self.scale_keep_aspect(image1, depth1, intrinsics1)
            bbox0 = bbox1 = np.array([0., 0.])
        return (image0, depth0, intrinsics0, pose01, bbox0,
                image1, depth1, intrinsics1, pose10, bbox1)

    def scale(self, image, depth, intrinsic):
        img_size_org = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        depth = cv2.resize(depth, (self.image_size, self.image_size))
        intrinsic = scale_intrinsics(intrinsic, (img_size_org[1] / self.image_size, img_size_org[0] / self.image_size))
        return image, depth, intrinsic

    def scale_keep_aspect(self, image, depth, intrinsic):
        h_org, w_org, _ = image.shape
        target = self.image_size
        scale_factor = target / max(h_org, w_org)
        new_w = max(1, int(round(w_org * scale_factor)))
        new_h = max(1, int(round(h_org * scale_factor)))

        resized_image = cv2.resize(image, (new_w, new_h))
        resized_depth = cv2.resize(depth, (new_w, new_h))

        intrinsic = scale_intrinsics(intrinsic, (w_org / new_w, h_org / new_h))

        return resized_image, resized_depth, intrinsic

    def crop_scale(self, image, depth, intrinsic, centeral):
        h_org, w_org, three = image.shape
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

        croped_image = image[h_start: h_start + image_size, w_start: w_start + image_size]
        croped_depth = depth[h_start: h_start + image_size, w_start: w_start + image_size]
        intrinsic[0, 2] = intrinsic[0, 2] - w_start
        intrinsic[1, 2] = intrinsic[1, 2] - h_start

        image = cv2.resize(croped_image, (self.image_size, self.image_size))
        depth = cv2.resize(croped_depth, (self.image_size, self.image_size))
        intrinsic = scale_intrinsics(intrinsic, (image_size / self.image_size, image_size / self.image_size))

        return image, depth, intrinsic

    def crop(self, image0, image1, central_match):
        bbox0_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox0_i + self.image_size >= image0.shape[0]:
            bbox0_i = image0.shape[0] - self.image_size
        bbox0_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox0_j + self.image_size >= image0.shape[1]:
            bbox0_j = image0.shape[1] - self.image_size

        bbox1_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        return (image0[bbox0_i: bbox0_i + self.image_size, bbox0_j: bbox0_j + self.image_size],
                np.array([bbox0_i, bbox0_j]),
                image1[bbox1_i: bbox1_i + self.image_size, bbox1_j: bbox1_j + self.image_size],
                np.array([bbox1_i, bbox1_j])
                )

    def __getitem__(self, idx):
        (image0, depth0, intrinsics0, pose01, bbox0,
         image1, depth1, intrinsics1, pose10, bbox1) \
            = self.recover_pair(idx)

        if self.gray:
            gray0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray0 = transforms.ToTensor()(gray0)
            gray1 = transforms.ToTensor()(gray1)
        if self.transforms is not None:
            image0, image1 = self.transforms(image0), self.transforms(image1)  # [C,H,W]
        ret = {'image0': image0,
               'image1': image1,
               'angle': 0,
               'overlap': 0.5,
               'dataset_name': 'AirGround',
               'pair_names': self.pair_infos[idx]['pair_names'],
               'warp01_params': {'mode': 'se3',
                                 'width': self.image_size if self.train else image0.shape[2],
                                 'height': self.image_size if self.train else image0.shape[1],
                                 'pose01': torch.from_numpy(pose01.astype(np.float32)),
                                 'bbox0': torch.from_numpy(bbox0.astype(np.float32)),
                                 'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
                                 'depth0': torch.from_numpy(depth0.astype(np.float32)),
                                 'depth1': torch.from_numpy(depth1.astype(np.float32)),
                                 'intrinsics0': torch.from_numpy(intrinsics0.astype(np.float32)),
                                 'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32))},
               'warp10_params': {'mode': 'se3',
                                 'width': self.image_size if self.train else image1.shape[2],
                                 'height': self.image_size if self.train else image1.shape[2],
                                 'pose01': torch.from_numpy(pose10.astype(np.float32)),
                                 'bbox0': torch.from_numpy(bbox1.astype(np.float32)),
                                 'bbox1': torch.from_numpy(bbox0.astype(np.float32)),
                                 'depth0': torch.from_numpy(depth1.astype(np.float32)),
                                 'depth1': torch.from_numpy(depth0.astype(np.float32)),
                                 'intrinsics0': torch.from_numpy(intrinsics1.astype(np.float32)),
                                 'intrinsics1': torch.from_numpy(intrinsics0.astype(np.float32))},
               }
        if self.gray:
            ret['gray0'] = gray0
            ret['gray1'] = gray1
        return ret