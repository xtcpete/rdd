import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
from RDD.dataset.megadepth.utils import read_megadepth_color, read_megadepth_depth, pad_bottom_right
import numpy.random as rnd
import pdb, tqdm, os

@torch.inference_mode()
def getDepth(filepath):
    depth = np.load(filepath)
    depth, _ = pad_bottom_right(depth, 2000, ret_mask=False)
    depth = torch.from_numpy(depth).float() # (h, w)
    return depth

class AirGroundDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 load_depth = True,
                 img_resize = 520, #or None
                 df=8,
                 img_padding=True,
                 depth_padding=True,
                 num_per_scene=100,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))['indices']
        self.img_resize = img_resize
        self.scene_id = npz_path.split('/')[-2]
        self.load_depth = load_depth
        self.mode = 'train'
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.
        
        if len(self.scene_info) > num_per_scene:
            idxs = np.random.choice([i for i in range(len(self.scene_info))], num_per_scene, replace=False)
            self.pair_infos = self.scene_info[idxs]
        else:
            self.pair_infos = self.scene_info

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):

        # read grayscale image and mask. (1, h, w) and (h, w)
        pair_names = self.pair_infos[idx]['pair_names']
        poses = self.pair_infos[idx]['poses']
        Ks = self.pair_infos[idx]['Ks']
        depths = self.pair_infos[idx]['depths']
        
        
        img_name0 = pair_names[0]
        img_name1 = pair_names[1]
        
        image0_path = osp.join(self.root_dir, 'images', img_name0)
        image1_path = osp.join(self.root_dir, 'images', img_name1)
        try:
            # TODO: Support augmentation & handle seeds for each worker correctly.
            image0, mask0, scale0 = read_megadepth_color(
                image0_path, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
            image1, mask1, scale1 = read_megadepth_color(
                image1_path, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        except Exception as e:
            return self.__getitem__(rnd.randint(len(self)))
        
            
        del mask0, mask1
        
        if self.load_depth:
            
            depth0_path = osp.join(self.root_dir, 'depths', depths[0])
            depth1_path = osp.join(self.root_dir, 'depths', depths[1])
            
            # read depth. shape: (h, w)
            if self.mode in ['train', 'val']:
                depth0 = getDepth(depth0_path)
                depth1 = getDepth(depth1_path)
            else:
                depth0 = depth1 = torch.tensor([])

            # read intrinsics of original size
            K0 = torch.tensor(Ks[0]).reshape(3, 3)
            K1 = torch.tensor(Ks[1]).reshape(3, 3)
            
            # read and compute relative poses
            T0 = poses[0]
            T1 = poses[1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()
            # E = get_essential(T0, T1)
            # Fm = get_fundamental(E, K_0, K_1)
            img_name0 = img_name0.split('/')[-1]
            img_name1 = img_name1.split('/')[-1]
            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'T0': T0,
                'T1': T1,
                'K0': K0,  # (3, 3)
                'K1': K1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': pair_names,
            }


        else:
            
            # read intrinsics of original size
            K0 = torch.tensor(Ks[0]).reshape(3, 3)
            K1 = torch.tensor(Ks[1]).reshape(3, 3)

            # read and compute relative poses
            T0 = poses[0]
            T1 = poses[1]
            
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()

            data = {
                'image0': image0,  # (1, h, w)
                'image1': image1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K0,  # (3, 3)
                'K1': K1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': pair_names,
            }

        return data