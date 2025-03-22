"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    MegaDepth data handling was adapted from 
    LoFTR official code: https://github.com/zju3dv/LoFTR/blob/master/src/datasets/megadepth.py
"""

import io
import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv
from kornia.geometry.epipolar import essential_from_Rt
from kornia.geometry.epipolar import fundamental_from_essential

def get_essential(T0, T1):
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]
    
    t0 = T0[:3, 3].reshape(3, 1)
    t1 = T1[:3, 3].reshape(3, 1)
    
    R0 = torch.tensor(R0, dtype=torch.float32)
    R1 = torch.tensor(R1, dtype=torch.float32)
    t0 = torch.tensor(t0, dtype=torch.float32)
    t1 = torch.tensor(t1, dtype=torch.float32)
    
    E = essential_from_Rt(R0, t0, R1, t1)
    
    return E

def get_fundamental(E, K0, K1):
    F = fundamental_from_essential(E, K0, K1)
    
    return F
try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def load_array_from_s3(
    path, client, cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), 1)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def fix_path_from_d2net(path):
    if not path:
        return None

    path = path.replace('Undistorted_SfM/', '')
    path = path.replace('images', 'dense0/imgs')
    path = path.replace('phoenix/S6/zl548/MegaDepth_v1/', '')

    return path

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # resize image
    w, h = image.shape[1], image.shape[0]

    if len(resize) == 2:
        w_new, h_new = resize
    else:
        resize = resize[0]
        w_new, h_new = get_resized_wh(w, h, resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)


    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    #image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2,0,1) / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale

def imread_color(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_COLOR
    # if str(path).startswith('s3://'):
    #     image = load_array_from_s3(str(path), client, cv_type)
    # else:
    #     image = cv2.imread(str(path), cv_type)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if augment_fn is not None:
        image = augment_fn(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (3, h, w)


def read_megadepth_color(path,
                         resize=None,
                         df=None,
                         padding=False,
                         augment_fn=None,
                         rotation=0):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (3, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    image = imread_color(path, augment_fn, client=MEGADEPTH_CLIENT)
    
    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()

    # resize image
    if resize is not None:
        w, h = image.shape[1], image.shape[0]
        if len(resize) == 2:
            w_new, h_new = resize
        else:
            resize = resize[0]
            w_new, h_new = get_resized_wh(w, h, resize)
            w_new, h_new = get_divisible_wh(w_new, h_new, df)


        image = cv2.resize(image, (w_new, h_new))
        scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)
        scale_wh = torch.tensor([w_new, h_new], dtype=torch.float)
    else:
        scale = torch.tensor([1., 1.], dtype=torch.float)
        scale_wh = torch.tensor([image.shape[1], image.shape[0]], dtype=torch.float)
        
    image = image.transpose(2, 0, 1)
    
    if padding:  # padding
        if resize is not None:
            pad_to = max(h_new, w_new)
        else:
            pad_to = 2000
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale

def read_megadepth_depth(path, pad_to=None):

    if str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth

def get_image_name(path):
    return path.split('/')[-1]

def scale_intrinsics(K, scales):
    scales = np.diag([1. / scales[0], 1. / scales[1], 1.])
    return np.dot(scales, K)