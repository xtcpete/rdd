from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.dataset.megadepth import MegaDepthDataset
from src.dataset.utils import scale_intrinsics, warp_depth, warp_points2d


class AerialMegaDepthDataset(MegaDepthDataset):
    """Aerial MegaDepth training split using the repo's MegaDepth sample format."""

    def __init__(
        self,
        root,
        npz_path: Optional[str | Path] = None,
        split_path: Optional[str | Path] = None,
        num_per_scene: int = 100,
        num_per_split: Optional[int] = None,
        image_size: int = 800,
        min_overlap_score: float = 0.0,
        max_overlap_score: float = 1.0,
        gray: bool = False,
        crop_or_scale: str = "scale",
        train: bool = True,
        depth_percentile: Optional[float] = 98.0,
        **_: object,
    ) -> None:
        self.data_path = Path(root)
        self.num_per_scene = num_per_scene
        self.train = train
        self.image_size = image_size
        self.gray = gray
        self.crop_or_scale = crop_or_scale
        self.depth_percentile = depth_percentile

        split_file = Path(split_path or npz_path) if (split_path or npz_path) else None
        if split_file is None:
            raise ValueError("AerialMegaDepthDataset requires npz_path or split_path.")

        split_data = np.load(split_file, allow_pickle=True)
        self.scenes = split_data["scenes"]
        self.images = split_data["images"]
        self.images_scene_name = split_data.get("images_scene_name")
        self.pair_infos = self._filter_pairs(
            split_data["pairs"],
            min_overlap_score=min_overlap_score,
            max_overlap_score=max_overlap_score,
        )

        if num_per_split is not None:
            self.pair_infos = self._sample_pairs(self.pair_infos, num_per_split)
        else:
            self.pair_infos = self._sample_pairs_per_scene(self.pair_infos, num_per_scene)

        self.transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    @staticmethod
    def _filter_pairs(
        pairs: np.ndarray,
        min_overlap_score: float,
        max_overlap_score: float,
    ) -> np.ndarray:
        scores = np.asarray([AerialMegaDepthDataset._pair_overlap(pair) for pair in pairs], dtype=np.float32)
        keep_mask = (scores >= min_overlap_score) & (scores <= max_overlap_score)
        return np.asarray(pairs)[keep_mask]

    @staticmethod
    def _pair_overlap(pair) -> float:
        if isinstance(pair, np.void) and pair.dtype.names:
            for key in ("overlap_score", "overlap", "score"):
                if key in pair.dtype.names:
                    return float(pair[key])
        if isinstance(pair, dict):
            for key in ("overlap_score", "overlap", "score"):
                if key in pair:
                    return float(pair[key])
        return float(pair[3])

    @staticmethod
    def _pair_indices(pair) -> tuple[int, int, int]:
        if isinstance(pair, np.void) and pair.dtype.names:
            names = pair.dtype.names
            scene_key = AerialMegaDepthDataset._first_existing_key(names, ("scene_id", "scene"))
            image0_key = AerialMegaDepthDataset._first_existing_key(names, ("image0_id", "im1_id", "idx0", "image0"))
            image1_key = AerialMegaDepthDataset._first_existing_key(names, ("image1_id", "im2_id", "idx1", "image1"))
            return int(pair[scene_key]), int(pair[image0_key]), int(pair[image1_key])
        if isinstance(pair, dict):
            scene_key = AerialMegaDepthDataset._first_existing_key(pair, ("scene_id", "scene"))
            image0_key = AerialMegaDepthDataset._first_existing_key(pair, ("image0_id", "im1_id", "idx0", "image0"))
            image1_key = AerialMegaDepthDataset._first_existing_key(pair, ("image1_id", "im2_id", "idx1", "image1"))
            return int(pair[scene_key]), int(pair[image0_key]), int(pair[image1_key])
        return int(pair[0]), int(pair[1]), int(pair[2])

    @staticmethod
    def _first_existing_key(container, keys: tuple[str, ...]) -> str:
        for key in keys:
            if key in container:
                return key
        raise KeyError(f"Expected one of {keys}, got {tuple(container)}")

    @staticmethod
    def _sample_pairs(pairs: np.ndarray, limit: int) -> np.ndarray:
        pairs = np.asarray(pairs)
        if limit <= 0 or len(pairs) <= limit:
            return pairs
        indices = np.random.choice(len(pairs), limit, replace=False)
        return pairs[indices]

    def _pair_scene_name(self, pair) -> str:
        scene_id, image0_id, _ = self._pair_indices(pair)

        if self.images_scene_name is not None and 0 <= image0_id < len(self.images_scene_name):
            return str(self.images_scene_name[image0_id])

        if 0 <= scene_id < len(self.scenes):
            return str(self.scenes[scene_id])

        return f"{scene_id:04d}"

    def _sample_pairs_per_scene(self, pairs: np.ndarray, num_per_scene: int) -> np.ndarray:
        pairs = np.asarray(pairs)
        if num_per_scene <= 0 or len(pairs) == 0:
            return pairs

        scene_to_indices: dict[str, list[int]] = {}
        for idx, pair in enumerate(pairs):
            scene_name = self._pair_scene_name(pair)
            scene_to_indices.setdefault(scene_name, []).append(idx)

        selected_indices = []
        for scene_indices in scene_to_indices.values():
            if len(scene_indices) > num_per_scene:
                selected_indices.extend(np.random.choice(scene_indices, num_per_scene, replace=False).tolist())
            else:
                selected_indices.extend(scene_indices)

        return pairs[np.asarray(selected_indices, dtype=np.int64)]

    def _resolve_path(self, scene_root: Path, name: str, suffixes: tuple[str, ...]) -> Path:
        candidates = []
        raw_path = Path(name)

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(scene_root / raw_path)
            candidates.append(self.data_path / raw_path)

        expanded_candidates = list(candidates)
        for candidate in candidates:
            for suffix in suffixes:
                expanded_candidates.append(Path(f"{candidate}{suffix}"))
                if candidate.suffix:
                    expanded_candidates.append(candidate.with_suffix(suffix))

        for candidate in expanded_candidates:
            if candidate.exists():
                return candidate

        checked = ", ".join(str(path) for path in candidates[:2])
        raise FileNotFoundError(f"Could not resolve AerialMegaDepth file '{name}' under {checked}")

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        if depth_path.suffix == ".npy":
            depth = np.load(depth_path)
        elif depth_path.suffix == ".npz":
            depth_data = np.load(depth_path)
            if "depth" not in depth_data:
                raise KeyError(f"No depth entry found in {depth_path}")
            depth = depth_data["depth"]
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Could not read depth file: {depth_path}")

        if depth.ndim == 3:
            depth = depth[..., 0]

        depth = np.asarray(depth, dtype=np.float32)
        valid_depth = depth[depth > 0]
        if self.depth_percentile is not None and valid_depth.size > 0:
            max_depth = np.percentile(valid_depth, self.depth_percentile)
            depth = depth.copy()
            depth[depth > max_depth] = 0.0
        return depth

    @staticmethod
    def _load_camera(camera_path: Path) -> tuple[np.ndarray, np.ndarray]:
        camera = np.load(camera_path)
        if "intrinsics" in camera:
            intrinsics = camera["intrinsics"]
        elif "K" in camera:
            intrinsics = camera["K"]
        else:
            raise KeyError(f"No intrinsics/K entry found in {camera_path}")

        if "pose" in camera:
            pose = camera["pose"]
        elif "w2c" in camera:
            pose = camera["w2c"]
        elif "cam2world" in camera:
            pose = np.linalg.inv(camera["cam2world"])
        else:
            raise KeyError(f"No pose/w2c/cam2world entry found in {camera_path}")

        return np.asarray(intrinsics, dtype=np.float32).reshape(3, 3), np.asarray(pose, dtype=np.float32).reshape(4, 4)

    def _load_view(self, scene_name: str, image_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        scene_root = self.data_path / scene_name
        image_path = self._resolve_path(scene_root, image_name, (".jpg", ".jpeg", ".png"))
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)

        depth_path = self._resolve_path(scene_root, image_name, (".exr", ".npy", ".npz"))
        camera_path = self._resolve_path(scene_root, image_name, (".npz",))

        depth = self._load_depth(depth_path)
        if image.shape[0] != depth.shape[0] or image.shape[1] != depth.shape[1]:
            raise AssertionError(f"Image/depth shape mismatch for {image_path} and {depth_path}")

        intrinsics, pose = self._load_camera(camera_path)
        pair_name = str(image_path.relative_to(self.data_path)) if image_path.is_relative_to(self.data_path) else str(image_path)
        return image, depth, intrinsics, pose, pair_name

    def __len__(self) -> int:
        return len(self.pair_infos)

    def _scene_name_for_image(self, scene_id: int, image_id: int) -> str:
        if self.images_scene_name is not None:
            return str(self.images_scene_name[image_id])
        return str(self.scenes[scene_id])

    def recover_pair(self, idx):
        pair = self.pair_infos[idx % len(self)]
        scene_id, image0_id, image1_id = self._pair_indices(pair)

        scene_name0 = self._scene_name_for_image(scene_id, image0_id)
        scene_name1 = self._scene_name_for_image(scene_id, image1_id)
        image_name0 = str(self.images[image0_id])
        image_name1 = str(self.images[image1_id])

        image0, depth0, intrinsics0, pose0, pair_name0 = self._load_view(scene_name0, image_name0)
        image1, depth1, intrinsics1, pose1, pair_name1 = self._load_view(scene_name1, image_name1)

        pose01 = pose1 @ np.linalg.inv(pose0)
        pose10 = np.linalg.inv(pose01)

        if self.train:
            if "crop" in self.crop_or_scale:
                central_match = self._sample_central_match(depth0, depth1, intrinsics0, intrinsics1, pose01, pose10)

            if self.crop_or_scale == "crop":
                image0, depth0 = self._pad_to_size(image0, depth0, self.image_size)
                image1, depth1 = self._pad_to_size(image1, depth1, self.image_size)
                image0, bbox0, image1, bbox1 = self.crop(image0, image1, central_match)
                depth0 = depth0[bbox0[0]: bbox0[0] + self.image_size, bbox0[1]: bbox0[1] + self.image_size]
                depth1 = depth1[bbox1[0]: bbox1[0] + self.image_size, bbox1[1]: bbox1[1] + self.image_size]
            elif self.crop_or_scale == "scale":
                image0, depth0, intrinsics0 = self.scale(image0, depth0, intrinsics0)
                image1, depth1, intrinsics1 = self.scale(image1, depth1, intrinsics1)
                bbox0 = bbox1 = np.array([0.0, 0.0])
            elif self.crop_or_scale == "crop_scale":
                bbox0 = bbox1 = np.array([0.0, 0.0])
                image0, depth0, intrinsics0 = self.crop_scale(image0, depth0, intrinsics0, central_match[:2])
                image1, depth1, intrinsics1 = self.crop_scale(image1, depth1, intrinsics1, central_match[2:])
            else:
                raise RuntimeError(f"Unknown type {self.crop_or_scale}")
        else:
            image0, depth0, intrinsics0 = self.scale_keep_aspect(image0, depth0, intrinsics0)
            image1, depth1, intrinsics1 = self.scale_keep_aspect(image1, depth1, intrinsics1)
            bbox0 = bbox1 = np.array([0.0, 0.0])

        return (
            image0,
            depth0,
            intrinsics0,
            pose01,
            bbox0,
            image1,
            depth1,
            intrinsics1,
            pose10,
            bbox1,
            (pair_name0, pair_name1),
        )

    def _sample_central_match(self, depth0, depth1, intrinsics0, intrinsics1, pose01, pose10):
        downsample = 10
        depth0s = cv2.resize(depth0, (max(1, depth0.shape[1] // downsample), max(1, depth0.shape[0] // downsample)))
        depth1s = cv2.resize(depth1, (max(1, depth1.shape[1] // downsample), max(1, depth1.shape[0] // downsample)))
        intrinsic0s = scale_intrinsics(intrinsics0, (downsample, downsample))
        intrinsic1s = scale_intrinsics(intrinsics1, (downsample, downsample))

        depth10s = warp_depth(depth1s, intrinsic1s, intrinsic0s, pose10, depth0s.shape)
        depth10s[depth10s < 0] = 0
        valid10s = np.logical_and(depth10s > 0, depth0s > 0)

        pos0 = np.array(valid10s.nonzero())
        try:
            idx0_random = np.random.choice(np.arange(pos0.shape[1]), 1)
            uv0s = pos0[:, idx0_random][[1, 0]].reshape(1, 2)
            d0s = np.array(depth0s[uv0s[0, 1], uv0s[0, 0]]).reshape(1, 1)
            uv01s, _ = warp_points2d(uv0s, d0s, intrinsic0s, intrinsic1s, pose01)

            uv0 = uv0s[0] * downsample
            uv1 = uv01s[0] * downsample
        except ValueError:
            uv0 = [depth0.shape[1] / 2, depth0.shape[0] / 2]
            uv1 = [depth1.shape[1] / 2, depth1.shape[0] / 2]

        return [uv0[1], uv0[0], uv1[1], uv1[0]]

    @staticmethod
    def _pad_to_size(image: np.ndarray, depth: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = image.shape
        if h < size:
            image_padding = np.zeros((size - h, w, 3), dtype=image.dtype)
            depth_padding = np.zeros((size - h, w), dtype=depth.dtype)
            image = np.concatenate([image, image_padding], axis=0)
            depth = np.concatenate([depth, depth_padding], axis=0)
            h = size
        if w < size:
            image_padding = np.zeros((h, size - w, 3), dtype=image.dtype)
            depth_padding = np.zeros((h, size - w), dtype=depth.dtype)
            image = np.concatenate([image, image_padding], axis=1)
            depth = np.concatenate([depth, depth_padding], axis=1)
        return image.astype(np.uint8), depth.astype(np.float32)

    def __getitem__(self, idx):
        (
            image0,
            depth0,
            intrinsics0,
            pose01,
            bbox0,
            image1,
            depth1,
            intrinsics1,
            pose10,
            bbox1,
            pair_names,
        ) = self.recover_pair(idx)

        if self.gray:
            gray0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray0 = transforms.ToTensor()(gray0)
            gray1 = transforms.ToTensor()(gray1)

        image0, image1 = self.transforms(image0), self.transforms(image1)

        ret = {
            "image0": image0,
            "image1": image1,
            "angle": 0,
            "overlap": self._pair_overlap(self.pair_infos[idx % len(self)]),
            "dataset_name": "AerialMegaDepth",
            "pair_names": pair_names,
            "warp01_params": {
                "mode": "se3",
                "width": self.image_size if self.train else image0.shape[2],
                "height": self.image_size if self.train else image0.shape[1],
                "pose01": torch.from_numpy(pose01.astype(np.float32)),
                "bbox0": torch.from_numpy(bbox0.astype(np.float32)),
                "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
                "depth0": torch.from_numpy(depth0.astype(np.float32)),
                "depth1": torch.from_numpy(depth1.astype(np.float32)),
                "intrinsics0": torch.from_numpy(intrinsics0.astype(np.float32)),
                "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            },
            "warp10_params": {
                "mode": "se3",
                "width": self.image_size if self.train else image1.shape[2],
                "height": self.image_size if self.train else image1.shape[1],
                "pose01": torch.from_numpy(pose10.astype(np.float32)),
                "bbox0": torch.from_numpy(bbox1.astype(np.float32)),
                "bbox1": torch.from_numpy(bbox0.astype(np.float32)),
                "depth0": torch.from_numpy(depth1.astype(np.float32)),
                "depth1": torch.from_numpy(depth0.astype(np.float32)),
                "intrinsics0": torch.from_numpy(intrinsics1.astype(np.float32)),
                "intrinsics1": torch.from_numpy(intrinsics0.astype(np.float32)),
            },
        }

        if self.gray:
            ret["gray0"] = gray0
            ret["gray1"] = gray1
        return ret


AerialMegaDepthPaddedDataset = AerialMegaDepthDataset
