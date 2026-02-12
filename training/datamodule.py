from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset.megadepth import MegaDepthDataset
from src.dataset.air_ground_dataset import AirGroundDataset

def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = (base_seed + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CombinedDataModule(pl.LightningDataModule):
    """Lightning DataModule with megadepth and airground training and validation splits."""

    def __init__(
        self,
        megadepth_root_path: Path,
        val_indices_root: Path,
        batch_size: int,
        num_workers: int,
        training_res: int,
        train_detector: bool,
        seed: int,
        val_res: int = 1600,
        air_ground_root: Optional[Path] = None,
        air_ground_npz_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.megadepth_root = Path(megadepth_root_path)
        self.val_indices_root = Path(val_indices_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_res = training_res
        self.train_detector = train_detector
        self.seed = seed
        self._needs_resample = True
        self.val_res = val_res
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None
        self.air_ground_root = Path(air_ground_root) if air_ground_root else None
        self.air_ground_npz_root = Path(air_ground_npz_root) if air_ground_npz_root else None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        if stage in (None, "fit"):
            self._ensure_train_dataset()
        if self.val_dataset is None:
            self.val_dataset = self._build_val_dataset()

    def _set_dataset_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _ensure_train_dataset(self) -> None:
        if self.train_dataset is None or self._needs_resample:
            self._set_dataset_seed()
            self.train_dataset = self._build_train_dataset()
            self._needs_resample = False

    def increment_seed_and_resample(self) -> None:
        self.seed += 1
        self.train_dataset = None
        self._needs_resample = True

    # Training / validation datasets -------------------------------------------------
    def _build_train_dataset(self) -> ConcatDataset:
        indices_root = self.megadepth_root / 'megadepth_indices' / 'scene_info_0.1_0.7'
        npz_paths = sorted(glob.glob(str(indices_root / '*.npz')))
        if not npz_paths:
            raise FileNotFoundError(f"No MegaDepth indices found under {indices_root}")

        if self.train_detector:
            min_overlap, max_overlap = 0.1, 0.8
            num_per_scene = 100
        else:
            min_overlap, max_overlap = 0.01, 0.7
            num_per_scene = 200

        def build_split(mode: str) -> ConcatDataset:
            datasets = [
                MegaDepthDataset(
                    root=self.megadepth_root,
                    npz_path=path,
                    min_overlap_score=min_overlap,
                    max_overlap_score=max_overlap,
                    image_size=self.training_res,
                    num_per_scene=num_per_scene,
                    gray=False,
                    crop_or_scale=mode,
                )
                for path in npz_paths
            ]
            return ConcatDataset(datasets)

        datasets = [build_split('crop'), build_split('scale')]

        if self.air_ground_root and self.air_ground_npz_root:
            air_ground_npz_paths = sorted(glob.glob(str(self.air_ground_npz_root / '*.npz')))

            def build_ag_split(mode: str) -> ConcatDataset:
                ag_datasets = [
                    AirGroundDataset(
                        root=self.air_ground_root,
                        npz_path=path,
                        num_per_scene= int(num_per_scene * 3),  # AirGround has fewer scenes, so we can sample more per scene
                        image_size=self.training_res,
                        min_overlap_score=min_overlap,
                        max_overlap_score=max_overlap,
                        gray=False,
                        crop_or_scale=mode,
                        train=True
                    )
                    for path in air_ground_npz_paths
                ]
                return ConcatDataset(ag_datasets)

            if air_ground_npz_paths:
                datasets.append(build_ag_split('crop'))
                datasets.append(build_ag_split('scale'))

        return ConcatDataset(datasets)

    def _build_val_dataset(self) -> Optional[ConcatDataset]:
        val_root = self.val_indices_root
        if not val_root.is_absolute():
            val_root = self.megadepth_root / val_root

        npz_paths = sorted(glob.glob(str(val_root / '*.npz')))
        if not npz_paths:
            raise FileNotFoundError(f"No MegaDepth validation indices found under {val_root}")

        datasets = [
            MegaDepthDataset(
                root=self.megadepth_root,
                npz_path=path,
                min_overlap_score=0.0,
                max_overlap_score=1.0,
                image_size=self.val_res,
                num_per_scene=-1,
                gray=False,
                train=False,
            )
            for path in npz_paths
        ]
        return ConcatDataset(datasets)

    # DataLoaders --------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        self._ensure_train_dataset()
        assert self.train_dataset is not None, "setup() must be called before requesting dataloaders."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=lambda worker_id: _seed_worker(worker_id, self.seed),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return []
        loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(self.num_workers, 4),
            pin_memory=True,
            worker_init_fn=lambda worker_id: _seed_worker(worker_id, self.seed),
        )
        return loader
