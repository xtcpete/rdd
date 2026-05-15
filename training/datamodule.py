from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset.megadepth import MegaDepthDataset
from src.dataset.air_ground_dataset import AirGroundDataset
from src.dataset.aerial_megadepth import AerialMegaDepthDataset


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = (base_seed + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CombinedDataModule(pl.LightningDataModule):
    """Lightning DataModule with MegaDepth plus one optional aerial training split."""

    def __init__(
        self,
        megadepth_root_path: Path,
        val_indices_root: Path,
        batch_size: int,
        num_workers: int,
        training_res: int,
        train_detector: bool,
        seed: int,
        no_crop: bool = False,
        val_res: int = 1600,
        aerial_train_dataset: str = "air_ground",
        air_ground_root: Optional[Path] = None,
        air_ground_npz_root: Optional[Path] = None,
        aerial_megadepth_root: Optional[Path] = None,
        aerial_megadepth_npz_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.megadepth_root = Path(megadepth_root_path)
        self.val_indices_root = Path(val_indices_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_res = training_res
        self.train_detector = train_detector
        self.seed = seed
        self.no_crop = no_crop
        self._needs_resample = True
        self.val_res = val_res
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None
        self.aerial_train_dataset = aerial_train_dataset
        self.air_ground_root = Path(air_ground_root) if air_ground_root else None
        self.air_ground_npz_root = Path(air_ground_npz_root) if air_ground_npz_root else None
        self.aerial_megadepth_root = Path(aerial_megadepth_root) if aerial_megadepth_root else None
        self.aerial_megadepth_npz_path = Path(aerial_megadepth_npz_path) if aerial_megadepth_npz_path else None

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
        npz_paths = self._load_train_npz_paths(indices_root)

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

        datasets = [build_split(mode) for mode in self._train_modes()]
        if self.aerial_train_dataset == "air_ground":
            self._append_air_ground_datasets(datasets, num_per_scene, min_overlap, max_overlap)
        elif self.aerial_train_dataset == "aerial_megadepth":
            self._append_aerial_megadepth_datasets(datasets, num_per_scene, min_overlap, max_overlap)
        elif self.aerial_train_dataset != "none":
            raise ValueError(
                "Unknown aerial_train_dataset "
                f"'{self.aerial_train_dataset}'. Expected air_ground, aerial_megadepth, or none."
            )

        return ConcatDataset(datasets)

    def _append_air_ground_datasets(
        self,
        datasets: list[ConcatDataset],
        num_per_scene: int,
        min_overlap: float,
        max_overlap: float,
    ) -> None:
        if not (self.air_ground_root and self.air_ground_npz_root):
            return

        air_ground_npz_paths = sorted(glob.glob(str(self.air_ground_npz_root / '*.npz')))
        if not air_ground_npz_paths:
            return

        def build_ag_split(mode: str) -> ConcatDataset:
            ag_datasets = [
                AirGroundDataset(
                    root=self.air_ground_root,
                    npz_path=path,
                    num_per_scene=int(num_per_scene * 3),  # AirGround has fewer scenes, so sample more per scene.
                    image_size=self.training_res,
                    min_overlap_score=min_overlap,
                    max_overlap_score=max_overlap,
                    gray=False,
                    crop_or_scale=mode,
                    train=True,
                )
                for path in air_ground_npz_paths
            ]
            return ConcatDataset(ag_datasets)

        datasets.extend(build_ag_split(mode) for mode in self._train_modes())

    def _append_aerial_megadepth_datasets(
        self,
        datasets: list[ConcatDataset],
        num_per_scene: int,
        min_overlap: float,
        max_overlap: float,
    ) -> None:
        if self.aerial_megadepth_root is None or self.aerial_megadepth_npz_path is None:
            raise ValueError(
                "aerial_train_dataset='aerial_megadepth' requires "
                "aerial_megadepth_root and aerial_megadepth_npz_path."
            )

        npz_paths = self._resolve_aerial_megadepth_npz_paths()
        if not npz_paths:
            raise FileNotFoundError(f"No AerialMegaDepth index files found at {self.aerial_megadepth_npz_path}")

        def build_aerial_md_split(mode: str) -> ConcatDataset:
            aerial_md_datasets = [
                AerialMegaDepthDataset(
                    root=self.aerial_megadepth_root,
                    npz_path=path,
                    num_per_scene=num_per_scene,
                    image_size=self.training_res,
                    min_overlap_score=min_overlap,
                    max_overlap_score=max_overlap,
                    gray=False,
                    crop_or_scale=mode,
                    train=True,
                )
                for path in npz_paths
            ]
            return ConcatDataset(aerial_md_datasets)

        datasets.extend(build_aerial_md_split(mode) for mode in self._train_modes())

    def _train_modes(self) -> tuple[str, ...]:
        return ("scale",) if self.no_crop else ("crop", "scale")

    def _resolve_aerial_megadepth_npz_paths(self) -> list[str]:
        assert self.aerial_megadepth_root is not None
        assert self.aerial_megadepth_npz_path is not None

        npz_path = self.aerial_megadepth_npz_path
        if not npz_path.is_absolute() and not npz_path.exists():
            candidate = self.aerial_megadepth_root / npz_path
            if candidate.exists():
                npz_path = candidate

        if npz_path.is_dir():
            train_npz_paths = sorted(glob.glob(str(npz_path / '*train*.npz')))
            if train_npz_paths:
                return train_npz_paths
            return sorted(glob.glob(str(npz_path / '*.npz')))
        if npz_path.exists():
            return [str(npz_path)]
        raise FileNotFoundError(f"AerialMegaDepth index path not found: {npz_path}")

    def _load_train_npz_paths(self, indices_root: Path) -> list[str]:
        train_list_path = self.megadepth_root / 'megadepth_indices' / 'trainvaltest_list' / 'train_list.txt'
        if not train_list_path.exists():
            raise FileNotFoundError(f"MegaDepth train list not found: {train_list_path}")

        npz_paths: list[str] = []
        missing_entries: list[str] = []
        seen_paths: set[str] = set()

        for raw_entry in train_list_path.read_text().splitlines():
            entry = raw_entry.strip()
            if not entry or entry.startswith('#'):
                continue

            normalized_entry = entry if entry.endswith('.npz') else f'{entry}.npz'
            listed_path = Path(normalized_entry)
            if not listed_path.is_absolute():
                listed_path = indices_root / listed_path

            npz_path = str(listed_path)
            if not listed_path.exists():
                missing_entries.append(entry)
                continue
            if npz_path not in seen_paths:
                seen_paths.add(npz_path)
                npz_paths.append(npz_path)

        if missing_entries:
            missing_preview = ', '.join(missing_entries[:10])
            raise FileNotFoundError(
                f"MegaDepth train list contains missing indices under {indices_root}: {missing_preview}"
            )
        if not npz_paths:
            raise FileNotFoundError(f"No MegaDepth training indices found from {train_list_path}")

        return npz_paths

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
