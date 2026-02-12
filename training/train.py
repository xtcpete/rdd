from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import math
from training.datamodule import CombinedDataModule
from training.lightning_module import RDDLightningModule


class ResampleDataCallback(pl.Callback):
    """Increment datamodule seed and trigger resampling after each training epoch."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        datamodule = trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "increment_seed_and_resample"):
            datamodule.increment_seed_and_resample()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightning-based RDD training pipeline.")

    parser.add_argument('--megadepth_root_path', type=Path, default=Path('./data/megadepth'),
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--test_data_root', type=Path, default=Path('./data/megadepth_test_1500'),
                        help='Path to the MegaDepth test dataset root directory.')
    parser.add_argument('--val_indices_root', type=Path, default=Path('megadepth_indices/scene_info_val_1500'),
                        help='Path (relative to MegaDepth root) to validation index files.')
    parser.add_argument('--air_ground_root', type=Path, default=Path('./data/air_ground_train/'),
                        help='Path to the AirGround dataset root directory.')
    parser.add_argument('--air_ground_npz_root', type=Path, default=Path('./data/air_ground_train/indices/'),
                        help='Path to the AirGround dataset indices directory.')
    parser.add_argument('--ckpt_save_path', type=Path, default=Path('./runs/'),
                        help='Directory to save checkpoints and logs.')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker count.')
    parser.add_argument('--training_res', type=int, default=800, help='Training resolution (short side).')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='Detector learning rate.')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Step size (in epochs) for StepLR scheduler.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5, help='Gamma value for StepLR scheduler.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW.')

    parser.add_argument('--descriptor_epochs', type=int, default=20, help='Number of descriptor training epochs.')
    parser.add_argument('--detector_epochs', type=int, default=5, help='Number of detector training epochs.')

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--accelerator', type=str, default='auto', help='Lightning accelerator (e.g. auto, gpu).')
    parser.add_argument('--devices', type=str, default='auto', help='Devices to use (e.g. auto, 1, 0,1).')
    parser.add_argument('--strategy', type=str, default=None, help='Lightning strategy (e.g. ddp).')
    parser.add_argument('--precision', type=str, default='32-true', help='Mixed precision policy.')

    parser.add_argument('--resume_descriptor', type=Path, default=None,
                        help='Checkpoint path to resume descriptor stage from.')
    parser.add_argument('--resume_detector', type=Path, default=None,
                        help='Checkpoint path to resume detector stage from.')
    parser.add_argument('--detector_from', type=Path, default=None,
                        help='Descriptor checkpoint used to initialise detector stage.')

    parser.add_argument('--warmup_step', type=int, default=1000,
                        help='Number of warmup steps (scaled with batch size) for learning rate scheduler.')

    parser.add_argument('--train_detector', dest='train_detector', action='store_true',
                        help='Enable detector training stage.')
    parser.set_defaults(train_detector=False)

    return parser.parse_args()


def devices_arg(devices: str):
    if devices == 'auto':
        return 'auto'
    if devices.isdigit():
        return int(devices)
    if ',' in devices:
        return [int(d) for d in devices.split(',')]
    return devices


def prepare_trainer_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "accelerator": args.accelerator,
        "devices": devices_arg(args.devices),
        "precision": args.precision,
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 50,
        "reload_dataloaders_every_n_epochs": 1,
    }
    if args.strategy:
        kwargs["strategy"] = args.strategy
    return kwargs


def main() -> None: 
    args = parse_args()
    if args.resume_detector is not None:
        args.train_detector = True
    CANONICAL_BS = 32 # canonical batch size for lr scaling

    pl.seed_everything(args.seed, workers=True)

    ckpt_root = args.ckpt_save_path.resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_root = ckpt_root / 'logdir'
    log_root.mkdir(exist_ok=True)
    
    trainer_kwargs = prepare_trainer_kwargs(args)

    descriptor_best: Optional[Path] = None

    # scaling lr and warmup steps
    if trainer_kwargs["devices"] == 'auto':
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    elif isinstance(trainer_kwargs["devices"], int):
        num_gpus = trainer_kwargs["devices"]
    elif isinstance(trainer_kwargs["devices"], list):
        num_gpus = len(trainer_kwargs["devices"])
    
    true_batch_size = args.batch_size * num_gpus
    _scaling = true_batch_size / CANONICAL_BS
    true_lr = args.lr * _scaling
    warmup_step = math.floor(args.warmup_step / _scaling)

    if not args.train_detector:
        descriptor_module = RDDLightningModule(
            stage='descriptor',
            lr=true_lr,
            lr_step_size=args.lr_step_size,
            gamma_steplr=args.gamma_steplr,
            weight_decay=args.weight_decay,
            descriptor_weights=None,
            test_data_root=args.test_data_root,
            warmup_step=warmup_step,
        )
        descriptor_dm = CombinedDataModule(
            megadepth_root_path=args.megadepth_root_path,
            val_indices_root=args.val_indices_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            training_res=args.training_res,
            train_detector=False,
            seed=args.seed,
            air_ground_root=args.air_ground_root,
            air_ground_npz_root=args.air_ground_npz_root,
        )
        descriptor_ckpt_dir = ckpt_root / 'descriptor'
        descriptor_ckpt_dir.mkdir(exist_ok=True)
        descriptor_checkpoint = ModelCheckpoint(
            dirpath=descriptor_ckpt_dir,
            filename='rdd-{epoch:02d}-{auc@10:.3f}',
            monitor='auc@10',
            mode='max',
            save_top_k=5,
            verbose=True,
        )
        descriptor_logger = TensorBoardLogger(str(log_root), name='descriptor')
        descriptor_trainer = pl.Trainer(
            max_epochs=args.descriptor_epochs,
            default_root_dir=str(descriptor_ckpt_dir),
            callbacks=[descriptor_checkpoint, ResampleDataCallback()],
            logger=descriptor_logger,
            check_val_every_n_epoch=1,
            **trainer_kwargs,
        )
        descriptor_trainer.fit(
            descriptor_module,
            datamodule=descriptor_dm,
            ckpt_path=str(args.resume_descriptor) if args.resume_descriptor else None,
        )
        if descriptor_checkpoint.best_model_path:
            descriptor_best = Path(descriptor_checkpoint.best_model_path)
        elif args.resume_descriptor:
            descriptor_best = args.resume_descriptor

    if args.train_detector:
        init_weights = args.detector_from or descriptor_best
        if init_weights is None:
            raise ValueError("Detector stage requested but no descriptor weights were provided. "
                             "Train the descriptor stage first or pass --detector_from <ckpt>.")
        detector_module = RDDLightningModule(
            stage='detector',
            lr=true_lr,
            lr_step_size=args.lr_step_size,
            gamma_steplr=args.gamma_steplr,
            weight_decay=args.weight_decay,
            descriptor_weights=init_weights,
            test_data_root=args.test_data_root,
            warmup_step=warmup_step,
        )
        detector_dm = CombinedDataModule(
            megadepth_root_path=args.megadepth_root_path,
            val_indices_root=args.val_indices_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            training_res=args.training_res,
            train_detector=True,
            seed=args.seed,
            air_ground_root=args.air_ground_root,
            air_ground_npz_root=args.air_ground_npz_root,
        )
        detector_ckpt_dir = ckpt_root / 'detector'
        detector_ckpt_dir.mkdir(exist_ok=True)
        detector_checkpoint = ModelCheckpoint(
            dirpath=detector_ckpt_dir,
            filename='rdd-{epoch:02d}-{auc@10:.3f}',
            monitor='auc@10',
            mode='max',
            save_top_k=5,
            verbose=True,
        )
        detector_logger = TensorBoardLogger(str(log_root), name='detector')
        detector_trainer = pl.Trainer(
            max_epochs=args.detector_epochs,
            default_root_dir=str(detector_ckpt_dir),
            callbacks=[detector_checkpoint, ResampleDataCallback()],
            logger=detector_logger,
            check_val_every_n_epoch=1,
            val_check_interval= 32000 // (true_batch_size),
            **trainer_kwargs,
        )
        detector_trainer.fit(
            detector_module,
            datamodule=detector_dm,
            ckpt_path=str(args.resume_detector) if args.resume_detector else None,
        )


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
