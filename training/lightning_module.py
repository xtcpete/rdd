from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from src.utils.misc import flattenList
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict
import torch.nn.functional as F
from src.RDD import build
from src.RDD_helper import RDD_helper
from src.dataset import warper
from training.losses.detector_loss import DetectorLoss, compute_correspondence
from training.losses.descriptor_loss import DescriptorLoss
from training.utils import check_accuracy, compute_symmetrical_epipolar_errors, aggregate_metrics, compute_pose_errors
from training.plotting import (
    make_matching_figures,
    close_figures,
)
import numpy as np
from src.utils.comm import gather, all_gather


class RDDLightningModule(pl.LightningModule):
    """Lightning module handling both descriptor and detector training phases."""

    def __init__(
        self,
        stage: str,
        lr: float,
        lr_step_size: int,
        milestones: Sequence[int],
        gamma_steplr: float,
        weight_decay: float,
        descriptor_weights: Optional[Path],
        test_data_root: Path,
        model_config: Dict[str, Any],
        warmup_step: int = 1000,
        warmup_type: str = 'linear',
        warmup_ratio: float = 0.,
        plots_every: int = 50,
    ) -> None:
        super().__init__()
        if stage not in {"descriptor", "detector"}:
            raise ValueError(f"Unknown training stage '{stage}'")
        self.save_hyperparameters(
            {
                "stage": stage,
                "lr": lr,
                "lr_step_size": lr_step_size,
                "milestones": list(milestones),
                "gamma_steplr": gamma_steplr,
                "weight_decay": weight_decay,
                "descriptor_weights": str(descriptor_weights) if descriptor_weights else None,
                "test_data_root": str(test_data_root),
                "model_config": deepcopy(model_config),
                "warmup_step": warmup_step,
                "warmup_type": warmup_type,
                "warmup_ratio": warmup_ratio,
                "plots_every": plots_every,
            }
        )

        config = {"rdd": deepcopy(model_config)}
        if stage == "detector":
            config["rdd"]["train_detector"] = True
        self.joint_training = stage == "detector" and config["rdd"]["detector"].get("joint_training", False)

        self.model = build(config["rdd"])
        if stage == "detector":
            self.model.set_softdetect(top_k=512, scores_th=0.2)

        if descriptor_weights and stage == "detector":
            state = torch.load(descriptor_weights, map_location="cpu")
            if 'model' in state:
                state = state['model']
            elif 'state_dict' in state:
                state = state['state_dict']
                # if keys start with model. remove it
                if list(state.keys())[0].startswith('model.'):
                    state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
            self.model.load_state_dict(state, strict=False)
            if not self.joint_training:
                for param in self.model.descriptor.parameters():
                    param.requires_grad = False
        if stage == "descriptor":
            for param in self.model.detector.parameters():
                param.requires_grad = False
        self.stage = stage

        self.best_auc10 = 0.0
        self._skip_first_epoch_benchmark = True
        self.detector_loss = DetectorLoss(temperature=0.1, scores_th=0.1) if stage == "detector" else None
        self.descriptor_loss = DescriptorLoss(inv_temp=20) if stage == "descriptor" or self.joint_training else None
        self.warper = warper if stage == "descriptor" or self.joint_training else None
        self.validation_helper: Optional[RDD_helper] = None
        self.n_vals_plot = 32

    # Lightning hooks ---------------------------------------------------------------
    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma_steplr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_closure):
                    # plotting using the model's own detector for detector training
        # learning rate warm up
        warmup_step = self.hparams.warmup_step
        if self.trainer.global_step < warmup_step:
            if self.hparams.warmup_type == 'linear':
                base_lr = self.hparams.warmup_ratio * self.hparams.lr
                lr = base_lr + \
                    (self.trainer.global_step / warmup_step) * \
                    abs(self.hparams.lr - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.hparams.warmup_type == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.hparams.warmup_type}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    # Training ----------------------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if self.joint_training:
            return self._joint_training_step(batch, batch_idx)
        if self.stage == "detector":
            return self._detector_training_step(batch, batch_idx)
        return self._descriptor_training_step(batch, batch_idx)

    def _compute_descriptor_loss_from_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        feats0: torch.Tensor,
        feats1: torch.Tensor,
        hmap0: torch.Tensor,
        hmap1: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        positives_md_coarse = self.warper.spvs_coarse(batch, getattr(self.model, "stride", 4))

        loss_items = []
        acc_coarse_items: List[float] = []
        acc_kp_items: List[float] = []
        match_counts = 0

        for idx, positives in enumerate(positives_md_coarse):
            if positives is None or len(positives) < 30:
                continue
            if len(positives) > 10000:
                perm = torch.randperm(len(positives), device=positives.device)
                positives = positives[perm[:10000]]

            pts1 = positives[:, :2].long()
            pts2 = positives[:, 2:].long()

            h1 = hmap0[idx]
            h2 = hmap1[idx]

            m1 = feats0[idx, :, pts1[:, 1], pts1[:, 0]].permute(1, 0)
            m2 = feats1[idx, :, pts2[:, 1], pts2[:, 0]].permute(1, 0)

            loss_ds, loss_h, acc_kp = self.descriptor_loss(m1, m2, h1, h2, pts1, pts2)
            loss_items.append(loss_ds + loss_h)

            acc_coarse_items.append(check_accuracy(m1, m2))
            acc_kp_items.append(acc_kp)
            match_counts += len(m1)

        match_counts /= max(1, len(batch["image0"]))
        if not loss_items:
            zero = (feats0.sum() + feats1.sum() + hmap0.sum() + hmap1.sum()) * 0.0
            return zero, 0.0, 0.0, float(match_counts)

        loss = torch.stack(loss_items).mean()
        acc_coarse = sum(acc_coarse_items) / len(acc_coarse_items)
        acc_kp = sum(acc_kp_items) / len(acc_kp_items)
        return loss, acc_coarse, acc_kp, float(match_counts)

    def _plot_training_matches(self, batch: Dict[str, torch.Tensor], *, use_model_detector: bool) -> None:
        plots_every = self.hparams.plots_every
        if plots_every <= 0 or (self.global_step % plots_every != 0):
            return

        backbone = self.model.module if hasattr(self.model, "module") else self.model
        if self.validation_helper is None:
            self.validation_helper = RDD_helper(backbone)
        else:
            self.validation_helper.RDD = backbone

        match_fn = self.validation_helper.match if use_model_detector else self.validation_helper.match_3rd_party

        m_bids_list: List[torch.Tensor] = []
        mkpts0_list: List[torch.Tensor] = []
        mkpts1_list: List[torch.Tensor] = []
        mconf_list: List[torch.Tensor] = []

        with torch.inference_mode():
            num_samples = batch['image0'].shape[0]

            for idx in range(num_samples):
                image0 = batch["image0"][idx].detach().cpu()
                image1 = batch["image1"][idx].detach().cpu()
                img0_rgb = (image0.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype("uint8")
                img1_rgb = (image1.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype("uint8")

                mkpts0, mkpts1, conf = match_fn(
                    img0_rgb[..., ::-1].copy(),
                    img1_rgb[..., ::-1].copy(),
                    thr=0.01,
                    resize=None,
                )

                if self.trainer.global_rank == 0:
                    m_bids_list.append(torch.full((len(mkpts0),), idx, dtype=torch.long))
                    mkpts0_list.append(torch.from_numpy(mkpts0))
                    mkpts1_list.append(torch.from_numpy(mkpts1))
                    mconf_list.append(torch.from_numpy(conf))

        if self.trainer.global_rank == 0 and m_bids_list:
            device = next(backbone.parameters()).device
            batch.update({
                'm_bids': torch.cat(m_bids_list, dim=0).to(device),
                'mkpts0': torch.cat(mkpts0_list, dim=0).to(device),
                'mkpts1': torch.cat(mkpts1_list, dim=0).to(device),
                'mconf': torch.cat(mconf_list, dim=0).to(device),
            })

            if 'm_bids' in batch:
                compute_symmetrical_epipolar_errors(batch)
                figures = make_matching_figures(batch, 'evaluation')

                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step, close=True)

                close_figures(figures)

    def _detector_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        feats0, scores_map0, _ = self.model(batch["image0"])
        feats1, scores_map1, _ = self.model(batch["image1"])

        pred0 = {
                'descriptor_map': F.interpolate(feats0, size=scores_map0.shape[-2:], mode='bilinear', align_corners=True),
                'scores_map': scores_map0
            }
        pred1 = {
                'descriptor_map': F.interpolate(feats1, size=scores_map1.shape[-2:], mode='bilinear', align_corners=True),
                'scores_map': scores_map1
            }
        
        correspondences, pred0_with_rand, pred1_with_rand = compute_correspondence(
            self.model,
            pred0,
            pred1,
            batch,
            score_map_temperature=self.detector_loss.temperature,
            debug=False,
        )

        loss = self.detector_loss(correspondences, pred0_with_rand, pred1_with_rand)
        matches = sum(len(c.get("ids0_d", [])) for c in correspondences) / len(correspondences) if correspondences else 0

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/matches", matches, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)

        # Optional plotting (same behaviour as descriptor step)
        self._plot_training_matches(batch, use_model_detector=True)

        return loss

    def _joint_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        feats0, scores_map0, hmap0 = self.model(batch["image0"])
        feats1, scores_map1, hmap1 = self.model(batch["image1"])

        descriptor_loss, acc_coarse, acc_kp, descriptor_matches = self._compute_descriptor_loss_from_outputs(
            batch,
            feats0,
            feats1,
            hmap0,
            hmap1,
        )

        pred0 = {
                'descriptor_map': F.interpolate(feats0, size=scores_map0.shape[-2:], mode='bilinear', align_corners=True),
                'scores_map': scores_map0
            }
        pred1 = {
                'descriptor_map': F.interpolate(feats1, size=scores_map1.shape[-2:], mode='bilinear', align_corners=True),
                'scores_map': scores_map1
            }

        correspondences, pred0_with_rand, pred1_with_rand = compute_correspondence(
            self.model,
            pred0,
            pred1,
            batch,
            score_map_temperature=self.detector_loss.temperature,
            debug=False,
        )

        detector_loss = self.detector_loss(correspondences, pred0_with_rand, pred1_with_rand)
        detector_matches = sum(len(c.get("ids0_d", [])) for c in correspondences) / len(correspondences) if correspondences else 0
        loss = descriptor_loss + detector_loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_descriptor", descriptor_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_detector", detector_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/acc_coarse", acc_coarse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/acc_kp", acc_kp, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/matches_descriptor", descriptor_matches, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/matches_detector", detector_matches, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        self._plot_training_matches(batch, use_model_detector=True)

        return loss

    def _descriptor_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        feats1, _, hmap1 = self.model(batch["image0"])
        feats2, _, hmap2 = self.model(batch["image1"])
        loss, acc_coarse, acc_kp, match_counts = self._compute_descriptor_loss_from_outputs(
            batch,
            feats1,
            feats2,
            hmap1,
            hmap2,
        )

        self._plot_training_matches(batch, use_model_detector=False)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/acc_coarse", acc_coarse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/acc_kp", acc_kp, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/matches", float(match_counts), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    # Validation --------------------------------------------------------------------
    def _compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [(batch['epi_errs'].reshape(-1,1))[batch['m_bids'] == b].reshape(-1).cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'num_matches': [batch['mconf'].shape[0]], # batch size = 1 only
            }
        ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names


    def _val_inference(self, batch):
        device = batch['image0'].device
        with torch.no_grad():
            num_samples = batch['image0'].shape[0]
            m_bids_list = []
            mkpts0_list = []
            mkpts1_list = []
            mconf_list = []
            for idx in range(num_samples):
                image0 = batch["image0"][idx].detach().cpu()
                image1 = batch["image1"][idx].detach().cpu()
                img0_rgb = (image0.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype("uint8")
                img1_rgb = (image1.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype("uint8")
                if self.stage == 'detector':
                    # for detector, use its own keypoints
                    mkpts0, mkpts1, conf = self.validation_helper.match(
                        img0_rgb[..., ::-1].copy(),
                        img1_rgb[..., ::-1].copy(),
                        thr=0.01,
                        resize=None
                    )
                else:
                    mkpts0, mkpts1, conf = self.validation_helper.match_3rd_party(
                        img0_rgb[..., ::-1].copy(),
                        img1_rgb[..., ::-1].copy(),
                        thr=0.01,
                        resize=None
                    )

                m_bids_list.append(torch.full((len(mkpts0),), idx, dtype=torch.long))
                mkpts0_list.append(torch.from_numpy(mkpts0))
                mkpts1_list.append(torch.from_numpy(mkpts1))
                mconf_list.append(torch.from_numpy(conf))

            batch.update({
                'm_bids': torch.cat(m_bids_list, dim=0).to(device),
                'mkpts0': torch.cat(mkpts0_list, dim=0).to(device),
                'mkpts1': torch.cat(mkpts1_list, dim=0).to(device),
                'mconf': torch.cat(mconf_list, dim=0).to(device),
            })
    
    def on_validation_epoch_start(self):
        if self.stage == 'detector':
            if hasattr(self.model, "module"):
                self.model.module.set_softdetect(top_k=self.model.module.top_k)
            else:
                self.model.set_softdetect(top_k=self.model.top_k)
        self.validation_step_outputs = []
        # update RDD_helper for validation
        backbone = self.model.module if hasattr(self.model, "module") else self.model
        if self.validation_helper is None:
            self.validation_helper = RDD_helper(backbone)
        else:
            self.validation_helper.RDD = backbone
        
    def validation_step(self, batch, batch_idx):
        self._val_inference(batch)
        
        ret_dict, _ = self._compute_metrics(batch)
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {'evaluation': []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, mode='evaluation')
        
        preds = {
            **ret_dict,
            'figures': figures,
        }
        close_figures(figures)
                
        self.validation_step_outputs.append(preds)
        return preds

    def on_validation_epoch_end(self):
        if self.stage == 'detector':
            if hasattr(self.model, "module"):
                self.model.module.set_softdetect(top_k=512, scores_th=0.2)
            else:
                self.model.set_softdetect(top_k=512, scores_th=0.2)

        outputs = self.validation_step_outputs
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.ckpt_path and self.trainer._run_sanity_check:
                cur_epoch = -1

            # 1. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_metrics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(metrics)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # 2. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
                
                # Close all validation figures to prevent memory leaks
                close_figures(figures)

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

        self.validation_step_outputs.clear()
        # Clear GPU cache and force garbage collection after validation
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Remove validation helper weights from checkpoints."""
        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return
        to_remove = [key for key in state_dict.keys() if key.startswith("validation_helper.")]
        for key in to_remove:
            state_dict.pop(key)
