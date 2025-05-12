import argparse
import os
import time
import sys
import glob
from pathlib import Path
from RDD.RDD_helper import RDD_helper
import torch.distributed

def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")

    parser.add_argument('--megadepth_root_path', type=str, default='./data/megadepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--test_data_root', type=str, default='./data/megadepth_test_1500',
                        help='Path to the MegaDepth test dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--model_name', type=str, default='RDD',
                        help='Name of the model to save.')
    parser.add_argument('--air_ground_root_path', type=str, default='./data/air_ground_data_2/AirGround')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training. Default is 4.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate. Default is 0.0001.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=int,
                        default=800, help='Training resolution as width,height. Default is 800 for training descriptor.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--test_every_iter', type=int, default=2000,
                        help='Save checkpoints every N steps. Default is 2000.')
    parser.add_argument('--weights', type=str, default=None,)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--enc_n_points', type=int, default=8)
    parser.add_argument('--num_feature_levels', type=int, default=5)
    parser.add_argument('--train_detector', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='./configs/default.yaml')
    args = parser.parse_args()

    return args

args = parse_arguments()

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from RDD.RDD import build
from training.utils import *
from training.losses import *
from benchmarks.mega_1500 import MegaDepthPoseMNNBenchmark
from RDD.dataset.megadepth.megadepth import MegaDepthDataset
from RDD.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, WeightedRandomSampler
from training.losses.detector_loss import compute_correspondence, DetectorLoss
from training.losses.descriptor_loss import DescriptorLoss
import tqdm
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from datetime import timedelta
from RDD.utils import read_config
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk, 
        the network training itself is quite fast.
    """
    
    def __init__(self, rank, args=None):
        config = read_config(args.config_path)
        
        config['num_encoder_layers'] = args.num_encoder_layers
        config['enc_n_points'] = args.enc_n_points
        config['num_feature_levels'] = args.num_feature_levels
        config['train_detector'] = args.train_detector
        config['weights'] = args.weights
        
        # distributed training
        if args.distributed:
            print(f"Training in distributed mode with {args.n_gpus} GPUs")
            assert torch.cuda.is_available()
            device = rank

            torch.distributed.init_process_group(
                backend="nccl",
                world_size=args.n_gpus,
                rank=device,
                init_method="file://" + str(args.lock_file),
                timeout=timedelta(seconds=2000)
            )
            torch.cuda.set_device(device)

            # adjust batch size and num of workers since these are per GPU
            batch_size = int(args.batch_size / args.n_gpus)
            self.n_gpus = args.n_gpus
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = args.batch_size
        print(f"Using device {device}")
        
        self.seed = 0
        self.set_seed(self.seed)
        self.training_res = args.training_res
        self.dev = device
        config['device'] = device
        model = build(config)
        
        self.rank = rank
        
        if args.weights is not None:
            print('Loading weights from ', args.weights)
            model.load_state_dict(torch.load(args.weights, map_location='cpu'))
                
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], find_unused_parameters=True
            )
        else:
            self.model = model.to(device)
        
        self.saved_ckpts = []
        self.best = -1.0
        self.best_loss = 1e6
        self.fine_weight = 1.0
        self.dual_softmax_weight = 1.0
        self.heatmaps_weight = 1.0
        #Setup optimizer 
        self.batch_size = batch_size
        self.epochs = args.epochs
        self.opt = optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()) , lr = args.lr, weight_decay=1e-4)

        # losses
        if args.train_detector:
            self.DetectorLoss = DetectorLoss(temperature=0.1, scores_th=0.1)
        else:
            self.DescriptorLoss = DescriptorLoss(inv_temp=20, dual_softmax_weight=1, heatmap_weight=1)
        
        self.benchmark = MegaDepthPoseMNNBenchmark(data_root=args.test_data_root)

        ##################### MEGADEPTH INIT ##########################
        
        TRAIN_BASE_PATH = f"{args.megadepth_root_path}/megadepth_indices"
        print('Loading MegaDepth dataset from ', TRAIN_BASE_PATH)
        TRAINVAL_DATA_SOURCE = args.megadepth_root_path
        self.TRAINVAL_DATA_SOURCE = TRAINVAL_DATA_SOURCE
        TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
        self.TRAIN_NPZ_ROOT = TRAIN_NPZ_ROOT
        npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
        self.npz_paths = npz_paths
        self.epoch = 0
        self.create_data_loader()

        ##################### MEGADEPTH INIT END #######################

        os.makedirs(args.ckpt_save_path, exist_ok=True)
        os.makedirs(args.ckpt_save_path / 'logdir', exist_ok=True)

        self.save_ckpt_every = args.save_ckpt_every
        self.ckpt_save_path = args.ckpt_save_path
        if rank == 0:
            self.writer = SummaryWriter(str(self.ckpt_save_path) + f'/logdir/{args.model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        else:
            self.writer = None
        self.model_name = args.model_name
        
        if args.distributed:
            self.scheduler = MultiStepLR(self.opt, milestones=[2, 4, 8, 16], gamma=args.gamma_steplr)
        else:
            self.scheduler = StepLR(self.opt, step_size=args.test_every_iter, gamma=args.gamma_steplr)
        
    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def create_data_loader(self):
        # Create sampler

            
        if not args.train_detector:
            mega_crop = torch.utils.data.ConcatDataset( [MegaDepthDataset(root = self.TRAINVAL_DATA_SOURCE,
                            npz_path = path, min_overlap_score=0.01, max_overlap_score=0.7, image_size=self.training_res, num_per_scene=200, gray=False, crop_or_scale='crop') for path in self.npz_paths] )
            mega_scale = torch.utils.data.ConcatDataset( [MegaDepthDataset(root = self.TRAINVAL_DATA_SOURCE,
                            npz_path = path, min_overlap_score=0.01, max_overlap_score=0.7, image_size=self.training_res, num_per_scene=200, gray=False, crop_or_scale='scale') for path in self.npz_paths] )
            combined_dataset = torch.utils.data.ConcatDataset([mega_crop, mega_scale])
            
        else:

            mega_crop = torch.utils.data.ConcatDataset( [MegaDepthDataset(root = self.TRAINVAL_DATA_SOURCE,
                            npz_path = path, min_overlap_score=0.1, max_overlap_score=0.8, image_size=self.training_res, num_per_scene=100, gray=False, crop_or_scale='crop') for path in self.npz_paths] )
            mega_scale = torch.utils.data.ConcatDataset( [MegaDepthDataset(root = self.TRAINVAL_DATA_SOURCE,
                            npz_path = path, min_overlap_score=0.1, max_overlap_score=0.8, image_size=self.training_res, num_per_scene=100, gray=False, crop_or_scale='scale') for path in self.npz_paths] )
            combined_dataset = torch.utils.data.ConcatDataset([mega_crop, mega_scale])

        # Create sampler
        if args.distributed:
            sampler = DistributedSampler(combined_dataset, rank=self.rank, num_replicas=self.n_gpus)
        else:
            # Create sampler
            sampler = RandomSampler(combined_dataset)

        # Create single DataLoader with combined dataset
        self.data_loader = DataLoader(combined_dataset, 
                                    batch_size=self.batch_size, 
                                    sampler=sampler, 
                                    num_workers=4,
                                    pin_memory=True)

    def validate(self, total_steps):
            
        with torch.no_grad():
            
            
            if args.train_detector:
                method = 'sparse'
            else:
                method = 'aliked'
            
            if args.distributed:
                self.model.module.eval()
                model_helper = RDD_helper(self.model.module)
                test_out = self.benchmark.benchmark(model_helper, model_name='experiment', plot_every_iter=1, plot=False, method=method)
            else:
                self.model.eval()
                model_helper = RDD_helper(self.model)
                test_out = self.benchmark.benchmark(model_helper, model_name='experiment', plot_every_iter=1, plot=False, method=method)
                
            auc5 = test_out['auc_5']
            auc10 = test_out['auc_10']
            auc20 = test_out['auc_20']
            if self.rank == 0:
                self.writer.add_scalar('Accuracy/auc5', auc5, total_steps)
                self.writer.add_scalar('Accuracy/auc10', auc10, total_steps)
                self.writer.add_scalar('Accuracy/auc20', auc20, total_steps)
                if auc5 > self.best:
                    self.best = auc5
                    if args.distributed:
                        torch.save(self.model.module.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_best.pth')
                    else:
                        torch.save(self.model.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_best.pth')
        
        self.model.train()
        
    
    def _inference(self, d):
        if d is not None:
            for k in d.keys():
                if isinstance(d[k], torch.Tensor):
                    d[k] = d[k].to(self.dev)
            p1, p2 = d['image0'], d['image1']
                
            if not args.train_detector:
                positives_md_coarse = megadepth_warper.spvs_coarse(d, self.stride)
        
        with torch.no_grad():
            p1 = p1 ; p2 = p2
            if not args.train_detector:
                positives_c = positives_md_coarse
            
        
        # Check if batch is corrupted with too few correspondences
        is_corrupted = False
        if not args.train_detector:
            for p in positives_c:
                if len(p) < 30:
                    is_corrupted = True

        if is_corrupted:
            return None, None, None, None

        # Forward pass
        
        feats1, scores_map1, hmap1 = self.model(p1)
        feats2, scores_map2, hmap2 = self.model(p2)
        
        if args.train_detector:
            
            # move all tensors on batch to GPU
            for k in d.keys():
                if isinstance(d[k], torch.Tensor):
                    d[k] = d[k].to(self.dev)
                elif isinstance(d[k], dict):
                    for k2 in d[k].keys():
                        if isinstance(d[k][k2], torch.Tensor):
                            d[k][k2] = d[k][k2].to(self.dev)
                            
            # Get positive correspondencies 
            pred0 = {'descriptor_map': F.interpolate(feats1, size=scores_map1.shape[-2:], mode='bilinear', align_corners=True), 'scores_map': scores_map1 }
            pred1 = {'descriptor_map': F.interpolate(feats2, size=scores_map2.shape[-2:], mode='bilinear', align_corners=True), 'scores_map': scores_map2 }
            if args.distributed:
                correspondences, pred0_with_rand, pred1_with_rand = compute_correspondence(self.model.module, pred0, pred1, d, debug=True)
            else:
                correspondences, pred0_with_rand, pred1_with_rand = compute_correspondence(self.model, pred0, pred1, d, debug=False)
            
            loss_kp = self.DetectorLoss(correspondences, pred0_with_rand, pred1_with_rand)
            
            loss = loss_kp
            acc_coarse, acc_kp, nb_coarse = 0, 0, 0
        else:
                
            loss_items = []
            acc_coarse_items = []
            acc_kp_items = []
            
            for b in range(len(positives_c)):
                
                if len(positives_c[b]) > 10000:
                    positives = positives_c[b][torch.randperm(len(positives_c[b]))[:10000]]
                else:
                    positives = positives_c[b]
                # Get positive correspondencies
                pts1, pts2 = positives[:, :2], positives[:, 2:]
                
                h1 = hmap1[b, :, :, :]
                h2 = hmap2[b, :, :, :]
                
                m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0) 
                # Compute losses
                loss_ds, loss_h, acc_kp = self.DescriptorLoss(m1, m2, h1, h2, pts1, pts2)
                
                loss_items.append(loss_ds.unsqueeze(0))

                acc_coarse = check_accuracy1(m1, m2)
                acc_kp_items.append(acc_kp)
                acc_coarse_items.append(acc_coarse)

            nb_coarse = len(m1)
            loss = loss_kp if args.train_detector else torch.cat(loss_items, -1).mean()
            acc_coarse = sum(acc_coarse_items) / len(acc_coarse_items)
            acc_kp = sum(acc_kp_items) / len(acc_kp_items)
        
        return loss, acc_coarse, acc_kp, nb_coarse
    
    def train(self):

        self.model.train()
        self.stride = 4 if args.num_feature_levels == 5 else 8
        total_steps = 0
        
        for epoch in range(self.epochs):
            
            if args.distributed:
                self.data_loader.sampler.set_epoch(epoch)
            pbar = tqdm.tqdm(total=len(self.data_loader), desc=f"Epoch {epoch+1}/{args.epochs}") if self.rank == 0 else None
           
            for i, d in enumerate(self.data_loader):
                
                loss, acc_coarse, acc_kp, nb_coarse = self._inference(d)
                
                if loss is None:
                    continue
                
                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()

                if (total_steps + 1) % self.save_ckpt_every == 0 and self.rank == 0:
                    print('saving iter ', total_steps + 1)
                    if args.distributed:
                        torch.save(self.model.module.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_{total_steps + 1}.pth')
                    else:
                        torch.save(self.model.state_dict(), str(self.ckpt_save_path) + f'/{self.model_name}_{total_steps + 1}.pth')
                    self.saved_ckpts.append(total_steps + 1)
                    if len(self.saved_ckpts) > 5:
                        os.remove(str(self.ckpt_save_path) + f'/{self.model_name}_{self.saved_ckpts[0]}.pth')
                        self.saved_ckpts = self.saved_ckpts[1:]
                
                if args.distributed:
                    torch.distributed.barrier()
                    
                if (total_steps+1) % args.test_every_iter == 0:
                    self.validate(total_steps)
                
                if pbar is not None:
                    
                    if args.train_detector:
                        pbar.set_description( 'Loss: {:.4f} '.format(loss.item()) )
                    else:
                        pbar.set_description( 'Loss: {:.4f} acc_coarse {:.3f} acc_kp: {:.3f} #matches_c: {:d}'.format(
                                                                                loss.item(), acc_coarse, acc_kp, nb_coarse) )
                    pbar.update(1)

                # Log metrics
                if self.rank == 0:
                    self.writer.add_scalar('Loss/total', loss.item(), total_steps)
                    self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, total_steps)
                    self.writer.add_scalar('Count/matches_coarse', nb_coarse, total_steps)
                
                if not args.distributed:
                    self.scheduler.step()
                total_steps = total_steps + 1

            self.validate(total_steps)
            if self.rank == 0:
                print('Epoch ', epoch, ' done.')
                print('Creating new data loader with seed ', self.seed)
            self.seed = self.seed + 1
            self.set_seed(self.seed)
            self.scheduler.step()
            self.epoch = self.epoch + 1
            self.create_data_loader()
             
def main_worker(rank, args):
    trainer = Trainer(
        rank=rank,
        args=args
    )

    # The most fun part
    trainer.train()

if __name__ == '__main__':
    if args.distributed:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)  
    
    if not Path(args.ckpt_save_path).exists():
        os.makedirs(args.ckpt_save_path)
    
    args.ckpt_save_path = Path(args.ckpt_save_path).resolve()

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = Path(args.ckpt_save_path) / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        
        # Each process gets its own rank and dataset
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus, args=(args,)
        )
    else:
        main_worker(0, args)