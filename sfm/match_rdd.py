"""
Modified from hloc
https://github.com/cvg/Hierarchical-Localization.git
"""
import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union
from torch import nn
import h5py
import torch
from tqdm import tqdm

from hloc import logger
from hloc.utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

from src.matchers import LightGlue

class LGMatcher(nn.Module):
    default_conf = {
        "features": "rdd",
        "depth_confidence": -1,
        "width_confidence": -1,
    }
    
    required_inputs = [
        "image0",
        "keypoints0",
        "descriptors0",
        "image1",
        "keypoints1",
        "descriptors1",
    ]
    
    def __init__(self, conf):
        super().__init__()
        print(f"Initializing LightGlue with config: {pprint.pformat(conf)}")
        self.net = LightGlue(conf.copy().pop("features"))
    
    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)
    
    def _forward(self, data):
        data["descriptors0"] = data["descriptors0"].transpose(-1, -2)
        data["descriptors1"] = data["descriptors1"].transpose(-1, -2)

        return self.net(
            {
                "image0": {k[:-1]: v for k, v in data.items() if k[-1] == "0"},
                "image1": {k[:-1]: v for k, v in data.items() if k[-1] == "1"},
            }
        )
    
class MNN(nn.Module):
    default_conf = {
        "features": "rdd",
        "threshold": -1,
    }
    
    required_inputs = [
        "image0",
        "keypoints0",
        "descriptors0",
        "image1",
        "keypoints1",
        "descriptors1",
    ]
    
    def __init__(self, conf):
        super().__init__()
        # thresholds and temperature similar to DualSoftmaxMatcher defaults
        self.threshold = (
            conf.get("threshold", self.default_conf["threshold"]) if isinstance(conf, dict) else self.default_conf["threshold"]
        )
        self.inv_temperature = (
            conf.get("inv_temperature", 20) if isinstance(conf, dict) else 20
        )
    
    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)
    
    def dual_softmax(self, desc0: torch.Tensor, desc1: torch.Tensor, thr: float | None = None):
        """Dual-softmax mutual nearest matching.
        Inputs: desc0 [B,M,D], desc1 [B,N,D]. Returns indices (b,i,j) and P [B,M,N].
        """
        if thr is None:
            thr = self.threshold
        sim = torch.matmul(desc0, desc1.transpose(-1, -2)) * self.inv_temperature  # [B,M,N]
        P = sim.softmax(dim=-2) * sim.softmax(dim=-1)
        # mutual nearest with threshold
        row_max = P.max(dim=-1, keepdim=True).values
        col_max = P.max(dim=-2, keepdim=True).values
        mutual = (P == row_max) & (P == col_max) & (P >= thr)
        inds = torch.nonzero(mutual, as_tuple=False)  # [K,3]
        return inds, P

    def _forward(self, data):
        def _prep_desc(desc: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
            """Ensure descriptors follow the [B, N, D] layout expected by the matcher."""
            if desc.ndim != 3:
                raise ValueError("Descriptors are expected to be 3D tensors.")
            # Descriptors can be saved either as [B, D, N] (hloc convention) or already
            # as [B, N, D]. Only transpose when needed so that the keypoint dimension
            # always lines up with the descriptor dimension we iterate over below.
            if desc.shape[-2] != kpts.shape[-2] and desc.shape[-1] == kpts.shape[-2]:
                desc = desc.transpose(-1, -2)
            return desc.contiguous()

        kpts0 = data["keypoints0"]
        kpts1 = data["keypoints1"]
        desc0 = _prep_desc(data["descriptors0"], kpts0)
        desc1 = _prep_desc(data["descriptors1"], kpts1)
        B, M, D = desc0.shape
        _, N, _ = desc1.shape

        inds, P = self.dual_softmax(desc0, desc1, thr=self.threshold)

        # initialize outputs similar to LightGlue
        matches0_list, matches1_list = [], []
        scores0_list, scores1_list = [], []
        matches_pairs, scores_list = [], []

        if inds.numel() == 0:
            for b in range(B):
                matches0_list.append(torch.full((M,), -1, dtype=torch.int64, device=P.device))
                matches1_list.append(torch.full((N,), -1, dtype=torch.int64, device=P.device))
                scores0_list.append(torch.zeros((M,), dtype=P.dtype, device=P.device))
                scores1_list.append(torch.zeros((N,), dtype=P.dtype, device=P.device))
                matches_pairs.append(torch.empty((0, 2), dtype=torch.long, device=P.device))
                scores_list.append(torch.empty((0,), dtype=P.dtype, device=P.device))
        else:
            for b in range(B):
                sel = inds[:, 0] == b
                ij = inds[sel][:, 1:]  # [K,2]
                m0 = torch.full((M,), -1, dtype=torch.int64, device=P.device)
                m1 = torch.full((N,), -1, dtype=torch.int64, device=P.device)
                s0 = torch.zeros((M,), dtype=P.dtype, device=P.device)
                s1 = torch.zeros((N,), dtype=P.dtype, device=P.device)
                if ij.numel() > 0:
                    i_idx = ij[:, 0]
                    j_idx = ij[:, 1]
                    conf = P[b, i_idx, j_idx]
                    m0[i_idx] = j_idx
                    m1[j_idx] = i_idx
                    s0[i_idx] = conf
                    s1[j_idx] = conf
                    matches_pairs.append(torch.stack([i_idx, j_idx], dim=-1))
                    scores_list.append(conf)
                else:
                    matches_pairs.append(torch.empty((0, 2), dtype=torch.long, device=P.device))
                    scores_list.append(torch.empty((0,), dtype=P.dtype, device=P.device))
                matches0_list.append(m0)
                matches1_list.append(m1)
                scores0_list.append(s0)
                scores1_list.append(s1)

        matches0 = torch.stack(matches0_list, dim=0)
        matches1 = torch.stack(matches1_list, dim=0)
        matching_scores0 = torch.stack(scores0_list, dim=0)
        matching_scores1 = torch.stack(scores1_list, dim=0)

        # produce output dict compatible with LightGlue writer_fn
        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": matching_scores0,
            "matching_scores1": matching_scores1,
            "matches": matches_pairs,
            "scores": scores_list
        }
        

        

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    "rdd+lightglue": {
        "output": "matches-rdd-lightglue",
        "model": {
            "name": "lightglue",
            "features": "rdd",
        },
    },
    "rdd+mast3r":{
        "output": "matches-rdd-mast3r",
        "model": {
            "name": "mast3r",
            "features": "rdd",
            "device": "cuda",
        },
    },
    "rdd+mnn":{
        "output": "matches-rdd-mnn",
        "model": {
            "name": "mnn",
            "features": "rdd",
        },
    }
}


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            if 'image0' not in data:
                data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            if 'image1' not in data:
                data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
    device: str = "cpu",
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    logger.info(
        "Matching local features with configuration:" f"\n{pprint.pformat(conf)}"
    )   
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping the matching.")
        return
    
    if conf["model"]["name"] == "lightglue":
        model = LGMatcher(conf["model"])
    elif conf["model"]["name"] == "mnn":
        model = MNN(conf["model"])
    else:
        raise ValueError(f"Unknown matcher name: {conf['model']['name']}")
    model.eval()
    model.to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        
        # if matches are less than 25 then skip 
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info("Finished exporting matches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--features", type=str, default="feats-superpoint-n4096-r1024")
    parser.add_argument("--matches", type=Path)
    parser.add_argument(
        "--conf", type=str, default="superglue", choices=list(confs.keys())
    )
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
