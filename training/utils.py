"""Utility helpers used by the training pipeline."""

from __future__ import annotations
from kornia.geometry.epipolar import numeric
import torch
from kornia.geometry.conversions import convert_points_to_homogeneous
import numpy as np
from collections import OrderedDict
import cv2

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def check_accuracy(embeddings0: torch.Tensor, embeddings1: torch.Tensor) -> float:
    """Return the top-1 matching accuracy between two embedding sets."""
    if embeddings0.shape[0] == 0:
        return 0.0

    with torch.no_grad():
        similarity = embeddings0 @ embeddings1.t()
        nearest_idx = similarity.argmax(dim=1)
        correct = (nearest_idx == torch.arange(len(embeddings0), device=embeddings0.device)).sum()
    return float(correct) / float(len(embeddings0))

def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d

def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    warp_params = data['warp01_params']
    Tx = numeric.cross_product_matrix(warp_params['pose01'][:, :3, 3])
    E_mat = Tx @ warp_params['pose01'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0']
    pts1 = data['mkpts1']
    intrinsics0 = warp_params['intrinsics0']
    intrinsics1 = warp_params['intrinsics1']

    bbox0 = warp_params.get('bbox0', None)
    bbox1 = warp_params.get('bbox1', None)
    if bbox0 is not None and bbox1 is not None:
        bbox0_xy = bbox0[:, [1, 0]].to(device=pts0.device, dtype=pts0.dtype)
        bbox1_xy = bbox1[:, [1, 0]].to(device=pts1.device, dtype=pts1.dtype)
    else:
        bbox0_xy = bbox1_xy = None

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        if not torch.any(mask):
            continue
        pts0_bs = pts0[mask]
        pts1_bs = pts1[mask]
        K0_bs = intrinsics0[bs]
        K1_bs = intrinsics1[bs]

        if bbox0_xy is not None:
            offset0 = bbox0_xy[bs]
            offset1 = bbox1_xy[bs]
            pts0_bs = pts0_bs + offset0
            pts1_bs = pts1_bs + offset1

        epi_errs.append(
            symmetric_epipolar_distance(pts0_bs, pts1_bs, E_mat[bs], K0_bs, K1_bs))
    if epi_errs:
        epi_errs = torch.cat(epi_errs, dim=0)
    else:
        epi_errs = torch.empty(0, device=pts0.device, dtype=pts0.dtype)

    data.update({'epi_errs': epi_errs})

def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def aggregate_metrics(metrics, epi_err_thr=1e-4, EVAL_TIMES=1):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())

    # pose auc
    angular_thresholds = [5, 10, 20]

    if EVAL_TIMES >= 1:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0).reshape(-1, EVAL_TIMES)[unq_ids].reshape(-1)
    else:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)
    
    u_num_matches = np.array(metrics['num_matches'], dtype=object)[unq_ids]
    num_matches = {f'num_matches': u_num_matches.mean() }
    return {**aucs, **precs, **num_matches}

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def compute_pose_errors(data):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = 0.5
    conf = 0.99999
    RANSAC = "RANSAC"
    EVAL_TIMES = 1
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0'].cpu().numpy()
    pts1 = data['mkpts1'].cpu().numpy()
    K0 = data['warp01_params']['intrinsics0'].cpu().numpy()
    K1 = data['warp01_params']['intrinsics1'].cpu().numpy()
    bbox0 = data['warp01_params'].get('bbox0', None)
    bbox1 = data['warp01_params'].get('bbox1', None)
    if bbox0 is not None and bbox1 is not None:
        bbox0_np = bbox0.cpu().numpy()
        bbox1_np = bbox1.cpu().numpy()
        bbox0_xy = np.stack([bbox0_np[:, 1], bbox0_np[:, 0]], axis=1)
        bbox1_xy = np.stack([bbox1_np[:, 1], bbox1_np[:, 0]], axis=1)
    else:
        bbox0_xy = bbox1_xy = None
    T_0to1 = data['warp01_params']['pose01'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        if EVAL_TIMES >= 1:
            bpts0, bpts1 = pts0[mask], pts1[mask]
            K0_bs = K0[bs]
            K1_bs = K1[bs]
            if bbox0_xy is not None:
                offset0 = bbox0_xy[bs]
                offset1 = bbox1_xy[bs]
                bpts0 = bpts0 + offset0
                bpts1 = bpts1 + offset1
            R_list, T_list, inliers_list = [], [], []
            # for _ in range(config.MODEL.EVAL_TIMES):
            for _ in range(5):
                shuffling = np.random.permutation(np.arange(len(bpts0)))
                if _ >= EVAL_TIMES:
                    continue
                bpts0 = bpts0[shuffling]
                bpts1 = bpts1[shuffling]
                
                if RANSAC == 'RANSAC':
                    ret = estimate_pose(bpts0, bpts1, K0_bs, K1_bs, pixel_thr, conf=conf)
                    if ret is None:
                        R_list.append(np.inf)
                        T_list.append(np.inf)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        R, t, inliers = ret
                        t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
                        R_list.append(R_err)
                        T_list.append(t_err)
                        inliers_list.append(inliers)

                else:
                    raise ValueError(f"Unknown RANSAC method: {RANSAC}")

            data['R_errs'].append(R_list)
            data['t_errs'].append(T_list)
            data['inliers'].append(inliers_list[0])