import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gc
from torch.nn import functional as F
import torch

def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth' or dataset_name == 'airground':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic'):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = (data['image0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0'][b_mask].detach().cpu().numpy()
    kpts1 = data['mkpts1'][b_mask].detach().cpu().numpy()

    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].detach().cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    # n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    # recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        # f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure

def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()

def make_matching_figure_gt(data):
    figures = {'gt_matches': []}
    for b_id in range(data['image0'].size(0)):
        img0 = (data['image0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.int32)
        kpts0 = data['mkpts0_gt'][b_id].numpy()
        kpts1 = data['mkpts1_gt'][b_id].numpy()
        
        color = ['green' for _ in range(len(kpts0))]
        text = [
            f'#GT Matches {len(kpts0)}',
        ]
        figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                      color, text=text)
        figures['gt_matches'].append(figure)
    return figures

def make_depth_figures(data):
    """ Make depth figures for a batch."""
    figures = {'depth': []}
    
    for b_id in range(data['image0'].size(0)):
        depth0 = data['pred_depth0'][b_id].detach().cpu().numpy()
        depth1 = data['pred_depth1'][b_id].detach().cpu().numpy()
        
        pred_depth0 = np.exp(depth0) # predicted depth is log-scaled
        pred_depth1 = np.exp(depth1)

        pred_inv_depth0 = 1/pred_depth0
        pred_inv_depth1 = 1/pred_depth1

        pred_inv_depth0 = pred_inv_depth0/np.amax(pred_inv_depth0)
        pred_inv_depth1 = pred_inv_depth1/np.amax(pred_inv_depth1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(pred_depth0[0], cmap='gray')
        axes[0].axis('off')
        axes[1].imshow(pred_depth1[0], cmap='gray')
        axes[1].axis('off')
        figures['depth'].append(fig)

    return figures

def make_matching_figures(data, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'gt']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha='dynamic')
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures

def make_matchability_figures(data):
    """ Make matchability figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    figures = {'matchability': []}
    for b_id in range(data['image0'].size(0)):
        matchability0 = data['matchability0'][b_id].detach().cpu().numpy()
        matchability1 = data['matchability1'][b_id].detach().cpu().numpy()
        fig = make_matchability_figure(matchability0, matchability1)
        figures['matchability'].append(fig)
    return figures
        

def make_matchability_figure(matchability0, matchability1, title=None):
    """
    Visualize matchability confidence maps for two images.
    
    Args:
        matchability0 (torch.Tensor): [H0, W0], confidence map for image 0
        matchability1 (torch.Tensor): [H1, W1], confidence map for image 1
        title (str, optional): Title of the figure
    
    Returns:
        fig (plt.Figure): Matplotlib figure object
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(matchability0[0], cmap='viridis', vmin=0, vmax=1)
    axes[0].axis('off')

    im1 = axes[1].imshow(matchability1[0], cmap='viridis', vmin=0, vmax=1)
    axes[1].axis('off')

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    return fig

def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)


def close_figures(figures_dict):
    """Close all matplotlib figures in a figures dictionary to prevent memory leaks."""
    for key, fig_list in figures_dict.items():
        for fig in fig_list:
            plt.close(fig)
    # Force garbage collection after closing figures
    gc.collect()