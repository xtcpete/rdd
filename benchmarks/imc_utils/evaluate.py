#!/usr/bin/env python3
"""
Evaluate SfM reconstructions by robustly aligning predicted poses to GT poses and
reporting absolute rotation / translation errors. Optionally visualize both GT
and aligned predictions in 3D.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .read_write_model import Image as ColmapImage
from .read_write_model import read_model

try:
    from viz_3d import init_figure, plot_camera, plot_points

    VIZ_AVAILABLE = True
except ModuleNotFoundError:
    init_figure = plot_camera = plot_points = None  # type: ignore
    VIZ_AVAILABLE = False


Color = str

ROTATION_THRESHOLD_GREEN = 3.0
ROTATION_THRESHOLD_RED = 5.0


@dataclass
class Pose:
    name: str
    rotation_wc: np.ndarray  # world -> camera
    rotation_cw: np.ndarray  # camera -> world
    center: np.ndarray
    camera_id: int


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


IDENTITY_SIMILARITY = SimilarityTransform(
    scale=1.0, rotation=np.eye(3), translation=np.zeros(3)
)


def discover_scenes(results_root: Path) -> List[str]:
    if not results_root.exists():
        return []
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()])


def build_pose(image: ColmapImage) -> Pose:
    rotation_wc = image.qvec2rotmat()
    center = -rotation_wc.T @ image.tvec
    return Pose(
        name=image.name,
        rotation_wc=rotation_wc,
        rotation_cw=rotation_wc.T,
        center=center,
        camera_id=image.camera_id,
    )


def load_model(model_dir: Path, ext: str) -> Tuple[Mapping[int, object], Dict[str, Pose], Mapping[int, object]]:
    cameras, images, points3d = read_model(str(model_dir), ext)
    poses = {image.name: build_pose(image) for image in images.values()}
    return cameras, poses, points3d


def parse_float_sequence(raw: str, expected_len: int) -> np.ndarray:
    tokens = [tok.strip() for tok in raw.replace(",", ";").split(";")]
    values = [float(tok) for tok in tokens if tok]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values but got {len(values)}")
    return np.array(values, dtype=float)


def load_gt_poses_from_csv(
    csv_path: Path, dataset_filter: Optional[str] = None
) -> Dict[str, Dict[str, Pose]]:
    scene_to_poses: Dict[str, Dict[str, Pose]] = {}
    camera_counter = 0
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return scene_to_poses
        rotation_key = next(
            (key for key in ("rotation_matrix", "rotation") if key in reader.fieldnames), None
        )
        translation_key = next(
            (
                key
                for key in ("translation_vector", "translation_matrix", "translation")
                if key in reader.fieldnames
            ),
            None,
        )
        if rotation_key is None or translation_key is None:
            raise ValueError("CSV file must contain rotation and translation columns.")
        for row in reader:
            if dataset_filter and row.get("dataset") != dataset_filter:
                continue
            scene_name = row.get("scene")
            image_name = row.get("image")
            rotation_raw = row.get(rotation_key, "")
            translation_raw = row.get(translation_key, "")
            if not scene_name or not image_name or not rotation_raw or not translation_raw:
                continue
            try:
                rotation_vals = parse_float_sequence(rotation_raw, 9)
                translation_vals = parse_float_sequence(translation_raw, 3)
            except ValueError as exc:
                print(
                    f"[WARN] Failed to parse GT pose for {scene_name}/{image_name}: {exc}"
                )
                continue
            rotation_wc = rotation_vals.reshape(3, 3)
            tvec = translation_vals
            center = -rotation_wc.T @ tvec
            camera_counter += 1
            pose = Pose(
                name=image_name,
                rotation_wc=rotation_wc,
                rotation_cw=rotation_wc.T,
                center=center,
                camera_id=camera_counter,
            )
            scene_to_poses.setdefault(scene_name, {})[image_name] = pose
    return scene_to_poses


def rotation_error_deg(R_rel_pred: np.ndarray, R_rel_gt: np.ndarray) -> float:
    delta = R_rel_pred @ R_rel_gt.T
    trace = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(trace)))


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform:
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate similarity transform.")
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = (dst_centered.T @ src_centered) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    rotation = U @ S @ Vt
    var_src = np.sum(src_centered ** 2) / src.shape[0]
    scale = 1.0 if var_src < 1e-12 else np.trace(np.diag(D) @ S) / var_src
    translation = mean_dst - scale * (rotation @ mean_src)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def estimate_similarity_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    max_iters: int,
    threshold: float,
    rng: np.random.Generator,
) -> Tuple[SimilarityTransform, np.ndarray]:
    """Robust similarity estimation from pose centers via RANSAC."""
    print(f"Estimating similarity transform with RANSAC: {max_iters} iters, threshold={threshold}")
    n = src.shape[0]
    if n < 3:
        return IDENTITY_SIMILARITY, np.zeros(n, dtype=bool)
    indices = np.arange(n)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(max_iters):
        sample_idx = rng.choice(indices, size=3, replace=False)
        try:
            sim = estimate_similarity_transform(src[sample_idx], dst[sample_idx])
        except ValueError:
            continue
        aligned = sim.scale * (src @ sim.rotation.T) + sim.translation
        residuals = np.linalg.norm(aligned - dst, axis=1)
        inliers = residuals <= threshold
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_count >= 3:
        sim = estimate_similarity_transform(src[best_inliers], dst[best_inliers])
        return sim, best_inliers
    return IDENTITY_SIMILARITY, np.zeros(n, dtype=bool)


def apply_similarity_to_point(point: np.ndarray, sim: SimilarityTransform) -> np.ndarray:
    return sim.scale * (sim.rotation @ point) + sim.translation


def apply_similarity_to_pose(pose: Pose, sim: SimilarityTransform) -> Pose:
    center = apply_similarity_to_point(pose.center, sim)
    rotation_cw = sim.rotation @ pose.rotation_cw
    rotation_wc = rotation_cw.T
    return Pose(
        name=pose.name,
        rotation_wc=rotation_wc,
        rotation_cw=rotation_cw,
        center=center,
        camera_id=pose.camera_id,
    )


def absolute_pose_errors(
    pred_poses: Mapping[str, Pose], gt_poses: Mapping[str, Pose], required_names: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    rot_errors: Dict[str, float] = {}
    trans_errors: Dict[str, float] = {}
    for name in required_names:
        pred_pose = pred_poses.get(name)
        gt_pose = gt_poses.get(name)
        if pred_pose is None or gt_pose is None:
            rot_errors[name] = math.inf
            trans_errors[name] = math.inf
            continue
        rot_errors[name] = rotation_error_deg(pred_pose.rotation_wc, gt_pose.rotation_wc)
        trans_errors[name] = float(np.linalg.norm(pred_pose.center - gt_pose.center))
    return rot_errors, trans_errors


def compute_auc(errors: Sequence[float], thresholds: Sequence[float]) -> Dict[float, float]:
    arr = np.array(list(errors), dtype=float)
    arr = np.nan_to_num(arr, nan=math.inf, posinf=math.inf, neginf=math.inf)
    aucs: Dict[float, float] = {}
    for thr in thresholds:
        gains = np.maximum(0.0, thr - arr)
        aucs[thr] = float(np.mean(gains) / thr) if len(arr) else 0.0
    return aucs


def error_color(err: float) -> Color:
    if not math.isfinite(err):
        return "rgba(128,128,128,0.8)"
    if err < ROTATION_THRESHOLD_GREEN:
        return "rgb(46, 204, 113)"
    if err > ROTATION_THRESHOLD_RED:
        return "rgb(231, 76, 60)"
    return "rgb(241, 196, 15)"


def camera_calibration_matrix(camera) -> np.ndarray:
    params = camera.params
    model = camera.model.upper()
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    elif model in {"FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    else:
        fx = params[0]
        fy = params[1] if len(params) > 1 else params[0]
        cx = params[2] if len(params) > 2 else camera.width / 2.0
        cy = params[3] if len(params) > 3 else camera.height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def visualize_scene(
    scene: str,
    pred_poses: Mapping[str, Pose],
    pred_cameras: Mapping[int, object],
    gt_poses: Mapping[str, Pose],
    gt_cameras: Mapping[int, object],
    gt_points: Mapping[int, object],
    aligned_pred_points: np.ndarray,
    errors: Mapping[str, float],
    output_path: Path,
    max_points: int,
):
    if not VIZ_AVAILABLE or init_figure is None:
        raise RuntimeError("Visualization requested but plotly/viz_3d is not available.")
    fig = init_figure()
    fig.update_layout(template="plotly_white", paper_bgcolor="white")
    fig.update_scenes(
        bgcolor="white",
        xaxis=dict(showbackground=False, gridcolor="lightgrey", zeroline=False),
        yaxis=dict(showbackground=False, gridcolor="lightgrey", zeroline=False),
        zaxis=dict(showbackground=False, gridcolor="lightgrey", zeroline=False),
    )

    gt_pts = (
        np.array([pt.xyz for pt in gt_points.values()], dtype=float) if gt_points else np.empty((0, 3))
    )
    if gt_pts.size > 0:
        pts = gt_pts
        if max_points and len(pts) > max_points:
            rng = np.random.default_rng(0)
            ids = rng.choice(len(pts), size=max_points, replace=False)
            pts = pts[ids]
        plot_points(fig, pts, color="rgba(52, 152, 219, 0.45)", ps=2, name=f"{scene} GT pts")

    pts = aligned_pred_points
    if pts.size > 0:
        if max_points and len(pts) > max_points:
            rng = np.random.default_rng(1)
            ids = rng.choice(len(pts), size=max_points, replace=False)
            pts = pts[ids]
        plot_points(fig, pts, color="rgba(231, 76, 60, 0.45)", ps=2, name=f"{scene} pred pts")

    for name, pose in gt_poses.items():
        camera = gt_cameras.get(pose.camera_id)
        if camera is None:
            continue
        plot_camera(
            fig,
            pose.rotation_cw,
            pose.center,
            camera_calibration_matrix(camera),
            color="rgb(41, 128, 185)",
            name=f"GT {name}",
        )

    for name, pose in pred_poses.items():
        camera = pred_cameras.get(pose.camera_id)
        if camera is None:
            continue
        err = errors.get(name, math.inf)
        color = error_color(err)
        text = f"{name}<br>err={err:.2f}" if math.isfinite(err) else f"{name}<br>err=inf"
        plot_camera(
            fig,
            pose.rotation_cw,
            pose.center,
            camera_calibration_matrix(camera),
            color=color,
            name=name,
            text=text,
        )

    fig.write_html(str(output_path))


def summarize_errors(errors: Mapping[str, float]) -> Dict[str, float]:
    arr = np.array(list(errors.values()), dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "num_images": len(arr),
        "num_finite": int(finite.size),
        "mean": float(finite.mean()) if finite.size else math.inf,
        "median": float(np.median(finite)) if finite.size else math.inf,
    }


def format_auc(auc: Mapping[float, float]) -> str:
    return ", ".join(f"AUC@{int(k)}={v:.3f}" for k, v in sorted(auc.items()))


def evaluate_scene(
    scene: str,
    gt_dir: Path,
    pred_dir: Path,
    args: argparse.Namespace,
    gt_csv_data: Optional[Dict[str, Dict[str, Pose]]] = None,
    available_images: Optional[Set[str]] = None,
) -> Optional[Dict[str, object]]:
    if not pred_dir.exists():
        print(f"[WARN] Missing prediction for {scene}: {pred_dir}")
        return None

    gt_cameras: Mapping[int, object]
    gt_points: Mapping[int, object]
    real_image_names: List[str]
    if args.gt_poses_csv:
        if gt_csv_data is None:
            raise ValueError("CSV GT data requested but not provided to evaluate_scene().")
        scene_gt = gt_csv_data.get(scene)
        if not scene_gt:
            print(f"[WARN] No GT poses found in CSV for scene {scene}, skipping scene.")
            return None
        real_image_names = sorted(scene_gt.keys())
        if available_images is not None:
            real_image_names = [name for name in real_image_names if name in available_images]
        if not real_image_names:
            print(f"[WARN] No matching images for scene {scene}, skipping scene.")
            return None
        gt_cameras = {}
        gt_points = {}
        gt_poses = {name: scene_gt[name] for name in real_image_names}
    else:
        if not gt_dir.exists():
            print(f"[WARN] Missing GT sparse model for {scene}: {gt_dir}")
            return None
        scene_root = args.gt_root / scene
        real_images_path = scene_root / "images"
        real_image_names = (
            sorted(p.name for p in real_images_path.iterdir() if p.is_file())
            if real_images_path.exists()
            else []
        )
        if not real_image_names:
            print(f"[WARN] No real images found at {real_images_path}, skipping scene.")
            return None
        gt_cameras, gt_poses, gt_points = load_model(gt_dir, ".txt")
        gt_poses = {name: pose for name, pose in gt_poses.items() if name in real_image_names}

    pred_cameras, pred_poses, pred_points = load_model(pred_dir, ".bin")
    pred_poses = {name: pose for name, pose in pred_poses.items() if name in real_image_names}

    shared_names = [name for name in real_image_names if name in pred_poses and name in gt_poses]
    similarity = IDENTITY_SIMILARITY
    inlier_mask = np.zeros(len(shared_names), dtype=bool)
    if len(shared_names) >= 3:
        src = np.stack([pred_poses[name].center for name in shared_names])
        dst = np.stack([gt_poses[name].center for name in shared_names])
        rng = np.random.default_rng(args.align_ransac_seed)
        similarity, mask = estimate_similarity_ransac(
            src, dst, args.align_ransac_iters, args.align_ransac_thresh, rng
        )
        inlier_mask = mask
    else:
        print(f"[WARN] Not enough shared poses for alignment in {scene}, using identity transform.")

    aligned_pred_poses = {
        name: apply_similarity_to_pose(pose, similarity) for name, pose in pred_poses.items()
    }
    rot_errors, trans_errors = absolute_pose_errors(aligned_pred_poses, gt_poses, real_image_names)
    rot_auc = compute_auc(rot_errors.values(), args.auc_thresholds)
    rot_summary = summarize_errors(rot_errors)
    trans_summary = summarize_errors(trans_errors)

    aligned_points = np.empty((0, 3))
    if pred_points:
        pts = np.array([pt.xyz for pt in pred_points.values()], dtype=float)
        aligned_points = similarity.scale * (pts @ similarity.rotation.T) + similarity.translation

    fig_path = None
    if not args.skip_visualization:
        fig_path = args.output_dir / f"{scene}_viz.html"
        visualize_scene(
            scene,
            aligned_pred_poses,
            pred_cameras,
            gt_poses,
            gt_cameras,
            gt_points,
            aligned_points,
            rot_errors,
            fig_path,
            args.max_points,
        )

    inliers = int(np.sum(inlier_mask))
    total = len(shared_names)
    print(
        f"[{scene}] rot {rot_summary['mean']:.2f}°/{rot_summary['median']:.2f}° {format_auc(rot_auc)} "
        f"| trans {trans_summary['mean']:.2f}/{trans_summary['median']:.2f} "
        f"| align inliers {inliers}/{total}"
    )

    return {
        "scene": scene,
        "rotation_errors": rot_errors,
        "translation_errors": trans_errors,
        "rotation_auc": rot_auc,
        "rotation_summary": rot_summary,
        "translation_summary": trans_summary,
        "figure": fig_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SfM reconstructions via absolute pose errors after robust alignment."
    )
    parser.add_argument("--gt-root", type=Path, default=Path("data/test_set"), help="Root directory containing GT scenes.")
    parser.add_argument(
        "--gt-sparse-subdir",
        type=str,
        default="sparse-txt",
        help="Sub-directory under each GT scene that contains COLMAP text files.",
    )
    parser.add_argument(
        "--gt-poses-csv",
        type=Path,
        default=None,
        help="Optional CSV file containing GT rotation / translation entries.",
    )
    parser.add_argument(
        "--gt-csv-dataset",
        type=str,
        default=None,
        help="If set, only use GT CSV rows with this dataset name.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("data/res_vggt_test/run/test_set"),
        help="Root directory containing predicted reconstructions.",
    )
    parser.add_argument(
        "--pred-sparse-subdir",
        type=str,
        default="arb_colmap/colmap/sparse/0",
        help="Sub-directory under each prediction scene with COLMAP binary outputs.",
    )
    parser.add_argument(
        "--pred-model-dir",
        type=Path,
        default=None,
        help="Direct path to a COLMAP sparse model directory (overrides --pred-root layout).",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional list of scene names to evaluate.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Directory containing source images when using CSV GT entries.",
    )
    parser.add_argument("--skip-visualization", action="store_true", help="Disable Plotly visualization export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory to store visualizations and reports.",
    )
    parser.add_argument(
        "--auc-thresholds",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 10.0],
        help="Error thresholds (degrees) for AUC computation.",
    )
    parser.add_argument("--max-points", type=int, default=20000, help="Maximum points to plot per cloud.")
    parser.add_argument(
        "--align-ransac-iters",
        type=int,
        default=2000,
        help="Number of RANSAC iterations for pose alignment.",
    )
    parser.add_argument(
        "--align-ransac-thresh",
        type=float,
        default=1.0,
        help="Inlier threshold (in GT units) for RANSAC pose alignment.",
    )
    parser.add_argument(
        "--align-ransac-seed",
        type=int,
        default=0,
        help="Random seed for RANSAC pose alignment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not VIZ_AVAILABLE and not args.skip_visualization:
        print("[WARN] Plotly/viz_3d not available; disabling visualization.")
        args.skip_visualization = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_csv_data: Optional[Dict[str, Dict[str, Pose]]] = None
    if args.gt_poses_csv is not None:
        if not args.gt_poses_csv.exists():
            raise SystemExit(f"GT CSV file not found: {args.gt_poses_csv}")
        gt_csv_data = load_gt_poses_from_csv(args.gt_poses_csv, args.gt_csv_dataset)
        if not gt_csv_data:
            raise SystemExit("No GT poses loaded from CSV.")

    available_images: Optional[Set[str]] = None
    if args.images_dir is not None and args.images_dir.exists():
        available_images = {p.name for p in args.images_dir.iterdir() if p.is_file()}
    elif args.images_dir is not None and args.gt_poses_csv is not None:
        print(f"[WARN] Images directory {args.images_dir} not found; using CSV names as-is.")

    if args.scenes:
        scenes = args.scenes
    elif gt_csv_data is not None:
        scenes = sorted(gt_csv_data.keys())
    else:
        scenes = discover_scenes(args.pred_root)
    if not scenes:
        raise SystemExit("No scenes found to evaluate.")

    scene_reports: List[Dict[str, object]] = []
    for scene in scenes:
        gt_dir = args.gt_root / scene / args.gt_sparse_subdir
        pred_dir = (
            args.pred_model_dir
            if args.pred_model_dir is not None
            else args.pred_root / scene / args.pred_sparse_subdir
        )
        report = evaluate_scene(scene, gt_dir, pred_dir, args, gt_csv_data, available_images)
        if report:
            scene_reports.append(report)

    if not scene_reports:
        raise SystemExit("No scenes were successfully evaluated.")

    all_rot_errors: List[float] = []
    all_trans_errors: List[float] = []
    for report in scene_reports:
        all_rot_errors.extend(report["rotation_errors"].values())
        all_trans_errors.extend(report["translation_errors"].values())
    overall_rot_auc = compute_auc(all_rot_errors, args.auc_thresholds)
    overall_rot_summary = summarize_errors({str(i): err for i, err in enumerate(all_rot_errors)})
    overall_trans_summary = summarize_errors({str(i): err for i, err in enumerate(all_trans_errors)})

    print(
        f"[ALL] rot {overall_rot_summary['mean']:.2f}°/{overall_rot_summary['median']:.2f}° {format_auc(overall_rot_auc)} "
        f"| trans {overall_trans_summary['mean']:.2f}/{overall_trans_summary['median']:.2f}"
    )


if __name__ == "__main__":
    main()
