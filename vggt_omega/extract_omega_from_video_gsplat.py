import argparse
import gc
import glob
import os
import shutil
from datetime import datetime

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from visual_util import apply_sky_mask, depth_edge, predictions_to_glb
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def run_model(target_dir: str, model: VGGTOmega, image_resolution: int) -> dict:
    print(f"Processing images from {target_dir}")

    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if len(image_names) == 0:
        raise RuntimeError("No images found. Please upload images or a video first.")

    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    torch.cuda.empty_cache()
    return predictions_np


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]

    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def verify_camera_convention(predictions, frame_step=1, sample_size=2000, seed=0):
    rng = np.random.default_rng(seed)

    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]
    world_points = predictions["world_points_from_depth"]
    num_frames, height, width = world_points.shape[:3]

    in_front_ratios = []
    in_bounds_ratios = []

    for i in range(num_frames - frame_step):
        j = i + frame_step

        pts = world_points[i].reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]

        if len(pts) == 0:
            continue

        if len(pts) > sample_size:
            idx = rng.choice(len(pts), sample_size, replace=False)
            pts = pts[idx]

        rot_j = extrinsic[j, :3, :3]
        trans_j = extrinsic[j, :3, 3]

        cam_pts = pts @ rot_j.T + trans_j

        z = cam_pts[:, 2]
        in_front = z > 0
        in_front_ratios.append(np.mean(in_front))

        fx, fy = intrinsic[j, 0, 0], intrinsic[j, 1, 1]
        cx, cy = intrinsic[j, 0, 2], intrinsic[j, 1, 2]

        with np.errstate(divide="ignore", invalid="ignore"):
            u = (cam_pts[:, 0] / z) * fx + cx
            v = (cam_pts[:, 1] / z) * fy + cy

        in_bounds = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        in_bounds_ratios.append(np.mean(in_bounds[in_front]) if np.any(in_front) else 0.0)

    mean_in_front = float(np.mean(in_front_ratios)) if in_front_ratios else 0.0
    mean_in_bounds = float(np.mean(in_bounds_ratios)) if in_bounds_ratios else 0.0

    print("--- Test croisé inter-frames ---")
    print(f"Points devant la caméra suivante : {mean_in_front * 100:.1f}%")
    print(f"Parmi eux, dans l'image : {mean_in_bounds * 100:.1f}%")

    if mean_in_front < 0.5:
        print("ATTENTION : convention extrinsic peut-être inversée.")
    else:
        print("Convention extrinsic world->camera probablement correcte.")

    return mean_in_front, mean_in_bounds


def file_path(file_data) -> str:
    if isinstance(file_data, dict):
        if "name" in file_data:
            return file_data["name"]
        if "path" in file_data:
            return file_data["path"]
        if file_data.get("video") is not None:
            return file_path(file_data["video"])

    if hasattr(file_data, "name"):
        return file_data.name

    return str(file_data)


def handle_uploads(input_video, input_images, video_sample_fps=1.0):
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join("demo_outputs", f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []

    if input_images is not None:
        for item in input_images:
            src_path = file_path(item)
            dst_path = os.path.join(target_dir_images, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        video_path = file_path(input_video)
        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        video_sample_fps = max(float(video_sample_fps), 0.1)
        frame_interval = max(int(round((fps if fps and fps > 0 else 1) / video_sample_fps)), 1)

        frame_idx = 0
        saved_idx = 0

        while True:
            ok, frame = video.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{saved_idx:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                saved_idx += 1

            frame_idx += 1

        video.release()

    image_paths = sorted(image_paths)
    return target_dir, image_paths


def glb_path(target_dir, conf_thres, mask_black_bg, mask_white_bg, show_cam, mask_sky, max_points_k):
    return os.path.join(
        target_dir,
        f"scene_conf{conf_thres}_black{mask_black_bg}_white{mask_white_bg}_"
        f"cam{show_cam}_sky{mask_sky}_max{int(max_points_k)}k.glb",
    )


def build_point_mask(
    predictions,
    target_dir,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    mask_sky,
    filter_depth_edges=True,
    depth_edge_rtol=0.03,
):
    points = predictions["world_points_from_depth"].reshape(-1, 3)
    conf = predictions["depth_conf"]

    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0

    if mask_sky:
        conf = apply_sky_mask(conf, target_dir)

    conf = conf.reshape(-1)

    colors = predictions["images"]
    if colors.ndim == 4:
        colors = np.transpose(colors, (0, 2, 3, 1))
    colors = colors.reshape(-1, 3)

    if colors.max() <= 1.5:
        colors = colors * 255.0

    colors = colors.clip(0, 255).astype(np.uint8)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf)

    conf_thres = max(2.0, float(conf_thres))
    if conf_thres > 0 and np.any(valid):
        conf_threshold = np.percentile(conf[valid], conf_thres)
        valid &= conf >= conf_threshold

    valid &= conf > 1e-5

    if mask_black_bg:
        valid &= colors.sum(axis=1) >= 16

    if mask_white_bg:
        valid &= ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))

    return valid


def export_colmap_for_gsplat(
    target_dir,
    predictions,
    conf_thres=50.0,
    mask_black_bg=False,
    mask_white_bg=False,
    mask_sky=False,
    filter_depth_edges=True,
    depth_edge_rtol=0.03,
    max_points=200000,
):
    colmap_dir = os.path.join(target_dir, "colmap")
    images_out = os.path.join(colmap_dir, "images")
    sparse_out = os.path.join(colmap_dir, "sparse", "0")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(sparse_out, exist_ok=True)

    src_images = sorted(glob.glob(os.path.join(target_dir, "images", "*.png")))
    if len(src_images) == 0:
        raise RuntimeError("Aucune image trouvée pour l'export COLMAP.")

    for img_path in src_images:
        shutil.copy(img_path, os.path.join(images_out, os.path.basename(img_path)))

    intrinsic = predictions["intrinsic"]
    extrinsic = predictions["extrinsic"]

    img0 = cv2.imread(src_images[0])
    h, w = img0.shape[:2]

    fx = intrinsic[0, 0, 0]
    fy = intrinsic[0, 1, 1]
    cx = intrinsic[0, 0, 2]
    cy = intrinsic[0, 1, 2]

    cameras_txt = os.path.join(sparse_out, "cameras.txt")
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    images_txt = os.path.join(sparse_out, "images.txt")
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, img_path in enumerate(src_images):
            rot = extrinsic[i, :3, :3]
            trans = extrinsic[i, :3, 3]

            quat = R.from_matrix(rot).as_quat()
            qx, qy, qz, qw = quat

            name = os.path.basename(img_path)

            f.write(
                f"{i + 1} "
                f"{qw} {qx} {qy} {qz} "
                f"{trans[0]} {trans[1]} {trans[2]} "
                f"1 {name}\n"
            )
            f.write("\n")

    points = predictions["world_points_from_depth"].reshape(-1, 3)

    colors = predictions["images"]
    if colors.ndim == 4:
        colors = np.transpose(colors, (0, 2, 3, 1))
    colors = colors.reshape(-1, 3)

    if colors.max() <= 1.5:
        colors = colors * 255.0

    colors = colors.clip(0, 255).astype(np.uint8)

    valid = build_point_mask(
        predictions,
        target_dir,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        mask_sky,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
    )

    points = points[valid]
    colors = colors[valid]

    if points.shape[0] > max_points:
        step = max(points.shape[0] // max_points, 1)
        points = points[::step]
        colors = colors[::step]

    points3d_txt = os.path.join(sparse_out, "points3D.txt")
    with open(points3d_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

        for i, (p, c) in enumerate(zip(points, colors)):
            f.write(
                f"{i + 1} "
                f"{p[0]} {p[1]} {p[2]} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"1.0\n"
            )

    points3d_ply = os.path.join(sparse_out, "points3D.ply")
    trimesh.PointCloud(vertices=points, colors=colors).export(file_obj=points3d_ply)

    print("Export COLMAP / GSplat sauvegardé ici :", colmap_dir)


def save_colored_ply(points_raw, images_raw, output_path):
    points = points_raw.reshape(-1, 3)

    if images_raw.shape[1] == 3:
        images_raw = images_raw.transpose(0, 2, 3, 1)

    colors = images_raw.reshape(-1, 3)

    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    colors = colors[mask]

    print("Points valides (omega_cloud) :", len(points))

    with open(output_path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode())

        vertex_data = np.zeros(
            len(points),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("red", np.uint8),
                ("green", np.uint8),
                ("blue", np.uint8),
            ],
        )

        vertex_data["x"] = points[:, 0]
        vertex_data["y"] = points[:, 1]
        vertex_data["z"] = points[:, 2]
        vertex_data["red"] = colors[:, 0]
        vertex_data["green"] = colors[:, 1]
        vertex_data["blue"] = colors[:, 2]

        f.write(vertex_data.tobytes())

    print(f"Créé : {output_path}")


def prepare_nksr_pca(
    points_raw,
    conf_raw,
    images_raw,
    extrinsic,
    output_npz,
    output_ply,
    max_points,
    conf_percentile,
    k_neighbors,
    seed,
):
    if images_raw.shape[1] == 3:
        images_raw = images_raw.transpose(0, 2, 3, 1)

    F, H, W, _ = points_raw.shape

    R_cam = extrinsic[:, :3, :3]
    t_cam = extrinsic[:, :3, 3]

    cam_centers = -np.einsum("fij,fj->fi", np.transpose(R_cam, (0, 2, 1)), t_cam)

    frame_ids = np.repeat(np.arange(F), H * W)

    points = points_raw.reshape(-1, 3)
    conf = conf_raw.reshape(-1)
    colors = images_raw.reshape(-1, 3)

    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    thr = np.percentile(conf[mask], conf_percentile)
    mask &= conf >= thr

    points = points[mask]
    colors = colors[mask]
    conf = conf[mask]
    frame_ids = frame_ids[mask]

    np.random.seed(seed)
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        conf = conf[idx]
        frame_ids = frame_ids[idx]

    points = points.astype(np.float32)
    conf = conf.astype(np.float32)
    colors = colors.astype(np.uint8)

    print("Points utilisés (PCA) :", len(points))

    print("Construction KDTree...")
    tree = cKDTree(points)

    print("Recherche voisins...")
    _, indices = tree.query(points, k=k_neighbors)

    print("Calcul PCA locale...")
    neigh = points[indices]
    center = neigh.mean(axis=1, keepdims=True)
    X = neigh - center
    C = np.einsum("nki,nkj->nij", X, X)

    eigvals, eigvecs = np.linalg.eigh(C)
    min_idx = np.argmin(eigvals, axis=1)
    normals = eigvecs[np.arange(len(points)), :, min_idx]

    print("Orientation des normales...")
    point_cam_centers = cam_centers[frame_ids].astype(np.float32)
    view_dirs = point_cam_centers - points

    flip = np.sum(normals * view_dirs, axis=1) < 0
    normals[flip] *= -1

    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals = normals.astype(np.float32)

    np.savez(
        output_npz,
        points=points,
        normals=normals,
        colors=colors,
        confidence=conf,
        frame_ids=frame_ids.astype(np.int32),
    )
    print(f"Créé : {output_npz}")

    with open(output_ply, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode())

        vertex_data = np.zeros(
            len(points),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("nx", np.float32),
                ("ny", np.float32),
                ("nz", np.float32),
                ("red", np.uint8),
                ("green", np.uint8),
                ("blue", np.uint8),
            ],
        )

        vertex_data["x"] = points[:, 0]
        vertex_data["y"] = points[:, 1]
        vertex_data["z"] = points[:, 2]
        vertex_data["nx"] = normals[:, 0]
        vertex_data["ny"] = normals[:, 1]
        vertex_data["nz"] = normals[:, 2]
        vertex_data["red"] = colors[:, 0]
        vertex_data["green"] = colors[:, 1]
        vertex_data["blue"] = colors[:, 2]

        f.write(vertex_data.tobytes())

    print(f"Créé : {output_ply}")


def process_video(
    model,
    input_video,
    image_resolution,
    video_sample_fps,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    mask_sky,
    max_points_k,
    pca_max_points,
    pca_conf_percentile,
    pca_k_neighbors,
    pca_seed,
):
    target_dir, image_paths = handle_uploads(
        input_video=input_video,
        input_images=None,
        video_sample_fps=video_sample_fps,
    )

    print(f"--- Traitement de {input_video} ---")
    print("Images extraites :", len(image_paths))
    print("Target dir :", target_dir)

    predictions = run_model(
        target_dir=target_dir,
        model=model,
        image_resolution=image_resolution,
    )

    verify_camera_convention(predictions)

    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)
    print("Sauvegardé :", prediction_save_path)

    raw_outputs = os.path.join(target_dir, "raw_outputs")
    os.makedirs(raw_outputs, exist_ok=True)

    for key, value in predictions.items():
        np.save(os.path.join(raw_outputs, key + ".npy"), value)
        print(key, value.shape)

    glbfile = glb_path(
        target_dir,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        max_points_k,
    )

    scene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        max_points=int(max_points_k * 1000),
    )

    scene.export(file_obj=glbfile)
    print("GLB sauvegardé :", glbfile)

    ply_path = os.path.join(target_dir, "omega_cloud.ply")
    save_colored_ply(
        points_raw=predictions["world_points_from_depth"],
        images_raw=predictions["images"],
        output_path=ply_path,
    )

    prepare_nksr_pca(
        points_raw=predictions["world_points_from_depth"],
        conf_raw=predictions["depth_conf"],
        images_raw=predictions["images"],
        extrinsic=predictions["extrinsic"],
        output_npz=os.path.join(target_dir, "nksr_input_pca.npz"),
        output_ply=os.path.join(target_dir, "nksr_input_pca.ply"),
        max_points=pca_max_points,
        conf_percentile=pca_conf_percentile,
        k_neighbors=pca_k_neighbors,
        seed=pca_seed,
    )

    export_colmap_for_gsplat(
        target_dir,
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        max_points=int(max_points_k * 1000),
    )

    print("\n[Succès] Tous les livrables VGGT + NKSR + GSplat sont prêts.")
    print("Dossier :", target_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-video",
        "--input_video",
        dest="input_video",
        nargs="+",
        default=["home.mp4"],
        help="Une ou plusieurs vidéos à traiter",
    )

    parser.add_argument("--checkpoint", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    parser.add_argument("--image-resolution", "--image_resolution", dest="image_resolution", type=int, default=512)
    parser.add_argument("--video-sample-fps", "--video_sample_fps", dest="video_sample_fps", type=float, default=1.0)

    parser.add_argument("--conf-thres", "--conf_thres", dest="conf_thres", type=float, default=50.0)
    parser.add_argument("--mask-black-bg", "--mask_black_bg", dest="mask_black_bg", action="store_true", default=False)
    parser.add_argument("--mask-white-bg", "--mask_white_bg", dest="mask_white_bg", action="store_true", default=False)
    parser.add_argument("--show-cam", "--show_cam", dest="show_cam", action="store_true", default=True)
    parser.add_argument("--hide-cam", "--hide_cam", dest="hide_cam", action="store_true", default=False)
    parser.add_argument("--mask-sky", "--mask_sky", dest="mask_sky", action="store_true", default=False)
    parser.add_argument("--max-points-k", "--max_points_k", dest="max_points_k", type=float, default=1000)

    parser.add_argument("--pca-max-points", "--pca_max_points", dest="pca_max_points", type=int, default=800000)
    parser.add_argument("--pca-conf-percentile", "--pca_conf_percentile", dest="pca_conf_percentile", type=int, default=10)
    parser.add_argument("--pca-k-neighbors", "--pca_k_neighbors", dest="pca_k_neighbors", type=int, default=30)
    parser.add_argument("--pca-seed", "--pca_seed", dest="pca_seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    show_cam = args.show_cam and not args.hide_cam

    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint)

    for input_video in args.input_video:
        process_video(
            model=model,
            input_video=input_video,
            image_resolution=args.image_resolution,
            video_sample_fps=args.video_sample_fps,
            conf_thres=args.conf_thres,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            show_cam=show_cam,
            mask_sky=args.mask_sky,
            max_points_k=args.max_points_k,
            pca_max_points=args.pca_max_points,
            pca_conf_percentile=args.pca_conf_percentile,
            pca_k_neighbors=args.pca_k_neighbors,
            pca_seed=args.pca_seed,
        )


if __name__ == "__main__":
    main()
