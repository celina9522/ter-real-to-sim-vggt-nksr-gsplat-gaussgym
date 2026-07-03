#!/usr/bin/env python3
"""Convert GaussGym mesh slices into gsplat's normalized world frame.

This keeps the Gaussian Splat PLY unchanged and instead rewrites the mesh
vertices/camera trajectory so GaussGym sends camera poses in the same frame as
the gsplat checkpoint exported by examples/simple_trainer.py with
normalize_world_space=True.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--scene-dir", type=Path, required=True)
  parser.add_argument("--out-scene-dir", type=Path, required=True)
  parser.add_argument(
    "--gsplat-root",
    type=Path,
    default=Path("/home/comsee/gauss_gym/gsplat"),
    help="Local gsplat repo root containing examples/datasets/colmap.py.",
  )
  parser.add_argument(
    "--data-dir",
    type=Path,
    default=None,
    help="The COLMAP data_dir used by gsplat training. Required unless "
    "--transform-json is provided.",
  )
  parser.add_argument("--data-factor", type=int, default=1)
  parser.add_argument(
    "--transform-json",
    type=Path,
    default=None,
    help="Optional JSON containing gsplat's 4x4 normalization transform. This "
    "is useful on another machine if you already exported parser.transform.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Allow replacing an existing output scene directory.",
  )
  return parser.parse_args()


def load_transform_from_gsplat_parser(
  gsplat_root: Path, data_dir: Path, data_factor: int
) -> np.ndarray:
  sys.path.insert(0, str(gsplat_root))
  from examples.datasets.colmap import Parser

  parser = Parser(
    data_dir=str(data_dir),
    factor=data_factor,
    normalize=True,
    load_exposure=False,
  )
  return np.asarray(parser.transform, dtype=np.float32)


def load_transform_json(path: Path) -> np.ndarray:
  data = json.loads(path.read_text())
  if "transform" in data:
    transform = np.asarray(data["transform"], dtype=np.float32)
  else:
    transform = np.asarray(data, dtype=np.float32)
  if transform.shape == (3, 4):
    transform = np.vstack([transform, np.array([0, 0, 0, 1], dtype=np.float32)])
  if transform.shape != (4, 4):
    raise ValueError(f"Expected a 4x4 or 3x4 transform, got {transform.shape}")
  return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
  return points @ transform[:3, :3].T + transform[:3, 3]


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
  quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
  x, y, z, w = np.moveaxis(quat, -1, 0)
  return np.stack(
    [
      1 - 2 * (y * y + z * z),
      2 * (x * y - z * w),
      2 * (x * z + y * w),
      2 * (x * y + z * w),
      1 - 2 * (x * x + z * z),
      2 * (y * z - x * w),
      2 * (x * z - y * w),
      2 * (y * z + x * w),
      1 - 2 * (x * x + y * y),
    ],
    axis=-1,
  ).reshape(quat.shape[:-1] + (3, 3))


def matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
  m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
  m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
  m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

  qw = np.sqrt(np.clip(1 + m00 + m11 + m22, 0, None)) * 0.5
  qx = np.sqrt(np.clip(1 + m00 - m11 - m22, 0, None)) * 0.5
  qy = np.sqrt(np.clip(1 - m00 + m11 - m22, 0, None)) * 0.5
  qz = np.sqrt(np.clip(1 - m00 - m11 + m22, 0, None)) * 0.5
  qx = np.copysign(qx, m21 - m12)
  qy = np.copysign(qy, m02 - m20)
  qz = np.copysign(qz, m10 - m01)
  quat = np.stack([qx, qy, qz, qw], axis=-1)
  return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def convert_npz(npz_path: Path, out_path: Path, transform: np.ndarray) -> dict:
  data = np.load(npz_path)
  values = {key: data[key] for key in data.files}

  ig_to_orig = values.get("from_ig_rotation", np.eye(3, dtype=np.float32)).astype(
    np.float32
  )
  offset = values.get("offset", np.zeros((1, 3), dtype=np.float32)).astype(np.float32)

  vertices_orig = values["vertices"].astype(np.float32) @ ig_to_orig + offset
  cam_trans_orig = values["cam_trans"].astype(np.float32) @ ig_to_orig + offset

  values["vertices"] = transform_points(vertices_orig, transform).astype(np.float32)
  values["cam_trans"] = transform_points(cam_trans_orig, transform).astype(np.float32)
  values["offset"] = np.zeros((1, 3), dtype=np.float32)
  values["from_ig_rotation"] = np.eye(3, dtype=np.float32)

  if "cam_quat" in values:
    linear = transform[:3, :3]
    scale = float(np.linalg.norm(linear[0]))
    norm_rot = linear / scale
    cam_rot_ig = quat_xyzw_to_matrix(values["cam_quat"].astype(np.float32))
    cam_rot_orig = np.einsum("ij,njk->nik", ig_to_orig, cam_rot_ig)
    cam_rot_norm = np.einsum("ij,njk->nik", norm_rot, cam_rot_orig)
    values["cam_quat"] = matrix_to_quat_xyzw(cam_rot_norm).astype(np.float32)

  np.savez(out_path, **values)
  return {
    "file": npz_path.name,
    "vertices": int(values["vertices"].shape[0]),
    "triangles": int(values["triangles"].shape[0]),
    "cam_poses": int(values["cam_trans"].shape[0]),
  }


def copy_or_replace(src: Path, dst: Path) -> None:
  if dst.exists():
    if dst.is_dir():
      shutil.rmtree(dst)
    else:
      dst.unlink()
  if src.is_dir():
    shutil.copytree(src, dst)
  else:
    shutil.copy2(src, dst)


def main() -> None:
  args = parse_args()
  if args.out_scene_dir.exists():
    if not args.overwrite:
      raise FileExistsError(
        f"{args.out_scene_dir} already exists. Pass --overwrite to replace it."
      )
    shutil.rmtree(args.out_scene_dir)

  if args.transform_json is not None:
    transform = load_transform_json(args.transform_json)
  else:
    if args.data_dir is None:
      raise ValueError("--data-dir is required when --transform-json is not provided")
    transform = load_transform_from_gsplat_parser(
      args.gsplat_root, args.data_dir, args.data_factor
    )

  args.out_scene_dir.mkdir(parents=True)
  for child in args.scene_dir.iterdir():
    if child.name == "meshes":
      continue
    copy_or_replace(child, args.out_scene_dir / child.name)

  out_meshes = args.out_scene_dir / "meshes"
  out_meshes.mkdir()
  summaries = []
  for npz_path in sorted((args.scene_dir / "meshes").glob("*.npz")):
    summaries.append(convert_npz(npz_path, out_meshes / npz_path.name, transform))

  # The splat is already in normalized gsplat space. Keep renderer transform identity.
  splatfacto = args.out_scene_dir / "splatfacto"
  splatfacto.mkdir(exist_ok=True)
  (splatfacto / "dataparser_transforms.json").write_text(
    json.dumps(
      {
        "scale": 1.0,
        "transform": [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
        ],
      },
      indent=2,
    )
    + "\n"
  )

  metadata = {
    "source_scene": str(args.scene_dir),
    "gsplat_root": str(args.gsplat_root),
    "data_dir": str(args.data_dir) if args.data_dir else None,
    "data_factor": args.data_factor,
    "transform": transform.tolist(),
    "meshes": summaries,
  }
  (args.out_scene_dir / "gsplat_normalization_transform.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
  )

  print(f"source_scene: {args.scene_dir}")
  print(f"out_scene: {args.out_scene_dir}")
  print(f"mesh_files: {len(summaries)}")
  print("transform:")
  print(transform)


if __name__ == "__main__":
  main()
