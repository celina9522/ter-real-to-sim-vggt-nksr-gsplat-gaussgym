# Pipeline Notes

This document summarizes the role of each stage in the project pipeline.

## 1. VGGT-Omega Reconstruction

VGGT-Omega is used to replace the original upstream reconstruction stage of
GaussGym. Starting from a monocular video, frames are extracted and processed by
VGGT-Omega to estimate:

- camera intrinsics;
- camera extrinsics;
- depth maps;
- depth confidence maps;
- dense 3D point maps.

These outputs are then used in two branches:

- a geometric branch for NKSR;
- a photorealistic branch for GSplat.

## 2. NKSR Physical Mesh

NKSR reconstructs a continuous triangular mesh from an oriented point cloud.
The point cloud must contain reliable 3D points and normals. In the TER work,
normals are estimated from VGGT-Omega points using a local PCA strategy because
it was more robust than direct cross-product normals computed from depth-map
neighbors.

The resulting mesh is used as the collision geometry in GaussGym.

## 3. GSplat Photorealistic Rendering

GSplat reconstructs the visual appearance of the scene as a set of 3D
Gaussians. It uses the images and camera parameters estimated from VGGT-Omega,
converted into a COLMAP-compatible structure.

GSplat is used only for photorealistic RGB rendering. It does not provide the
physical collision geometry.

## 4. GaussGym Integration

GaussGym combines:

- the NKSR mesh for physics;
- the GSplat scene for RGB observations.

The main integration issue is coordinate alignment. GSplat normalizes the
COLMAP scene during training, so the exported splat is expressed in a normalized
frame. The NKSR mesh and GaussGym camera trajectories must therefore be
transformed into that same normalized frame.

The script `gaussgym/adapt_meshes_to_gsplat_normalization.py` performs this
conversion.

