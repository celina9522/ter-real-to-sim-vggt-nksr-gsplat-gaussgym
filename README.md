# Real-to-Sim Pipeline: VGGT-Omega, NKSR, GSplat and GaussGym

This repository contains the scripts developed during a TER project on
photorealistic and physically coherent 3D environment reconstruction from
monocular videos.

The goal of this work is to connect four existing research tools in a single
Real-to-Sim pipeline:

- VGGT-Omega for 3D reconstruction from monocular video frames;
- NKSR for physical mesh reconstruction;
- GSplat for photorealistic Gaussian Splatting rendering;
- GaussGym for robotic simulation with Isaac Gym.

This repository does not redistribute these external projects, their pretrained
models, large datasets, Gaussian Splat files, reconstructed scenes or Isaac Gym
binaries. They must be installed separately from their official sources.

Only the adaptation scripts developed for this project are provided here.

## Project Context

GaussGym is a modular Real-to-Sim framework that separates the representation
used for physics from the representation used for visual rendering. In the
pipeline studied in this project, the physical geometry is represented by a mesh
generated with NKSR, while the photorealistic RGB observations are generated
with GSplat.

The original GaussGym examples mainly rely on already reconstructed scenes, for
example from smartphone scans or Polycam data. In this project, this upstream
reconstruction step is replaced by VGGT-Omega in order to start from a simple
monocular video.

The complete pipeline is:

1. Extract frames from a monocular video.
2. Run VGGT-Omega to estimate camera parameters, depth maps, confidence maps and
   dense 3D points.
3. Convert VGGT-Omega outputs into a COLMAP-like structure for GSplat.
4. Prepare an oriented point cloud for NKSR.
5. Reconstruct a physical mesh with NKSR.
6. Train or export a photorealistic GSplat scene.
7. Align the NKSR mesh and GaussGym camera trajectories with the normalized
   GSplat coordinate frame.
8. Load the adapted scene in GaussGym.

## External Dependencies

Install the main projects from their official repositories or official
installation instructions:

- VGGT-Omega
- NKSR
- GSplat
- GaussGym
- NVIDIA Isaac Gym
- COLMAP or a COLMAP-compatible dataset structure

The scripts in this repository assume that these tools are already installed and
available in your Python environments.

## Repository Structure

```text
vggt_omega/
  extract_omega_from_video_gsplat.py

nksr/
  run_nksr_from_vggt_omega.py

gaussgym/
  adapt_meshes_to_gsplat_normalization.py

docs/
  pipeline.md
```

## Scripts

### 1. VGGT-Omega to GSplat Data Preparation

File:

```text
vggt_omega/extract_omega_from_video_gsplat.py
```

This script runs VGGT-Omega on images extracted from a video and prepares the
outputs for the next stages of the pipeline.

It is used to:

- load a VGGT-Omega checkpoint;
- extract frames from a monocular video;
- estimate camera intrinsics and extrinsics;
- predict depth maps and confidence maps;
- reconstruct dense 3D points from depth;
- check the camera-pose convention;
- filter unreliable points;
- prepare data that can be used for GSplat training and NKSR preprocessing.

In the TER report, this corresponds to the adaptation of VGGT-Omega outputs for
the geometric branch and the photorealistic branch of the pipeline.

### 2. NKSR Mesh Reconstruction

File:

```text
nksr/run_nksr_from_vggt_omega.py
```

This script reconstructs a physical mesh from the point cloud and normals
prepared from VGGT-Omega outputs.

It is used to:

- load one or several `.npz` files containing points, normals and colors;
- filter invalid points and normals;
- optionally downsample the point cloud;
- run NKSR reconstruction on GPU;
- extract a triangular mesh;
- transfer colors from the source point cloud to the mesh vertices;
- export the result as `nksr_mesh.ply`.

This mesh is intended to be used as the physical collision geometry in GaussGym.

### 3. GaussGym / GSplat Coordinate Alignment

File:

```text
gaussgym/adapt_meshes_to_gsplat_normalization.py
```

This is the key integration script for GaussGym.

The main issue solved by this script is the coordinate-frame mismatch between:

- the NKSR mesh, initially expressed in the original reconstruction frame;
- the GaussGym camera trajectories;
- the exported GSplat scene, expressed in GSplat's normalized COLMAP frame.

The chosen solution is not to modify the exported Gaussian Splat. Instead, the
script transforms the GaussGym scene files so that the mesh vertices and camera
trajectories are expressed in the same normalized frame as GSplat.

The script:

- reads the GSplat/COLMAP parser normalization transform;
- transforms GaussGym mesh vertices;
- transforms camera positions (`cam_trans`);
- transforms camera orientations (`cam_quat`);
- resets `offset` to zero;
- resets `from_ig_rotation` to identity;
- writes an identity `splatfacto/dataparser_transforms.json`, because the splat
  is already normalized.

This step is required before loading the NKSR mesh and GSplat rendering together
inside GaussGym.

## Example Usage

The exact paths depend on your local installation. The following commands show
the intended order of use.

### Run VGGT-Omega and prepare GSplat/NKSR data

Edit the paths inside:

```text
vggt_omega/extract_omega_from_video_gsplat.py
```

Then run it inside your VGGT-Omega environment:

```bash
python vggt_omega/extract_omega_from_video_gsplat.py
```

The script produces VGGT-Omega outputs such as camera parameters, depth maps,
confidence maps, point clouds and converted data for the next stages.

### Run NKSR

Inside your NKSR environment:

```bash
python nksr/run_nksr_from_vggt_omega.py \
  --inputs /path/to/nksr_input_pca.npz \
  --output_dir /path/to/nksr_outputs \
  --max_points 300000 \
  --detail_level 1.0 \
  --mise_iter 1
```

The output mesh is saved as:

```text
nksr_mesh.ply
```

### Align the GaussGym Scene with GSplat

Inside the GaussGym environment:

```bash
python gaussgym/adapt_meshes_to_gsplat_normalization.py \
  --scene-dir /path/to/original_gaussgym_scene \
  --out-scene-dir /path/to/aligned_gaussgym_scene \
  --gsplat-root /path/to/gsplat \
  --data-dir /path/to/colmap_data \
  --overwrite
```

The output scene can then be loaded in GaussGym as a local scene.

Example:

```bash
gauss_play \
  --runner.load_run=<RUN_NAME> \
  --terrain.scenes.max_num_scenes=1 \
  --terrain.scenes.iphone_data.repo_id=local:/path/to/aligned_gaussgym_scene \
  --terrain.scenes.iphone_data.scene='' \
  --env.force_renderer=True
```

## Notes

This repository is intended as a lightweight code release for reproducing the
adaptation work, not as a full standalone implementation of VGGT-Omega, NKSR,
GSplat or GaussGym.

Large files such as checkpoints, videos, reconstructed meshes, Gaussian Splat
PLY files and simulation logs should not be committed to this repository.

