import os
import argparse
import numpy as np
import torch
import nksr
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruction NKSR batch depuis plusieurs fichiers NPZ"
    )

    parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        required=True,
        help="Un ou plusieurs fichiers .npz contenant points, normals, colors"
    )

    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Dossier racine où sauvegarder les résultats"
    )

    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--detail_level", type=float, default=1.0)
    parser.add_argument("--mise_iter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def process_one(npz_path, args, device, reconstructor):
    name = os.path.basename(os.path.dirname(npz_path))
    output_dir = os.path.join(args.output_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n==============================")
    print("Traitement :", npz_path)
    print("Sortie     :", output_dir)
    print("==============================")

    if not os.path.exists(npz_path):
        print("Fichier introuvable, ignoré :", npz_path)
        return

    data = np.load(npz_path)
    print("Clés disponibles :", data.files)

    points_np = data["points"].astype(np.float32)
    normals_np = data["normals"].astype(np.float32)
    colors_np = data["colors"].astype(np.uint8)

    print("Points  :", points_np.shape)
    print("Normals :", normals_np.shape)
    print("Colors  :", colors_np.shape)

    valid = (
        np.isfinite(points_np).all(axis=1)
        & np.isfinite(normals_np).all(axis=1)
        & (np.linalg.norm(normals_np, axis=1) > 0.5)
    )

    points_np = points_np[valid]
    normals_np = normals_np[valid]
    colors_np = colors_np[valid]

    print("Points valides :", len(points_np))

    norm = np.linalg.norm(normals_np, axis=1, keepdims=True)
    normals_np = normals_np / np.maximum(norm, 1e-8)

    if len(points_np) > args.max_points:
        print(f"Downsampling : {len(points_np)} -> {args.max_points}")
        np.random.seed(args.seed)
        ids = np.random.choice(len(points_np), args.max_points, replace=False)
        points_np = points_np[ids]
        normals_np = normals_np[ids]
        colors_np = colors_np[ids]

    points = torch.from_numpy(points_np).float().to(device)
    normals = torch.from_numpy(normals_np).float().to(device)

    print("Reconstruction...")
    field = reconstructor.reconstruct(
        points,
        normals,
        detail_level=args.detail_level,
    )

    print("Extraction du mesh...")
    result = field.extract_dual_mesh(
        mise_iter=args.mise_iter
    )

    vertices_np = result.v.cpu().numpy()
    faces_np = result.f.cpu().numpy()

    print(f"Vertices : {len(vertices_np)}, Faces : {len(faces_np)}")

    print("Transfert des couleurs sur le mesh...")
    tree = cKDTree(points_np)
    _, idx = tree.query(vertices_np, k=1)
    vertex_colors = colors_np[idx]

    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_colors=vertex_colors,
        process=False,
    )

    output_path = os.path.join(output_dir, "nksr_mesh.ply")
    mesh.export(output_path)

    print("Mesh sauvegardé :", output_path)
    print(f"Vertices : {len(mesh.vertices)}, Faces : {len(mesh.faces)}")

    del points, normals, field, result
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible.")

    device = torch.device("cuda")
    print("GPU :", torch.cuda.get_device_name(0))

    os.makedirs(args.output_dir, exist_ok=True)

    # Reconstructeur instancié une seule fois pour tous les fichiers
    print("Initialisation NKSR...")
    reconstructor = nksr.Reconstructor(device)

    for npz_path in args.inputs:
        try:
            process_one(npz_path, args, device, reconstructor)
        except Exception as e:
            print("Erreur avec le fichier :", npz_path)
            print(e)
            continue

    print("\nTous les traitements sont terminés.")


if __name__ == "__main__":
    main()
