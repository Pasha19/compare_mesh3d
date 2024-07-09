import func

import argparse
import datetime
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import pathlib
import random
import roma.utils
import trimesh


def gen_rot(mesh: trimesh.Trimesh, rot_vec, data: dict) -> tuple[trimesh.Trimesh, np.ndarray]:
    angle = rot_vec.norm()
    e_axis = rot_vec / angle
    data["rotation"] = {
        "vec": rot_vec.tolist(),
        "e_axis": e_axis.tolist(),
        "angle": float(angle),
        "angle_grad": int(angle / np.pi * 180 + 0.5),
    }
    return func.rotate(mesh, e_axis.tolist(), float(angle))


def gen(root: str, num: int, vox_size: float) -> None:
    root_dir = os.path.abspath(root)
    pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True)
    src_path = os.path.join(root_dir, "source.stl")
    if os.path.exists(src_path):
        source = trimesh.load(src_path)
    else:
        source = func.generate_plane(10)
        source.export(src_path)
    rot_vecs = roma.utils.random_rotvec(num)
    completed = 0
    for rot_vec in rot_vecs:
        now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        rand = random.randint(1, 999)
        rot_dir = os.path.join(root_dir, f"rot_{now}_{rand:03}")
        pathlib.Path(rot_dir).mkdir()
        data = {
            "id": f"rot_{now}_{rand:03}",
        }
        rot_mesh, transform = gen_rot(source, rot_vec, data)
        data["voxel_size"] = vox_size
        voxels = func.voxelize(rot_mesh, vox_size).matrix
        with h5py.File(os.path.join(rot_dir, "voxels.h5"), "w") as f:
            f.create_dataset("volume", data=voxels, compression="gzip")
        restored = func.restore_rotate_and_move_back(voxels, vox_size, source, transform)
        restored.export(os.path.join(rot_dir, "restored.stl"))
        with open(os.path.join(rot_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=4)
        completed += 1
        print(f"rot_{now}_{rand:03} - done {completed}/{num}")


def compare_mesh(source: trimesh.Trimesh, restored: trimesh.Trimesh, data: dict) -> tuple[any, np.ndarray]:
    dist, norm_dist = func.get_distances(restored, source)
    data["result"] = {
        "min_dist": np.min(dist),
        "max_dist": np.max(dist),
    }
    return func.add_colors(restored, norm_dist), dist


def hist_dist(dist: np.ndarray, file_name: str, title: str) -> None:
    plt.figure()
    plt.hist(dist, bins=20)
    plt.title(title)
    plt.xlabel("distance")
    plt.ylabel("vertices")
    plt.savefig(file_name)
    plt.close()


def compare(root: str) -> None:
    root_dir = os.path.abspath(root)
    source = trimesh.load(os.path.join(root_dir, "source.stl"))
    for rot_dir in pathlib.Path(root_dir).iterdir():
        if rot_dir.is_file():
            continue
        data_json = os.path.join(str(rot_dir), "data.json")
        if not os.path.exists(data_json):
            continue
        data = json.load(open(data_json))
        if "result" in data:
            continue
        restored = trimesh.load(os.path.join(str(rot_dir), "restored.stl"))
        colored_mesh, dist = compare_mesh(source, restored, data)
        hist_dist(dist, os.path.join(str(rot_dir), "dist.svg"), "")
        o3d.io.write_triangle_mesh(os.path.join(str(rot_dir), "restored.obj"), colored_mesh)
        with open(os.path.join(rot_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=4)
        print(f"{data['id']} done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", type=str)
    parser.add_argument("root", type=str)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--vox-size", type=float, default=0.02)
    args = parser.parse_args()
    match args.cmd:
        case "gen":
            gen(args.root, args.num, args.vox_size)
        case "cmp":
            compare(args.root)
        case _:
            raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
