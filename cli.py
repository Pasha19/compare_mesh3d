import func

import argparse
import copy
import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import roma
import sqlite3
import trimesh


def create_tables_if_not_exist(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    with connection:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "data" (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                rot_x FLOAT NOT NULL, rot_y FLOAT NOT NULL, rot_z FLOAT NOT NULL,
                e_rot_x FLOAT NOT NULL, e_rot_y FLOAT NOT NULL, e_rot_z FLOAT NOT NULL,
                angle FLOAT NOT NULL, angle_grad INT NOT NULL,
                voxel_size FLOAT NOT NULL,
                UNIQUE (title)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "result" (
                id INTEGER PRIMARY KEY,
                rotation_data_id INTEGER NOT NULL,
                icp BOOLEAN NOT NULL,
                min_dist FLOAT NOT NULL, max_dist FLOAT NOT NULL,
                FOREIGN KEY (rotation_data_id) REFERENCES "data" (id),
                UNIQUE (rotation_data_id, icp)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quantile (
                id INTEGER PRIMARY KEY,
                result_id INTEGER NOT NULL,
                quantile INT NOT NULL,
                "value" FLOAT NOT NULL,
                FOREIGN KEY (result_id) REFERENCES "result" (id),
                UNIQUE (result_id, quantile)
            )
        """)
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_result
            (id, angle, max_dist_no_icp, q90_no_icp, q95_no_icp, q99_no_icp, max_dist_icp, q90_icp, q95_icp, q99_icp)
            AS
            SELECT d.id,d.angle_grad,
                   round(r1.max_dist, 3),
                   round(q190.value, 3),round(q195.value, 3),round(q199.value, 3),
                   round(r2.max_dist, 3),
                   round(q290.value, 3),round(q295.value, 3),round(q299.value, 3)
            FROM data d
            JOIN result r1 on
                d.id = r1.rotation_data_id AND NOT r1.icp
            LEFT JOIN quantile q190 ON
                r1.id = q190.result_id AND q190.quantile = 90
            LEFT JOIN quantile q195 ON
                r1.id = q195.result_id AND q195.quantile = 95
            LEFT JOIN quantile q199 ON
                r1.id = q199.result_id AND q199.quantile = 99
            JOIN result r2 on
                d.id = r2.rotation_data_id AND r2.icp
            LEFT JOIN quantile q290 ON
                r2.id = q290.result_id AND q290.quantile = 90
            LEFT JOIN quantile q295 ON
                r2.id = q295.result_id AND q295.quantile = 95
            LEFT JOIN quantile q299 ON
                r2.id = q299.result_id AND q299.quantile = 99
        """)


def calc_dist(
        connection: sqlite3.Connection,
        last_id: int,
        restored_mesh: trimesh.Trimesh,
        source: trimesh.Trimesh,
        icp: bool,
) -> np.ndarray:
    dist, _ = func.get_distances(restored_mesh, source)
    cursor = connection.cursor()
    with connection:
        cursor.execute(
            """
            INSERT INTO "result"
            (rotation_data_id, icp, min_dist, max_dist)
            VALUES (?, ?, ?, ?)
            """,
            (last_id, icp, np.min(dist), np.max(dist))
        )
        last_id = cursor.lastrowid
    with connection:
        for q in [50, 60, 70, 80, 90, 95, 98, 99]:
            cursor.execute(
                """
                INSERT INTO "quantile" (result_id, quantile, "value")
                VALUES (?, ?, ?)
                """,
                (last_id, q, np.quantile(dist, q / 100))
            )
    return dist


def hist(dist: np.ndarray, e_axis: list[float], angle_grad: int, filename: pathlib.Path, title: str) -> None:
    plt.figure()
    plt.hist(dist, bins=20)
    plt.title(f"{title} axis: ({e_axis[0]:.3f}, {e_axis[1]:.3f}, {e_axis[2]:.3f}) angle: {angle_grad}")
    plt.xlabel("distance")
    plt.ylabel("vertices")
    plt.savefig(filename)
    plt.close()


def process_vec(
        root_path: pathlib.Path,
        connection: sqlite3.Connection,
        source: trimesh.Trimesh,
        vec,
        voxel_size: float
) -> None:
    angle = vec.norm()
    e_axis = vec / angle
    angle = float(angle)
    angle_grad = int(angle / np.pi * 180 + 0.5)
    e_axis = e_axis.tolist()
    cursor = connection.cursor()
    tmp_vec = vec.tolist()
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with connection:
        cursor.execute(
            """
            INSERT INTO "data"
            (title, rot_x, rot_y, rot_z, e_rot_x, e_rot_y, e_rot_z, angle, angle_grad, voxel_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (now, tmp_vec[0], tmp_vec[1], tmp_vec[2], e_axis[0], e_axis[1], e_axis[2], angle, angle_grad, voxel_size)
        )
        last_id = cursor.lastrowid
    dir_path = root_path / f"rot_{now}_{last_id}"
    dir_path.mkdir()
    rotated_mesh, transform = func.rotate(copy.deepcopy(source), e_axis, angle)
    print("  rotated")
    voxel_volume = func.voxelize(rotated_mesh, voxel_size)
    print("  voxelized")
    restored_mesh = func.restore_rotate_and_move_back(voxel_volume.matrix, voxel_size, source, transform)
    restored_mesh_icp = func.icp(source, restored_mesh)
    dist = calc_dist(connection, last_id, restored_mesh, source, False)
    print("  distance calculated")
    dist_icp = calc_dist(connection, last_id, restored_mesh_icp, source, True)
    print("  icp distance calculated")
    rotated_mesh.export(dir_path / "rotated.stl")
    with h5py.File(dir_path / "voxels.h5", "w") as f:
        f.create_dataset("volume", data=voxel_volume.matrix, compression="gzip")
    restored_mesh.export(dir_path / "restored.stl")
    restored_mesh_icp.export(dir_path / "restored_icp.stl")
    hist(dist, e_axis, angle_grad, dir_path / "hist.svg", "no ICP")
    hist(dist_icp, e_axis, angle_grad, dir_path / "hist_icp.svg", "ICP")


def run(root_dir: str, num: int, vox_size: float) -> None:
    root_path = pathlib.Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    source_path = root_path / "source.stl"
    if source_path.exists():
        source = trimesh.load(source_path)
    else:
        source = func.generate_plane(10)
        source.export(source_path)
    db_file = root_path / "data.db"
    with sqlite3.connect(db_file) as connection:
        create_tables_if_not_exist(connection)
        rot_vecs = roma.utils.random_rotvec(num)
        number = 1
        for vec in rot_vecs:
            print(f"start {number}/{num}")
            process_vec(root_path, connection, source, vec, vox_size)
            print(f"done {number}/{num}")
            number += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--vox-size", type=float, default=0.02)
    args = parser.parse_args()
    run(os.path.abspath(args.root), args.num, args.vox_size)


if __name__ == "__main__":
    main()
