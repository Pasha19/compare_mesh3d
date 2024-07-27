import compare

import csv
import pathlib
import pandas as pd
import sys
import torch


def run(std_vox_size: float,
        voxel_size: float,
        angle: float,
        dx: float, dy: float, dz: float,
        e_x: float, e_y: float, e_z: float,
) -> tuple[tuple[float, float, float], float]:
    plane = compare.generate_plane(0.1)
    transformed_plane = plane.copy()
    voxel_size *= std_vox_size
    dx = dx * voxel_size
    dy = dy * voxel_size
    dz = dz * voxel_size
    compare.do_transform(transformed_plane, (e_x, e_y, e_z), angle, (dx, dy, dz))
    volume = compare.do_voxelization(transformed_plane, plane.diagonal_size(), voxel_size, pad=4)
    restored_plane = compare.bin_volume_to_mesh(volume, voxel_size)
    restored_plane_icp = restored_plane.clone()
    compare.do_icp(restored_plane_icp, plane)
    dist = compare.calc_distance(plane, restored_plane_icp)
    theta = compare.calc_rot_distance(torch.tensor([e_x, e_y, e_z]) * angle, restored_plane, restored_plane_icp)
    return dist, theta[0]


def main() -> None:
    args = compare.get_args()
    in_tsv_path: pathlib.Path = args.input
    out_tsv_path: pathlib.Path = args.output
    out_tsv_path.parent.mkdir(exist_ok=True, parents=True)

    in_tsv = pd.read_csv(in_tsv_path, sep="\t", lineterminator="\n")
    num = len(in_tsv)

    n = 0
    with open(out_tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["voxel_size", "angle", "dx", "dy", "dz", "e_x", "e_y", "e_z", "max_dist", "d1", "d2", "icp_angle_err"])
        std_vox_size = 0.02
        for row in in_tsv.itertuples(index=True):
            params = [row.voxel_size, row.angle, row.dx, row.dy, row.dz, row.e_x, row.e_y, row.e_z]
            dist, theta = run(std_vox_size, *params)
            writer.writerow(params + [round(d / (std_vox_size * row.voxel_size), 1) for d in dist] + [theta])

            n += 1
            if n % 10 == 0:
                f.flush()
            print(f"done {n} / {num}", file=sys.stderr)


if __name__ == '__main__':
    main()
