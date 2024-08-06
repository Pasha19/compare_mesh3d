import compare

import csv
import pathlib
import pandas as pd
import sys
import torch
import vedo


def run(voxel_size: float,
        angle: float,
        dx: float, dy: float, dz: float,
        e_x: float, e_y: float, e_z: float,
) -> tuple[tuple[float, float, float], float]:
    plane = compare.generate_plane(0.1)
    transformed_plane = plane.copy()
    dx *= voxel_size
    dy *= voxel_size
    dz *= voxel_size
    lt = (vedo.LinearTransform()
          .rotate(angle, (e_x, e_y, e_z), rad=True)
          .translate((dx, dy, dz))
          )
    transformed_plane.apply_transform(lt)
    volume = compare.do_voxelization(transformed_plane, plane.diagonal_size(), voxel_size, pad=4)
    volume = compare.add_noise(volume, 2, 0.05)
    import matplotlib.pyplot as plt
    plt.hist(volume.ravel(), bins="auto", density=True)
    plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
    plt.savefig("fig-hist.png", dpi=300, bbox_inches="tight")
    plt.close()
    volume = volume > 0.5
    restored_plane = compare.bin_volume_to_mesh(volume, voxel_size, 0.5)
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
    std_vox_size = args.std_vox_size

    in_tsv = pd.read_csv(in_tsv_path, sep="\t", lineterminator="\n")
    num = len(in_tsv)

    n = 0
    with open(out_tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["voxel_size", "angle", "dx", "dy", "dz", "e_x", "e_y", "e_z", "max_dist", "d1", "d2", "icp_angle_err"])
        for row in in_tsv.itertuples(index=True):
            vox_size = std_vox_size * row.voxel_size
            params = [vox_size, row.angle, row.dx, row.dy, row.dz, row.e_x, row.e_y, row.e_z]
            dist, theta = run(*params)
            params[0] = row.voxel_size
            writer.writerow(params + [round(d / vox_size, 2) for d in dist] + [theta])
            n += 1
            print(f"done {n} / {num}", file=sys.stderr)


if __name__ == '__main__':
    main()
