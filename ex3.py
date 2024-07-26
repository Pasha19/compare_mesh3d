import compare

import csv
import datetime
import json
import numpy as np
import pathlib
import torch


def run(
        rotvec,
        voxel_size: float,
        blur_size: int,
        noise: float,
) -> tuple[tuple[float, float, float], float]:
    rotation_axis, angle = compare.axis_angle_from_rotvec(rotvec)

    plane = compare.generate_plane(0.1)
    transformed_plane = plane.copy()
    dx = 0.1 * voxel_size
    compare.do_transform(transformed_plane, rotation_axis, angle, (dx, dx, dx))

    volume = compare.do_voxelization(transformed_plane, voxel_size, pad=8)
    noise_volume = compare.add_noise(volume, blur_size, noise)
    bin_volume = compare.do_binarization(noise_volume)

    restored_plane = compare.bin_volume_to_mesh(bin_volume, voxel_size)

    restored_plane_icp = restored_plane.clone()
    compare.do_icp(restored_plane_icp, plane)

    dist = compare.calc_distance(plane, restored_plane_icp)
    theta = compare.calc_rot_distance(rotvec, restored_plane, restored_plane_icp)

    return dist, theta[0]


def main() -> None:
    args = compare.get_args()
    root_path: pathlib.Path = args.result_dir
    root_path = root_path.resolve()
    root_path.mkdir(exist_ok=True, parents=True)

    num = args.num
    with open(args.axes.resolve(), "r") as f:
        rot_vecs_list = json.load(f)
    rot_vecs = torch.tensor(rot_vecs_list)
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(root_path / f"ex3_{now}.tsv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", lineterminator="\n")
        writer.writerow(["blur", "noise", "e_x", "e_y", "e_z", "angle", "angle_deg", "dist", "d1", "d2", "icp_angle_err"])
        for n in range(num):
            for blur in args.blur:
                for noise in args.noise:
                    rot_vec = rot_vecs[n]
                    dist, theta = run(rot_vec, args.vox_size, blur, noise)
                    axis, angle = compare.axis_angle_from_rotvec(rot_vec)
                    writer.writerow([blur, noise, *axis, angle, int(np.rad2deg(angle) + 0.5), *dist, theta])
            print(f"Done {n+1}/{num}")


if __name__ == '__main__':
    main()
