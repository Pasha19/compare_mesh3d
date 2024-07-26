import compare

import csv
import datetime
import numpy as np
import pathlib
import roma


def run(
        rotvec,
        voxel_size: float,
        blur_size: int,
        noise: float,
) -> tuple[tuple[float, float, float], float]:
    rotation_axis, angle = compare.axis_angle_from_rotvec(rotvec)

    plane = compare.generate_plane(0.1)
    transformed_plane = plane.copy()
    compare.do_transform(transformed_plane, rotation_axis, angle, (0.0, 0.0, 0.0))

    volume = compare.do_voxelization(transformed_plane, voxel_size, pad=8)
    noise_volume = compare.add_noise(volume, blur_size, noise)
    bin_volume = compare.do_binarization(noise_volume)

    restored_plane = compare.bin_volume_to_mesh(bin_volume, voxel_size)

    restored_plane_icp = restored_plane.copy()
    compare.do_icp(restored_plane_icp, plane)

    dist = compare.calc_distance(plane, restored_plane_icp)
    theta, icp_rot_axis, icp_rot_angle = compare.calc_rot_distance(rotvec, restored_plane, restored_plane_icp)

    return dist, theta


def main() -> None:
    args = compare.get_args()
    root_path: pathlib.Path = args.result_dir
    root_path = root_path.resolve()

    num = args.num
    n = 0
    rot_vecs = roma.random_rotvec(num)
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(root_path / f"ex2_{now}.tsv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", lineterminator="\n")
        writer.writerow(["blur", "noise", "e_x", "e_y", "e_z", "angle", "angle_deg", "dist", "icp_angle_err"])
        for rot_vec in rot_vecs:
            dist, theta = run(rot_vec, args.vox_size, args.blur, args.noise)
            axis, angle = compare.axis_angle_from_rotvec(rot_vec)
            writer.writerow([args.blur, args.noise, *axis, angle, int(np.rad2deg(angle) + 0.5), dist, theta])
            n += 1
            print(f"Done {n}/{num}")


if __name__ == '__main__':
    main()
