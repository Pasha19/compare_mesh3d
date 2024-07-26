import compare

import csv
import datetime
import h5py
import json
import numpy as np
import pathlib
import torch


def run(
        rotvec,
        voxel_size: float,
        blur_size: int,
        noise: float,
        result_path: pathlib.Path | None = None,
        num: int | None = None
) -> tuple[float, float, float]:
    rotation_axis, angle = compare.axis_angle_from_rotvec(rotvec)

    plane = compare.generate_plane(0.1)
    transformed_plane = plane.copy()
    dx = voxel_size * 0.1
    compare.do_transform(transformed_plane, rotation_axis, angle, (dx, dx, dx))

    volume = compare.do_voxelization(transformed_plane, voxel_size, pad=8)
    noise_volume = compare.add_noise(volume, blur_size, noise)
    bin_volume = compare.do_binarization(noise_volume)

    restored_plane = compare.bin_volume_to_mesh(bin_volume, voxel_size)

    if result_path is None:
        dist = compare.calc_distance(transformed_plane, restored_plane)
    else:
        plane.write(str(result_path / f"{num}_M_G.stl"))
        transformed_plane.write(str(result_path / f"{num}_M_D.stl"))
        restored_plane.write(str(result_path / f"{num}_M_I.stl"))
        with h5py.File(result_path / f"{num}_volume.h5", "w") as f:
            f.create_dataset("volume", data=volume, compression="gzip")
            f.create_dataset("noise_volume", data=noise_volume, compression="gzip")
            f.create_dataset("bin_volume", data=bin_volume, compression="gzip")
        dist = 0.0, 0.0, 0.0
        dist = compare.calc_distance(transformed_plane, restored_plane)

    print(dist)
    return dist


def do_all(args, root_path: pathlib.Path) -> None:
    num = args.num
    with open(args.axes.resolve(), "r") as f:
        rot_vecs_list = json.load(f)
    rot_vecs = torch.tensor(rot_vecs_list)
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(root_path / f"ex1_{now}.tsv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", lineterminator="\n")
        writer.writerow(["blur", "noise", "e_x", "e_y", "e_z", "angle", "angle_deg", "max dist", "d1", "d2"])
        for n in range(num):
            rot_vec = rot_vecs[n]
            for blur in args.blur:
                for noise in args.noise:
                    dist = run(rot_vec, args.vox_size, blur, noise)
                    axis, angle = compare.axis_angle_from_rotvec(rot_vec)
                    writer.writerow([blur, noise, *axis, angle, int(np.rad2deg(angle) + 0.5), *dist])
            print(f"Done {n + 1}/{num}")


def do_one(args, root_path: pathlib.Path) -> None:
    tsv_path = args.tsv.resolve()
    with open(tsv_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        stop = args.row
        data = None
        for row in reader:
            if stop == 0:
                data = row
                break
            stop -= 1
    blur = int(data[0])
    noise = float(data[1])
    axis = float(data[2]), float(data[3]), float(data[4])
    angle = float(data[5])
    vox_size = args.vox_size
    rot_vec = torch.tensor(axis) * angle
    tsv_name = tsv_path.stem
    result_path = root_path / tsv_name
    result_path.mkdir(exist_ok=True, parents=True)
    run(rot_vec, vox_size, blur, noise, result_path, args.row)


def main() -> None:
    args = compare.get_args()
    root_path: pathlib.Path = args.result_dir
    root_path = root_path.resolve()
    root_path.mkdir(exist_ok=True, parents=True)

    if args.axes is not None:
        do_all(args, root_path)
    elif args.tsv is not None:
        do_one(args, root_path)


if __name__ == '__main__':
    main()
