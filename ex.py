import compare

import argparse
import csv
import pathlib
import pandas as pd
import sys
import torch
import typing
import vedo


def run_ex1(
        voxel_size: float,
        angle: float,
        dx: float, dy: float, dz: float,
        e_x: float, e_y: float, e_z: float,
        blur: int, noise: float,
) -> tuple[float, float, float]:
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
    volume = compare.add_noise(volume, blur, noise)
    volume = volume > 0.5
    restored_plane = compare.bin_volume_to_mesh(volume, voxel_size, 0.5)
    dist = compare.calc_distance(transformed_plane, restored_plane)
    return dist


def run_ex2(
        voxel_size: float,
        angle: float,
        dx: float, dy: float, dz: float,
        e_x: float, e_y: float, e_z: float,
        blur: int, noise: float,
) -> tuple[float, float, float]:
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
    volume = compare.add_noise(volume, blur, noise)
    volume = volume > 0.5
    volume_transormed = compare.bin_volume_to_volume(volume, voxel_size)
    volume_transormed.apply_transform(lt.invert())
    restored_plane = volume_transormed.isosurface(0.5, flying_edges=True)
    dist = compare.calc_distance(plane, restored_plane)
    return dist


def run_ex3(
        voxel_size: float,
        angle: float,
        dx: float, dy: float, dz: float,
        e_x: float, e_y: float, e_z: float,
        blur: int, noise: float,
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
    volume = compare.add_noise(volume, blur, noise)
    volume = volume > 0.5
    restored_plane = compare.bin_volume_to_mesh(volume, voxel_size, 0.5)
    restored_plane_icp = restored_plane.clone()
    compare.do_icp(restored_plane_icp, plane)
    dist = compare.calc_distance(plane, restored_plane_icp)
    theta = compare.calc_rot_distance(torch.tensor([e_x, e_y, e_z]) * angle, restored_plane, restored_plane_icp)
    return dist, theta[0]


def get_run_ex(ex: str) -> typing.Callable:
    match ex:
        case "ex1":
            return run_ex1
        case "ex2":
            return run_ex2
        case "ex3":
            return run_ex3
        case _:
            raise Exception("unknown experiment")


def result_to_row(ex: str) -> typing.Callable[..., list]:
    match ex:
        case "ex1" | "ex2":
            return lambda dist, voxel_size: [round(d / voxel_size, 2) for d in dist]
        case "ex3":
            return lambda res, voxel_size: [round(d / voxel_size, 2) for d in res[0]] + [res[1]]
        case _:
            raise Exception("unknown experiment")


def result_header(ex: str) -> list[str]:
    base = ["voxel_size", "angle", "dx", "dy", "dz", "e_x", "e_y", "e_z", "blur", "noise"]
    match ex:
        case "ex1" | "ex2":
            return base + ["max_dist", "d1", "d2"]
        case "ex3":
            return base + ["max_dist", "d1", "d2", "angle"]
        case _:
            raise Exception("unknown experiment")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.add_argument("--ex", required=True, choices=["ex1", "ex2", "ex3"])
    return parser.parse_args()


def main() -> None:
    args = get_args()
    in_tsv_path: pathlib.Path = args.input
    out_tsv_path: pathlib.Path = args.output
    out_tsv_path.parent.mkdir(exist_ok=True, parents=True)
    run_ex = get_run_ex(args.ex)
    get_row = result_to_row(args.ex)
    header = result_header(args.ex)
    in_tsv = pd.read_csv(in_tsv_path, sep="\t", lineterminator="\n")
    num = len(in_tsv)
    n = 0
    with open(out_tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in in_tsv.itertuples(index=True):
            params = [row.voxel_size, row.angle, row.dx, row.dy, row.dz, row.e_x, row.e_y, row.e_z, row.blur, row.noise]
            res = run_ex(*params)
            params[0] = row.voxel_size
            writer.writerow(params + get_row(res, row.voxel_size))
            n += 1
            print(f"done {n} / {num}", file=sys.stderr)
            if n % 10 == 0:
                f.flush()


if __name__ == '__main__':
    main()
