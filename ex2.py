import compare

import csv
import pathlib
import pandas as pd
import sys
import vedo


def run(voxel_size: float,
        angle: float,
        dx: float, dy: float, dz: float,
        e_x: float, e_y: float, e_z: float,
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
    volume_transormed = compare.bin_volume_to_volume(volume, voxel_size)
    volume_transormed.apply_transform(lt.invert())
    restored_plane = volume_transormed.isosurface(0.5, flying_edges=True)
    dist = compare.calc_distance(plane, restored_plane)
    return dist


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
        writer.writerow(["voxel_size", "angle", "dx", "dy", "dz", "e_x", "e_y", "e_z", "max_dist", "d1", "d2"])
        for row in in_tsv.itertuples(index=True):
            vox_size = std_vox_size * row.voxel_size
            params = [vox_size, row.angle, row.dx, row.dy, row.dz, row.e_x, row.e_y, row.e_z]
            dist = run(*params)
            params[0] = row.voxel_size
            writer.writerow(params + [round(d / (vox_size), 2) for d in dist])
            n += 1
            print(f"done {n} / {num}", file=sys.stderr)


if __name__ == '__main__':
    main()
