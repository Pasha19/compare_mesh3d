import compare

import argparse
import csv
import json
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import shutil
import sys
import torch
import vedo

from PIL import Image


def init_common(result: pathlib.Path, rows: int, cols: int, views: int, det_pixel_size: float) -> xc.CatSim:
    ct = xc.CatSim()
    ct.protocol.viewsPerRotation = views
    ct.protocol.viewCount = ct.protocol.viewsPerRotation
    ct.protocol.stopViewId = ct.protocol.viewCount - 1
    ct.scanner.detectorColsPerMod = 1
    ct.scanner.detectorRowsPerMod = rows
    ct.scanner.detectorColCount = cols
    ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod
    ct.scanner.detectorColSize = det_pixel_size
    ct.scanner.detectorRowSize = det_pixel_size
    ct.scanner.sid = 950
    ct.scanner.sdd = 1000
    ct.resultsName = str(result.resolve())
    return ct


def init_proj(
        json_path: pathlib.Path,
        result: pathlib.Path,
        rows: int, cols: int,
        views: int,
        voxel_size: float,
) -> xc.CatSim:
    ct = init_common(result, rows, cols, views, voxel_size * 10)
    ct.phantom.filename = str(json_path)
    return ct


def init_rec(
        result: pathlib.Path,
        rows: int,
        cols: int,
        views: int,
        voxel_size: float,
        rec_size: int,
) -> xc.CatSim:
    ct = init_common(result, rows, cols, views, voxel_size * 10)
    ct.protocol.viewsPerRotation = views
    ct.do_Recon = 1
    ct.recon.sliceCount = rec_size
    ct.recon.sliceThickness = voxel_size * 10
    ct.recon.imageSize = rec_size
    ct.recon.fov = rec_size * voxel_size * 10
    return ct


def gen_json(volume: np.ndarray, raw_path: pathlib.Path, vox_size: float) -> dict:
    result = {
        "n_materials": 1,
        "mat_name": ["Al"],
        "volumefractionmap_filename": [raw_path.name],
        "volumefractionmap_datatype": ["float"],
        "cols": [volume.shape[0]],
        "rows": [volume.shape[1]],
        "slices": [volume.shape[2]],
        "x_size": [vox_size],
        "y_size": [vox_size],
        "z_size": [vox_size],
        "x_offset": [volume.shape[0]/2],
        "y_offset": [volume.shape[1]/2],
        "z_offset": [volume.shape[2]/2],
    }
    return result


def save_img(data: np.ndarray, output: pathlib.Path) -> None:
    viridis_cmap = plt.get_cmap("viridis")
    im = Image.fromarray((255 * viridis_cmap(data)).astype(np.uint8))
    im.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(output)


def raw_to_pngs(raw_path: pathlib.Path, rows: int, cols: int, num: int) -> None:
    pngs_path = raw_path.parent / f"{raw_path.stem}_pngs"
    if pngs_path.exists():
        shutil.rmtree(pngs_path)
    pngs_path.mkdir(parents=True)
    raw = xc.rawread(raw_path.resolve(), (num, rows, cols), "float")
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    for i in range(raw.shape[0]):
        save_img(raw[i], pngs_path / f"{raw_path.name}_{i}.png")


def run(
        voxel_size: float,
        angle: float,
        dx: float, dy: float, dz: float,
        e_x: float, e_y: float, e_z: float,
        working_dir: pathlib.Path,
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
    # volume = np.transpose(volume, (2, 0, 1))
    # working_dir.mkdir(exist_ok=True, parents=True)

    # volume_raw_path = working_dir / "volume.raw"
    # json_path = working_dir / "volume.json"
    # projs_path = working_dir / "projs"

    # xc.rawwrite(volume_raw_path, volume.copy(order="C"))
    # phantom_desc = gen_json(volume, volume_raw_path, voxel_size * 10)
    # with (open(json_path, "w", newline="\n")) as f:
    #     json.dump(phantom_desc, f, indent=4)
    # raw_to_pngs(volume_raw_path, phantom_desc["rows"][0], phantom_desc["cols"][0], phantom_desc["slices"][0])

    # det_rows, det_cols = 384, 384
    # projs_count = 180
    # ct = init_proj(json_path, projs_path, det_rows, det_rows, projs_count, voxel_size)
    # ct.run_all()
    # raw_to_pngs(projs_path.with_suffix(".prep"), det_rows, det_rows, projs_count)

    rec_size = max(volume.shape)
    # ct = init_rec(projs_path, det_rows, det_cols, projs_count, voxel_size, rec_size)
    # recon.recon(ct)
    # raw_rec_path = working_dir / f"{projs_path.name}_{rec_size}x{rec_size}x{rec_size}.raw"
    recon_path = working_dir / f"recon.raw"
    # recon_path.unlink(missing_ok=True)
    # raw_rec_path.rename(recon_path)
    # raw_to_pngs(recon_path, rec_size, rec_size, rec_size)

    raw = xc.rawread(recon_path.resolve(), (rec_size, rec_size, rec_size), "float")
    raw = np.transpose(raw, (1, 2, 0))
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(raw.ravel(), bins="auto")
    plt.savefig(str(working_dir / "hist.png"), dpi=300, bbox_inches="tight")
    plt.close()
    raw = (raw - raw.min()) / (raw.max() - raw.min())

    plt.figure(figsize=(6.4, 4.8))
    plt.hist(raw.ravel(), bins="auto")
    plt.savefig(str(working_dir / "hist_norm.png"), dpi=300, bbox_inches="tight")
    plt.close()

    volume_recon = compare.bin_volume_to_volume(raw > 0.5, voxel_size)
    # vedo.show(volume_recon).close()
    recon_plane = volume_recon.isosurface(0.5, flying_edges=True)
    recon_plane.write(str(working_dir / f"recon.stl"))
    # vedo.show(transformed_plane.c("red5").alpha(0.5), recon_plane.c("green5").alpha(0.5)).close()
    recon_plane_icp = recon_plane.clone()
    compare.do_icp(recon_plane_icp, plane)
    recon_plane_icp.write(str(working_dir / f"recon_icp.stl"))
    dist = compare.calc_distance(plane, recon_plane_icp)
    theta = compare.calc_rot_distance(torch.tensor([e_x, e_y, e_z]) * angle, recon_plane, recon_plane_icp)
    return dist, theta[0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    in_tsv_path = args.input
    out_tsv_path = args.output
    working_dir = out_tsv_path.parent / f"{out_tsv_path.stem}_working_dir"
    working_dir.mkdir(exist_ok=True, parents=True)
    in_tsv = pd.read_csv(in_tsv_path, sep="\t", lineterminator="\n")
    num = len(in_tsv)
    n = 0
    header = ["voxel_size", "angle", "dx", "dy", "dz", "e_x", "e_y", "e_z", "max_dist", "d1", "d2", "icp_angle_err"]
    with open(out_tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in in_tsv.itertuples(index=True):
            params = [row.voxel_size, row.angle, row.dx, row.dy, row.dz, row.e_x, row.e_y, row.e_z]
            res = run(*params, working_dir / str(n+1))
            writer.writerow(params + [d / row.voxel_size for d in res[0]] + [res[1]])
            n += 1
            print(f"done {n} / {num}", file=sys.stderr)
            if n % 10 == 0:
                f.flush()


if __name__ == "__main__":
    main()
