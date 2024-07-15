import func

import json
import gecatsim as xc
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import shutil

from PIL import Image


def init(phantom: pathlib.Path, result: pathlib.Path, rows: int, cols: int, views: int) -> xc.CatSim:
    ct = xc.CatSim()
    ct.phantom.filename = str(phantom.resolve() / phantom.stem) + ".json"
    ct.protocol.viewsPerRotation = views
    ct.protocol.viewCount = ct.protocol.viewsPerRotation
    ct.protocol.stopViewId = ct.protocol.viewCount - 1
    ct.scanner.detectorColsPerMod = 1
    ct.scanner.detectorRowsPerMod = rows
    ct.scanner.detectorColCount = cols
    ct.scanner.detectorRowCount = ct.scanner.detectorRowsPerMod
    ct.scanner.detectorColSize = 0.5
    ct.scanner.detectorRowSize = 0.5
    ct.resultsName = str(result.resolve())
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


def gen_files(raw_path: pathlib.Path, json_path: pathlib.Path) -> None:
    size = 10
    vox_size = 0.02
    plane = func.generate_plane(10)
    volume = func.voxelize(plane, 0.02)
    xc.rawwrite(raw_path, volume.astype(np.float32))
    phantom_desc = gen_json(volume, raw_path, vox_size * size)
    with (open(json_path, "w", newline="\n")) as f:
        json.dump(phantom_desc, f, indent=4)


def normalize(data: np.ndarray) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def save_img(data: np.ndarray, output: str) -> None:
    viridis_cmap = plt.get_cmap("viridis")
    norm_data = normalize(data)
    im = Image.fromarray((255 * viridis_cmap(norm_data)).astype(np.uint8))
    im.save(output)


def raw_to_pngs(raw_path: pathlib.Path, pngs_path: pathlib.Path, scans: int, rows: int, cols: int) -> None:
    if pngs_path.exists():
        shutil.rmtree(pngs_path)
    pngs_path.mkdir(parents=True)
    raw = xc.rawread(raw_path.resolve(), (scans, rows, cols), "float")
    for i in range(raw.shape[0]):
        save_img(raw[i], pngs_path / f"{raw_path.name}_{i}.png")


def main() -> None:
    root_path = pathlib.Path().resolve()
    plane_path = root_path / "plane"
    plane_path.mkdir(parents=True, exist_ok=True)
    raw_path = plane_path / "plane.raw"
    json_path = plane_path / "plane.json"
    # gen_files(raw_path, json_path)
    projs_path = plane_path / "projs"
    side = 256
    views = 10
    ct = init(plane_path, projs_path, side, side, views)
    ct.run_all()
    raw_to_pngs(projs_path.with_suffix(".prep"), projs_path, views, side, side)


if __name__ == '__main__':
    main()
