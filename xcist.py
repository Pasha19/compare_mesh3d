import func

import copy
import datetime
import json
import gecatsim as xc
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import roma
import shutil

from PIL import Image


def init(cfg_path: pathlib.Path, local_cfg_path: pathlib.Path) -> xc.CatSim:
    ct = xc.CatSim(
        cfg_path / "Phantom.cfg",
        cfg_path / "Physics.cfg",
        cfg_path / "Protocol.cfg",
        cfg_path / "Scanner.cfg"
    )
    if local_cfg_path.exists() and local_cfg_path.is_dir():
        cfgs = list(local_cfg_path.glob("*.cfg"))
        if len(cfgs) != 0:
            ct.load_cfg(*cfgs)
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


def normalize(data: np.ndarray) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def save_img(data: np.ndarray, output: pathlib.Path) -> None:
    viridis_cmap = plt.get_cmap("viridis")
    norm_data = normalize(data)
    im = Image.fromarray((255 * viridis_cmap(norm_data)).astype(np.uint8))
    im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    im.save(output)


def raw_to_pngs(raw_path: pathlib.Path, pngs_path: pathlib.Path, scans: int, rows: int, cols: int) -> None:
    if pngs_path.exists():
        shutil.rmtree(pngs_path)
    pngs_path.mkdir(parents=True)
    raw = xc.rawread(raw_path.resolve(), (scans, rows, cols), "float")
    for i in range(raw.shape[0]):
        save_img(raw[i], pngs_path / f"{raw_path.name}_{i}.png")


def process_rot_vec(rot_vec, data_path: pathlib.Path, obj=func.generate_plane(10)) -> None:
    angle = rot_vec.norm()
    e_axis = rot_vec / angle
    angle = float(angle)
    angle_grad = int(angle / np.pi * 180 + 0.5)
    e_axis = e_axis.tolist()
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    rand = random.randint(1, 999)
    rot_vec_path = data_path / f"{now}_{rand}"
    rot_vec_path.mkdir(parents=True)
    rotated_mesh, _ = func.rotate(copy.deepcopy(obj), e_axis, angle)
    rotated_mesh.export(rot_vec_path / "mesh.stl")
    vox_size = 0.02
    volume = func.voxelize(rotated_mesh, vox_size)
    volume = np.transpose(volume,(2, 0, 1))
    volume = volume.copy(order="C")
    volume_raw_file = rot_vec_path / "volume.raw"
    xc.rawwrite(volume_raw_file, volume.astype(np.float32))
    phantom_desc = gen_json(volume, volume_raw_file, vox_size * 10)
    with (open(rot_vec_path / "phantom.json", "w", newline="\n")) as f:
        json.dump(phantom_desc, f, indent=4)
    desc = {
        "axis": rot_vec.tolist(),
        "e_axis": e_axis,
        "angle": angle,
        "angle_grad": angle_grad,
        "shape": volume.shape,
        "voxel_size": vox_size,
    }
    with (open(rot_vec_path / "desc.json", "w", newline="\n")) as f:
        json.dump(desc, f, indent=4)


def gen_data(data_path: pathlib.Path, num: int) -> None:
    rot_vecs = roma.utils.random_rotvec(num)
    n = 0
    for rot_vec in rot_vecs:
        process_rot_vec(rot_vec, data_path)
        n += 1
        print(f"done {n}/{num}")


def gen_proj(root_path: pathlib.Path, obj_path: pathlib.Path) -> None:
    ct = init(root_path / "cfg", obj_path / "cfg")
    ct.phantom.filename = str(obj_path / "phantom.json")
    ct.resultsName = str(obj_path / "projs")
    ct.run_all()
    with open(obj_path / "desc.json", "r+", newline="\n") as f:
        desc = json.load(f)
        desc["views"] = ct.protocol.stopViewId - ct.protocol.startViewId + 1
        desc["rows"] = ct.scanner.detectorRowCount
        desc["cols"] = ct.scanner.detectorColCount
        desc["size"] = {
            "col": ct.scanner.detectorColSize,
            "row": ct.scanner.detectorRowSize,
        }
        f.seek(0)
        f.truncate()
        json.dump(desc, f, indent=4)


def projections(root_path: pathlib.Path, data_path: pathlib.Path) -> None:
    for obj_path in data_path.iterdir():
        if not obj_path.is_dir():
            continue
        gen_proj(root_path, obj_path)


def make_pngs(data_path: pathlib.Path) -> None:
    # import vedo
    for obj_path in data_path.iterdir():
        if not obj_path.is_dir():
            continue
        # with open(obj_path / "phantom.json", "r") as f:
        #     phantom_desc = json.load(f)
        #     raw_to_pngs(
        #         obj_path / "volume.raw",
        #         obj_path / "volume",
        #         phantom_desc["slices"][0],
        #         phantom_desc["rows"][0],
        #         phantom_desc["cols"][0],
        #     )
        #     volume_raw = xc.rawread(
        #         obj_path / "volume.raw",
        #         (phantom_desc["slices"][0], phantom_desc["rows"][0], phantom_desc["cols"][0]),
        #         "float",
        #     )
        #     vedo.show(vedo.Volume(volume_raw), new=True)
        with open(obj_path / "desc.json", "r") as f:
            desc = json.load(f)
            raw_to_pngs(
                obj_path / "projs.prep",
                obj_path / "projs",
                desc["views"],
                desc["rows"],
                desc["cols"],
            )


def main() -> None:
    root_path = pathlib.Path().resolve()
    data_path = root_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    my_path = xc.pyfiles.CommonTools.my_path
    my_path.add_search_path(str(root_path / "spectrum"))
    my_path.add_search_path(str(root_path / "material"))

    gen_data(data_path, 5)
    projections(root_path, data_path)
    make_pngs(data_path)


if __name__ == '__main__':
    main()
