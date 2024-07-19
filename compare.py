import datetime
import h5py
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import torch
import torchio as tio
import vedo


def generate_plane(scale: float) -> vedo.Mesh:
    side = 48
    height = 8
    radius = 2
    centers = [
        (6, 6),    # 1
        (12, 6),   # 2
        (18, 6),   # 3
        (24, 6),   # 4
        (33, 6),   # 5
        (42, 6),   # 6
        (6, 12),   # 7
        (24, 12),  # 8
        (39, 12),  # 9
        (15, 15),  # 10
        (6, 18),   # 11
        (24, 18),  # 12
        (36, 18),  # 13
        (6, 24),   # 14
        (12, 24),  # 15
        (18, 24),  # 16
        (24, 24),  # 17
        (33, 24),  # 18
        (42, 24),  # 19
        (30, 30),  # 20
        (6, 33),   # 21
        (24, 33),  # 22
        (18, 36),  # 23
        (36, 36),  # 24
        (12, 39),  # 25
        (6, 42),   # 26
        (24, 42),  # 27
        (42, 42),  # 28
    ]
    box = vedo.TessellatedBox(pos=(-side // 2, -side // 2, -height // 2), n=(side, side, height),
                              alpha=1.0).triangulate()
    cyls = [
        vedo
        .Circle(pos=(xc - side//2, yc - side//2, -height//2), r=radius, res=24)
        .extrude(res=height, zshift=height, cap=False)
        for xc, yc in centers]
    cyls = vedo.merge(cyls)
    return box.boolean("intersect", cyls).scale(scale)


def do_transform(
        obj: vedo.Mesh,
        translation: tuple[float, float, float],
        rotation_axis: tuple[float, float, float],
        angle: float,
        scale: float,
) -> None:
    lt = (vedo.LinearTransform()
          .scale(scale)
          .rotate(angle, rotation_axis, rad=True)
          .translate(translation)
          )
    obj.apply_transform(lt)
    # vedo.show(obj, axes=1).close()


def do_voxelization(obj: vedo.Mesh, voxel_size: float, pad: int) -> np.ndarray:
    vol = obj.binarize((1, 0), spacing=(voxel_size, voxel_size, voxel_size))
    # vedo.show(vol, axes=1).close()
    volume = vol.tonumpy().astype(np.float32)
    if pad > 0:
        volume = np.pad(volume, pad)
    return volume


def add_noise(volume: np.ndarray, blur_size: int, noise: float) -> np.ndarray:
    transform = tio.Compose([
        tio.RandomBlur((blur_size, blur_size)),
        tio.RandomNoise(std=(noise, noise))
    ])
    t = torch.from_numpy(volume).unsqueeze(0)
    t = transform(t)
    return t[0].numpy()


def do_binarization(volume: np.ndarray) -> np.ndarray:
    # plt.hist(volume.ravel(), bins="auto")
    # plt.show()
    return volume > 0.5


def bin_volume_to_mesh(volume: np.ndarray, voxel_size: float) -> vedo.Mesh:
    vol = vedo.Volume(volume)
    # vedo.show(vol, axes=1).close()
    return vol.isosurface_discrete(1).scale(voxel_size)


def do_icp(mesh: vedo.Mesh, target: vedo.Mesh) -> None:
    mesh.align_to(target, rigid=True)
    mesh.c("red")
    target.c("blue")
    # vedo.show(mesh, target, axes=1).close()


def calc_distance(m1: vedo.Mesh, m2: vedo.Mesh) -> tuple[float, float, float]:
    d1 = m1.distance_to(m2).max()
    d2 = m2.distance_to(m1).max()
    return max(d1, d2), d1, d2


def run(
        translation: tuple[float, float, float],
        rotation_axis: tuple[float, float, float],
        angle: float,
        blur_size: int,
        noise: float,
        result_path: pathlib.Path,
) -> None:
    scale = 1.0
    voxel_size = 0.01

    plane = generate_plane(0.1)
    transformed_plane = plane.copy()
    do_transform(transformed_plane, translation, rotation_axis, angle, scale)

    transformed_plane.write(str(result_path / "transformed_plane.stl"))

    volume = do_voxelization(transformed_plane, voxel_size, pad=8)
    noise_volume = add_noise(volume, blur_size, noise)

    plt.figure()
    plt.hist(noise_volume.ravel(), bins="auto")
    plt.savefig(result_path / "histogram.png")
    plt.close()

    bin_volume = do_binarization(noise_volume)

    with h5py.File(result_path / "voxels.h5", "w") as f:
        f.create_dataset("volume", data=volume, compression="gzip")
        f.create_dataset("noise_volume", data=noise_volume, compression="gzip")
        f.create_dataset("bin_volume", data=bin_volume, compression="gzip")

    restored_plane = bin_volume_to_mesh(bin_volume, voxel_size)

    restored_plane.write(str(result_path / "restored_plane.stl"))

    do_icp(restored_plane, plane)

    restored_plane.write(str(result_path / "restored_plane_icp.stl"))

    dist = calc_distance(plane, restored_plane)
    print(f"Max distance: {dist[0]:.3f}\nd1: {dist[1]:.3f}\nd2: {dist[2]:.3f}\n")

    with open(result_path / "desc.json", "w", newline="\n") as f:
        desc = {
            "scale": scale,
            "translation": translation,
            "rotation_axis": rotation_axis,
            "angle": angle,
            "blur_size": blur_size,
            "noise": noise,
            "voxel_size": voxel_size,
            "distance": dist,
        }
        json.dump(desc, f, indent=4)


def main() -> None:
    root_path = pathlib.Path().resolve()
    tmp_path = root_path / "tmp"
    data = [
        {  # max=0.036 d1=0.036 d2=0.20
            "translation": (2.0, 2.0, 1.0),
            "rotation_axis": (math.sqrt(3)/3, math.sqrt(3)/3, math.sqrt(3)/3),
            "angle": math.pi / 5,
            "blur_size": 2,
            "noise": 0.05,
        },
        {  # max=0.035 d1=0.035 d2=0.019
            "translation": (10.0, 0.0, 0.0),
            "rotation_axis": (math.sqrt(2)/2, math.sqrt(2)/2, 0),
            "angle": math.pi / 3,
            "blur_size": 2,
            "noise": 0.05,
        },
        {  # max=1.563 d1=0.042 d2=1.563
            "translation": (1.0, 2.0, 3.0),
            "rotation_axis": (0.062280360609292984, 0.9798505306243896, 0.18977344036102295),
            "angle": 1.2554593086242676,
            "blur_size": 3,
            "noise": 0.1,
        },
        {  # max=1.946 d1=0.086 d2=1.946
            "translation": (5.0, 4.0, 2.0),
            "rotation_axis": (-0.5714665055274963, -0.04317374899983406, 0.8194889426231384),
            "angle": 0.6999582648277283,
            "blur_size": 4,
            "noise": 0.1,
        },
    ]

    scale = 1.0
    voxel_size = 0.01

    # import roma
    # rot = roma.random_rotvec()
    # print(rot)
    # angle = rot.norm()
    # rotation_axis = (rot / angle).tolist()
    # angle = float(angle)
    # print(rotation_axis, angle)

    for d in data:
        now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        result_path = tmp_path / f"run_{now}"
        result_path.mkdir(exist_ok=True, parents=True)
        run(d["translation"], d["rotation_axis"], d["angle"], d["blur_size"], d["noise"], result_path)


if __name__ == '__main__':
    main()
