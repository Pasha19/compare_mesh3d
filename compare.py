import matplotlib.pyplot as plt
import numpy as np
import pathlib
import roma
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


def do_voxelization(obj: vedo.Mesh, diag: float, voxel_size: float, pad: int) -> np.ndarray:
    diag_vox = int(diag / voxel_size)
    dim = diag_vox + 2*pad
    side_in_mesh = dim * voxel_size
    vol = obj.binarize((1, 0), spacing=(voxel_size, voxel_size, voxel_size),
                       origin=(-side_in_mesh / 2, -side_in_mesh / 2, -side_in_mesh / 2), dims=(dim, dim, dim))
    volume = vol.tonumpy().astype(np.float32)
    return volume


def add_noise(volume: np.ndarray, blur_size: int, noise: float) -> np.ndarray:
    transform = tio.Compose([
        tio.RandomBlur((blur_size, blur_size)),
        tio.RandomNoise(std=(noise, noise))
    ])
    t = torch.from_numpy(volume).unsqueeze(0)
    t = transform(t)
    return t[0].numpy()


def bin_volume_to_volume(volume: np.ndarray, voxel_size: float) -> vedo.Volume:
    origin = -voxel_size * volume.shape[0] / 2, -voxel_size * volume.shape[1] / 2, -voxel_size * volume.shape[2] / 2
    vol = vedo.Volume(volume, spacing=(voxel_size, voxel_size, voxel_size), origin=origin)
    return vol


def bin_volume_to_mesh(volume: np.ndarray, voxel_size: float, iso_value: float) -> vedo.Mesh:
    vol = bin_volume_to_volume(volume, voxel_size)
    iso_surf = vol.isosurface(iso_value, flying_edges=True)
    return iso_surf


def do_icp(mesh: vedo.Mesh, target: vedo.Mesh) -> None:
    mesh.align_to(target, rigid=True)


def calc_distance(m1: vedo.Mesh, m2: vedo.Mesh) -> tuple[float, float, float]:
    d1 = m1.distance_to(m2).max()
    d2 = m2.distance_to(m1).max()
    return max(d1, d2), d1, d2


def axis_angle_from_rotvec(rotvec) -> tuple[tuple[float, float, float], float]:
    angle = rotvec.norm()
    rotation_axis = (rotvec / angle).tolist()
    return rotation_axis, float(angle)


def calc_rot_distance(
        rotvec,
        restored: vedo.Mesh,
        icp: vedo.Mesh,
) -> tuple[float, tuple[float, float, float], float]:
    rot = roma.rotvec_to_rotmat(rotvec).inverse()
    icp_rot, _ = roma.rigid_points_registration(
        torch.from_numpy(restored.vertices),
        torch.from_numpy(icp.vertices),
    )
    icp_rotvec = roma.rotmat_to_rotvec(icp_rot)
    icp_rot_axis, icp_rot_angle = axis_angle_from_rotvec(icp_rotvec)
    theta = float(roma.rotmat_geodesic_distance(rot, icp_rot))
    return theta, icp_rot_axis, icp_rot_angle


def hist(data: list[float], file_path: pathlib.Path, title: str) -> None:
    plt.figure()
    plt.title(title)
    plt.hist(data, bins="auto")
    plt.savefig(file_path)
    plt.close()
