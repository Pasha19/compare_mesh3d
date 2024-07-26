import argparse
import datetime
import json
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


def do_transform(
        obj: vedo.Mesh,
        rotation_axis: tuple[float, float, float],
        angle: float,
        move: tuple[float, float, float],
) -> None:
    lt = (vedo.LinearTransform()
          .rotate(angle, rotation_axis, rad=True)
          .translate(move)
          )
    obj.apply_transform(lt)


def do_voxelization(obj: vedo.Mesh, voxel_size: float, pad: int) -> np.ndarray:
    vol = obj.binarize((1, 0), spacing=(voxel_size, voxel_size, voxel_size))
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
    return volume > 0.5


def bin_volume_to_mesh(volume: np.ndarray, voxel_size: float) -> vedo.Mesh:
    vol = vedo.Volume(volume)
    iso_surf = vol.isosurface(1, flying_edges=True).scale(voxel_size)
    iso_surf.shift(vol.shape * (voxel_size/2))
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


def run(
        rotvec,
        voxel_size: float,
        blur_size: int,
        noise: float,
        result_path: pathlib.Path,
) -> tuple[tuple[float, float, float], float]:
    rotation_axis, angle = axis_angle_from_rotvec(rotvec)

    plane = generate_plane(0.1)
    transformed_plane = plane.copy()
    do_transform(transformed_plane, rotation_axis, angle, (0.0, 0.0, 0.0))

    transformed_plane.write(str(result_path / "transformed_plane.stl"))

    volume = do_voxelization(transformed_plane, voxel_size, pad=8)
    noise_volume = add_noise(volume, blur_size, noise)

    plt.figure()
    plt.hist(noise_volume.ravel(), bins="auto")
    plt.savefig(result_path / "histogram.svg")
    plt.close()

    bin_volume = do_binarization(noise_volume)

    restored_plane = bin_volume_to_mesh(bin_volume, voxel_size)
    restored_plane.write(str(result_path / "restored_plane.stl"))

    restored_plane_icp = restored_plane.copy()
    do_icp(restored_plane_icp, plane)

    restored_plane_icp.write(str(result_path / "restored_plane_icp.stl"))

    dist = calc_distance(plane, restored_plane_icp)

    theta, icp_rot_axis, icp_rot_angle = calc_rot_distance(rotvec, restored_plane, restored_plane_icp)

    with open(result_path / "desc.json", "w", newline="\n") as f:
        desc = {
            "rotation": {
                "axis": rotation_axis,
                "angle": angle,
                "angle_deg": int(np.rad2deg(angle) + 0.5),
            },
            "blur_size": blur_size,
            "noise": noise,
            "voxel_size": voxel_size,
            "distance": dist,
            "rotation_dist": theta,
            "rotation_dist_deg": int(np.rad2deg(theta) + 0.5),
            "icp_rotation": {
                "axis": icp_rot_axis,
                "angle": icp_rot_angle,
                "angle_deg": int(np.rad2deg(icp_rot_angle) + 0.5),
            }
        }
        json.dump(desc, f, indent=4)

    return dist, theta


def hist(data: list[float], file_path: pathlib.Path, title: str) -> None:
    plt.figure()
    plt.title(title)
    plt.hist(data, bins="auto")
    plt.savefig(file_path)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", default=10, type=int)
    parser.add_argument("--vox-size", default=0.01, type=float)
    parser.add_argument("--blur", default=2, type=int)
    parser.add_argument("--noise", default=0.05, type=float)
    parser.add_argument("result_dir", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    root_path: pathlib.Path = args.result_dir
    root_path = root_path.resolve()

    num = args.num
    n = 0
    rot_vecs = roma.random_rotvec(num)
    dists1 = []
    dists2 = []
    angles = []
    for rot_vec in rot_vecs:
        now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        result_path = root_path / f"run_{now}"
        result_path.mkdir(exist_ok=True, parents=True)
        dist, angle = run(rot_vec, args.vox_size, args.blur, args.noise, result_path)
        dists1.append(dist[1])
        dists2.append(dist[2])
        angles.append(int(np.rad2deg(angle) + 0.5))
        n += 1
        print(f"Done {n}/{num}")
    hist(dists1, root_path / "hist_dist1.svg", "plane to icp dist")
    hist(dists2, root_path / "hist_dist2.svg", "icp to plane dist")
    hist(angles, root_path / "hist_angles.svg", "angles hist")
    plane = generate_plane(0.1)
    plane.write(str(root_path / "plane.stl"))


if __name__ == '__main__':
    main()
