import h5py
import numpy as np
import open3d as o3d
import trimesh
import trimesh.voxel.ops as ops


def create_mesh(width: float, height: float) -> trimesh.Trimesh:
    box = trimesh.creation.box((width, width, height))
    cyl = trimesh.creation.cylinder(radius=width/3, height=height/1.5)
    cyl.invert()
    sphere = trimesh.creation.icosphere(radius=min(width/10, height/10))
    sphere.apply_translation((0.4*width, 0.4*width, 0.4*height))
    sphere.invert()
    return trimesh.util.concatenate((box, cyl, sphere))


def voxelize(mesh: trimesh.Trimesh, voxsize: float) -> trimesh.voxel.VoxelGrid:
    sphere: trimesh.primitives.Sphere = mesh.bounding_sphere
    if not hasattr(sphere.primitive, "radius"):
        raise RuntimeError("Sphere has no radius")
    radius = int(sphere.primitive.radius / voxsize + 0.5)
    volume = trimesh.voxel.creation.local_voxelize(mesh, sphere.center, voxsize, radius)
    if volume is None:
        raise RuntimeError("Could not voxelize mesh")
    return volume


def scale_and_move(mesh: trimesh.Trimesh, src: trimesh.Trimesh) -> None:
    sphere_mesh: trimesh.primitives.Sphere = mesh.bounding_sphere
    sphere_src: trimesh.primitives.Sphere = src.bounding_sphere
    mesh.apply_translation(sphere_src.center - sphere_mesh.center)
    mesh.apply_scale(sphere_src.primitive.radius / sphere_mesh.primitive.radius)


def icp(file_mesh: str, file_src: str) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(file_mesh)
    src = o3d.io.read_triangle_mesh(file_src)
    points_num = 100_000
    pcd = mesh.sample_points_poisson_disk(points_num)
    pcd_src = src.sample_points_poisson_disk(points_num)
    pcd.paint_uniform_color([0.1, 0.1, 0.8])
    pcd_src.paint_uniform_color([0.1, 0.8, 0.1])
    # o3d.visualization.draw([pcd, pcd_src])
    transform = np.identity(4, dtype=float)
    threshold = 0.03
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd, pcd_src, threshold, transform)
    # print(evaluation)
    p2p = o3d.pipelines.registration.registration_icp(
        pcd, pcd_src, threshold, transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1_000),
    )
    # print(p2p)
    pcd.transform(p2p.transformation)
    # o3d.visualization.draw([pcd, pcd_src])
    return p2p.transformation


def main_create() -> None:
    mesh = create_mesh(1.0, 2.0)
    # mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    mesh.export("cube_1.stl")
    rot = trimesh.transformations.rotation_matrix(np.pi/3, (0, 0, 1))
    rot @= trimesh.transformations.rotation_matrix(np.pi/6, (0, 1, 0))
    mesh.apply_transform(rot)
    voxsize = 0.01
    volume = voxelize(mesh, voxsize)
    with h5py.File("cube_1.h5", "w") as f:
        f.create_dataset("volume", data=volume.matrix, compression="gzip")
    restored = ops.matrix_to_marching_cubes(volume.matrix, voxsize)
    scale_and_move(restored, mesh)
    restored.export("cube_1_r.stl")


def main_show() -> None:
    orig = trimesh.load_mesh("mesh_1_2.stl")
    mesh = trimesh.load("mesh_1_2_r.stl")
    scene = trimesh.Scene()
    scene.add_geometry([orig, mesh])
    scene.show()


def main_icp() -> None:
    file_mesh = "mesh_1_2_r.stl"
    file_src = "mesh_1_2.stl"
    transform = icp(file_mesh, file_src)
    mesh = trimesh.load(file_mesh)
    mesh.apply_transform(transform)
    mesh.export("mesh_1_2_icp.stl")


if __name__ == "__main__":
    # main_create()
    # main_show()
    main_icp()
