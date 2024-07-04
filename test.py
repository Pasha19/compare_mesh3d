import h5py
import matplotlib.pyplot as plt
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


def show_pcds(pcd, pcd_src, transform: np.ndarray = np.identity(4, dtype=float)) -> None:
    import copy
    pcd_tmp = copy.deepcopy(pcd)
    pcd_src_tmp = copy.deepcopy(pcd_src)
    pcd_tmp.paint_uniform_color([0.1, 0.1, 0.8])
    pcd_src_tmp.paint_uniform_color([0.1, 0.8, 0.1])
    pcd.transform(transform)
    o3d.visualization.draw([pcd_tmp, pcd_src_tmp])


def icp(mesh, src) -> np.ndarray:
    points_num = 100_000
    pcd = mesh.sample_points_poisson_disk(points_num)
    pcd_src = src.sample_points_poisson_disk(points_num)
    transform = global_registration(pcd, pcd_src, 0.05)
    # show_pcds(pcd, pcd_src, transform)
    threshold = 0.012
    # evaluation = o3d.pipelines.registration.evaluate_registration(pcd, pcd_src, threshold, transform)
    # print(evaluation)
    p2p = o3d.pipelines.registration.registration_icp(
        pcd, pcd_src, threshold, transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2_000),
    )
    # print(p2p)
    # show_pcds(pcd, pcd_src, p2p.transformation)
    return p2p.transformation


def preprocess(pcd, voxsize: float) -> tuple[any, any]:
    pcd_ds = pcd.voxel_down_sample(voxsize)
    pcd_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxsize, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxsize, max_nn=100)
    )
    return pcd_ds, pcd_fpfh


def global_registration(src, target, voxsize: float) -> np.ndarray:
    src_ds, src_fpfh = preprocess(src, voxsize)
    target_ds, target_fpfh = preprocess(target, voxsize)
    # show_pcds(src_ds, target_ds)
    distance_threshold = 1.5 * voxsize
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, target_ds, src_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result.transformation


def get_distances(src: trimesh.Trimesh, target: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    result = trimesh.proximity.closest_point(target, src.vertices)
    dist = np.asarray(result[1])
    norm_dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    return dist, norm_dist


def add_colors(mesh: trimesh.Trimesh, norm_dist: np.ndarray):
    res = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    res.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
    res.vertex_colors = o3d.utility.Vector3dVector()
    colors = np.zeros((norm_dist.shape[0], 3), dtype=float)
    colors[:, 0] = 0.1 + norm_dist * 0.8
    colors[:, 1] = 0.9 - norm_dist * 0.8
    colors[:, 2] = 0.1
    res.vertex_colors = o3d.utility.Vector3dVector(colors)
    return res


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
    mesh = o3d.io.read_triangle_mesh(file_mesh)
    src = o3d.io.read_triangle_mesh("mesh_1_2.stl")
    transform = icp(mesh, src)
    mesh = trimesh.load(file_mesh)
    mesh.apply_transform(transform)
    mesh.export("mesh_1_2_icp.stl")


def main_dist() -> None:
    mesh = trimesh.load_mesh("mesh_1_2_icp.stl")
    target = trimesh.load("mesh_1_2.stl")
    dist, norm_dist = get_distances(mesh, target)
    o3d.io.write_triangle_mesh("mesh_1_2_icp.obj", add_colors(mesh, norm_dist))
    print(f"Max distance: {np.max(dist):.3f}")
    plt.hist(dist, bins=20)
    plt.xlabel("distance")
    plt.ylabel("vertices")
    plt.show()


if __name__ == "__main__":
    # main_create()
    # main_show()
    # main_icp()
    main_dist()
