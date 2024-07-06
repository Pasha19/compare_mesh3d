import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
import trimesh.voxel.ops as ops


def voxelize(mesh: trimesh.Trimesh, voxsize: float) -> trimesh.voxel.VoxelGrid:
    sphere: trimesh.primitives.Sphere = mesh.bounding_sphere
    if not hasattr(sphere.primitive, "radius"):
        raise RuntimeError("Sphere has no radius")
    radius = int(sphere.primitive.radius / voxsize + 0.5)
    volume = trimesh.voxel.creation.local_voxelize(mesh, sphere.center, voxsize, radius)
    if volume is None:
        raise RuntimeError("Could not voxelize mesh")
    return volume


def rotate_back_and_move(mesh: trimesh.Trimesh, src: trimesh.Trimesh, transform: np.array) -> None:
    mesh.apply_transform(np.linalg.inv(transform))
    sphere_mesh: trimesh.primitives.Sphere = mesh.bounding_sphere
    sphere_src: trimesh.primitives.Sphere = src.bounding_sphere
    mesh.apply_translation(sphere_src.center - sphere_mesh.center)
    # mesh.apply_scale(sphere_src.primitive.radius / sphere_mesh.primitive.radius)


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


def add_circle(
        center: tuple[int, int],
        n: int,
        radius: int,
        plane_vertices: list[list[int]],
        plane_vertices_list: list[tuple[(int | float), (int | float)]],
        plane_indices: list[tuple[int, int, int]]
) -> (list[int], int):
    xc, yc = center
    right = plane_vertices[yc][xc+radius]
    top = plane_vertices[yc+radius][xc]
    left = plane_vertices[yc][xc-radius]
    bottom = plane_vertices[yc-radius][xc]
    top_right = plane_vertices[yc+radius][xc+radius]
    top_left = plane_vertices[yc+radius][xc-radius]
    bottom_left = plane_vertices[yc-radius][xc-radius]
    bottom_right = plane_vertices[yc-radius][xc+radius]
    ids = [right]

    def helper(start: int, stop: int, corner: int, last: int) -> None:
        nonlocal n
        for i in range(start, stop):
            angle = np.pi * i / 16
            point = radius * np.cos(angle), radius * np.sin(angle)
            plane_indices.append((ids[-1], n, corner))
            ids.append(n)
            n += 1
            plane_vertices_list.append((point[0] + xc, point[1] + yc))
        plane_indices.append((ids[-1], last, corner))
        ids.append(last)

    helper(1, 8, top_right, top)
    helper(9, 16, top_left, left)
    helper(17, 24, bottom_left, bottom)
    helper(25, 32, bottom_right, right)
    return ids, n


def generate_plane(size: int) -> trimesh.Trimesh:
    side = 48
    radius = 2
    centers = [
        (6,  6),   # 1
        (12, 6),   # 2
        (18, 6),   # 3
        (24, 6),   # 4
        (33, 6),   # 5
        (42, 6),   # 6
        (6,  12),  # 7
        (24, 12),  # 8
        (39, 12),  # 9
        (15, 15),  # 10
        (6,  18),  # 11
        (24, 18),  # 12
        (36, 18),  # 13
        (6,  24),  # 14
        (12, 24),  # 15
        (18, 24),  # 16
        (24, 24),  # 17
        (33, 24),  # 18
        (42, 24),  # 19
        (30, 30),  # 20
        (6,  33),  # 21
        (24, 33),  # 22
        (18, 36),  # 23
        (36, 36),  # 24
        (12, 39),  # 25
        (6,  42),  # 26
        (24, 42),  # 27
        (42, 42),  # 28
    ]
    exclude_vertices = {(x+i, y+j) for i in range(1-radius, radius) for j in range(1-radius, radius) for x, y in centers}
    plane_vertices = [[-1 for _ in range(side+1)] for _ in range(side+1)]
    n = 0
    for y in range(side+1):
        for x in range(side+1):
            if (x, y) not in exclude_vertices:
                plane_vertices[y][x] = n
                n += 1
    plane_faces = []
    for y in range(side):
        for x in range(side):
            if plane_vertices[y][x] == -1 or \
                    plane_vertices[y+1][x] == -1 or \
                    plane_vertices[y][x+1] == -1 or \
                    plane_vertices[y+1][x+1] == -1:
                continue
            plane_faces.append((plane_vertices[y][x], plane_vertices[y+1][x], plane_vertices[y+1][x+1]))
            plane_faces.append((plane_vertices[y][x], plane_vertices[y+1][x+1], plane_vertices[y][x+1]))

    plane_vertices_list = [(j, i) for i in range(side+1) for j in range(side+1) if plane_vertices[i][j] != -1]
    for (xc, yc) in centers:
        ids, n = add_circle((xc, yc), n, radius, plane_vertices, plane_vertices_list, plane_faces)
    vertices = np.asarray([((x - side//2) / size, (y - side//2) / size, 0) for x, y in plane_vertices_list], dtype=float)
    faces = np.asarray(plane_faces, dtype=int)
    plane = trimesh.Trimesh(vertices=vertices, faces=faces)
    return plane


def main_create() -> None:
    mesh = generate_plane(10)
    mesh.export("plane.stl")
    rot = trimesh.transformations.rotation_matrix(np.pi/18, (0, 0, 1))
    mesh.apply_transform(rot)
    voxsize = 0.01
    volume = voxelize(mesh, voxsize)
    with h5py.File("plate_r10.h5", "w") as f:
        f.create_dataset("volume", data=volume.matrix, compression="gzip")
    restored = ops.matrix_to_marching_cubes(volume.matrix, voxsize)
    rotate_back_and_move(restored, mesh, rot)
    restored.export("plate_r10.stl")


def main_show() -> None:
    orig = trimesh.load_mesh("mesh_1_2.stl")
    mesh = trimesh.load("mesh_1_2_r.stl")
    scene = trimesh.Scene()
    scene.add_geometry([orig, mesh])
    scene.show()


def main_icp() -> None:
    file_mesh = "box_1_1_2_r.stl"
    mesh = o3d.io.read_triangle_mesh(file_mesh)
    src = o3d.io.read_triangle_mesh("box_1_1_2.stl")
    transform = icp(mesh, src)
    mesh = trimesh.load(file_mesh)
    mesh.apply_transform(transform)
    mesh.export("box_1_1_2_r_icp.stl")


def main_dist() -> None:
    mesh = trimesh.load_mesh("box_1_1_2_r_icp.stl")
    target = trimesh.load("box_1_1_2.stl")
    dist, norm_dist = get_distances(mesh, target)
    o3d.io.write_triangle_mesh("box_1_1_2_r_icp.obj", add_colors(mesh, norm_dist))
    print(f"Max distance: {np.max(dist):.3f}")
    plt.hist(dist, bins=20)
    plt.xlabel("distance")
    plt.ylabel("vertices")
    plt.savefig("box_1_1_2_r_icp.png")


if __name__ == "__main__":
    main_create()
    # main_show()
    # main_icp()
    # main_dist()
