import copy
import numpy as np
import trimesh
import trimesh.voxel.ops as ops
import vedo


def rotate(
        mesh: trimesh.Trimesh,
        e_axis: tuple[float, float, float],
        angle: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    rot = trimesh.transformations.rotation_matrix(angle, e_axis)
    mesh.apply_transform(rot)
    return mesh, rot


def voxelize(mesh: trimesh.Trimesh, vox_size: float) -> np.ndarray:
    size = np.max(mesh.extents)
    radius = int(size/2 / vox_size + 0.5) + 2
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    volume = trimesh.voxel.creation.local_voxelize(mesh, center, vox_size, radius)
    if volume is None:
        raise RuntimeError("Could not voxelize mesh")
    return volume.matrix


def restore_rotate_and_move_back(
        volume: np.ndarray,
        vox_size: float,
        src: trimesh.Trimesh,
        transform: np.array
) -> trimesh.Trimesh:
    mesh = ops.matrix_to_marching_cubes(volume, vox_size)
    mesh.apply_transform(np.linalg.inv(transform))
    sphere_mesh: trimesh.primitives.Sphere = mesh.bounding_sphere
    sphere_src: trimesh.primitives.Sphere = src.bounding_sphere
    mesh.apply_translation(sphere_src.center - sphere_mesh.center)
    # mesh.apply_scale(sphere_src.primitive.radius / sphere_mesh.primitive.radius)
    return mesh


def get_distances(src: trimesh.Trimesh, target: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    result = trimesh.proximity.closest_point(target, src.vertices)
    dist = np.asarray(result[1])
    norm_dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    return dist, norm_dist


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


def helper_create_faces(points: list[list[int]], width: int, height: int) -> list[tuple[int, int, int]]:
    faces = []
    for z in range(height):
        for x in range(width):
            faces.append((points[z][x], points[z+1][x], points[z+1][x+1]))
            faces.append((points[z][x], points[z+1][x+1], points[z][x+1]))
    return faces


def create_xz_plane(width: int, height: int) -> trimesh.Trimesh:
    points = [[(width + 1) * j + i for i in range(width+1)] for j in range(height+1)]
    faces = helper_create_faces(points, width, height)
    vertices_list = [(j, 0, i) for i in range(height + 1) for j in range(width + 1)]
    return trimesh.Trimesh(
        vertices=np.asarray(vertices_list, dtype=float),
        faces=np.asarray(faces, dtype=int),
    )


def create_inner_cylinder(radius: int, height: int, angles: int) -> trimesh.Trimesh:
    vertices = []
    for i in range(height+1):
        for j in range(angles):
            a = 2*np.pi * j / angles
            vertices.append((radius * np.cos(a), radius * np.sin(a), i))
    vertices_ind = []
    for i in range(height+1):
        row = []
        for j in range(angles):
            row.append(i*angles + j)
        row.append(i*angles)
        vertices_ind.append(row)
    faces = helper_create_faces(vertices_ind, angles, height)
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
    )


def generate_plane(size: int) -> trimesh.Trimesh:
    side = 48
    height = 8
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
    vertices = np.asarray([(x, y, 0) for x, y in plane_vertices_list], dtype=float)
    faces = np.asarray(plane_faces, dtype=int)
    plane = trimesh.Trimesh(vertices=vertices, faces=faces)
    plane_bottom = copy.deepcopy(plane)
    plane.apply_translation((0, 0, height))
    plane.invert()
    obj = trimesh.util.concatenate(plane, plane_bottom)
    wall = create_xz_plane(side, height)
    wall.invert()
    obj = trimesh.util.concatenate(obj, wall)
    wall.apply_translation((0, side, 0))
    wall.invert()
    obj = trimesh.util.concatenate(obj, wall)
    wall.apply_translation((0, -side, 0))
    wall.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, (0, 0, 1)))
    obj = trimesh.util.concatenate(obj, wall)
    wall.apply_translation((side, 0, 0))
    wall.invert()
    obj = trimesh.util.concatenate(obj, wall)
    cyl = create_inner_cylinder(radius, height, 32)
    for (xc, yc) in centers:
        cyl.apply_translation((xc, yc, 0))
        obj = trimesh.util.concatenate(obj, cyl)
        cyl.apply_translation((-xc, -yc, 0))
    obj.process()
    obj.apply_translation((-side//2, -side//2, -height//2))
    obj.apply_scale(1/size)
    return obj


def icp(target: trimesh.Trimesh, source: trimesh.Trimesh) -> trimesh.Trimesh:
    target_mesh = vedo.Mesh([target.vertices, target.faces])
    source_mesh = vedo.Mesh([source.vertices, source.faces])
    source_mesh.align_to(target_mesh)
    return trimesh.Trimesh(vertices=source_mesh.vertices, faces=source_mesh.cells)
