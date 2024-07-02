import h5py
import numpy as np
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


def main() -> None:
    mesh = create_mesh(1.0, 2.0)
    mesh.export("test.stl")
    rot = trimesh.transformations.rotation_matrix(np.pi/3, (0, 0, 1))
    mesh.apply_transform(rot)
    voxsize = 0.01
    volume = voxelize(mesh, voxsize)
    with h5py.File("test.h5", "w") as f:
        f.create_dataset("volume", data=volume.matrix, compression="gzip")
    restored = ops.matrix_to_marching_cubes(volume.matrix, voxsize)
    restored.export("restored.stl")


if __name__ == "__main__":
    main()
