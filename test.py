import h5py
import trimesh


def create_mesh(side: float, void: float) -> trimesh.Trimesh:
    outer = trimesh.creation.box((side, side, side))
    inner = trimesh.creation.box((void, void, void))
    inner.invert()
    return trimesh.util.concatenate(inner, outer)


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
    mesh = create_mesh(1.0, 0.1)
    volume = voxelize(mesh, 0.01)
    with h5py.File("test.h5", "w") as f:
        f.create_dataset("volume", data=volume.matrix, compression="gzip")


if __name__ == "__main__":
    main()
