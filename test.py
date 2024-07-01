import trimesh


def create_mesh() -> trimesh.Trimesh:
    outer = trimesh.creation.box((0.2, 0.4, 0.5))

    inner = trimesh.creation.box((0.05, 0.05, 0.05))
    inner.apply_translation((0.05, 0.1, 0.2))
    inner.invert()

    return trimesh.util.concatenate(inner, outer)


def main() -> None:
    mesh = create_mesh()
    mesh.show()


if __name__ == "__main__":
    main()
