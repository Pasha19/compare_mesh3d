import json
import roma


def main() -> None:
    rot_vecs = roma.random_rotvec(1000)
    data = []
    for rot_vec in rot_vecs:
        data.append(rot_vec.tolist())
    with open("axes.json", "w", newline="\n") as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
