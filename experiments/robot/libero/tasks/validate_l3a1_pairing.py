"""Validate exact Er/Ec reset-attempt pairing in serialized L3-A1 states."""

import argparse

import h5py


def reset_attempts(path: str, task_description: str) -> list[int]:
    key = task_description.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        group = handle[key]
        return [
            int(group[f"demo_{index}"].attrs["reset_attempt"])
            for index in range(len(group))
        ]


def validate_pairing(er_path: str, ec_path: str, task_description: str) -> list[int]:
    er = reset_attempts(er_path, task_description)
    ec = reset_attempts(ec_path, task_description)
    if not er or er != ec or len(set(er)) != len(er):
        raise ValueError(
            f"Er/Ec reset_attempt mismatch: Er={er}, Ec={ec}; "
            "formal conditions require identical, unique attempt sequences"
        )
    return er


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--task_description", required=True)
    args = parser.parse_args()
    attempts = validate_pairing(args.er, args.ec, args.task_description)
    print(f"PASS_L3A1_PAIRED_RESETS count={len(attempts)} attempts={','.join(map(str, attempts))}")


if __name__ == "__main__":
    main()
