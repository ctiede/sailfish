from pickle import load, dump
from sys import path, argv
from pathlib import Path
import argparse

path.append(str(Path(__file__).parent.parent))


def scale_by_n(infile, n):
    with open(infile, "rb") as inf:
        c = load(inf)

    c["solution"] = c["solution"].repeat(n, axis=0).repeat(n, axis=1)
    c["driver"] = c["driver"]._replace(resolution=c["driver"].resolution * n)
    c["mesh"] = c["mesh"]._replace(ni=c["mesh"].ni * n, nj=c["mesh"].nj * n)
    outfile = infile.replace(".pk", ".u{:d}x.pk".format(n))

    with open(outfile, "wb") as f:
        dump(c, f)

    print(f"write {infile} scaled by {n}x to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument("--factor", '-n', type=int, default=2)
    args = parser.parse_args()
    for file in args.checkpoints:
        scale_by_n(file, args.factor)
