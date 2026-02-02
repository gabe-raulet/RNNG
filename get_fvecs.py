import sys
import numpy as np
from dataset_io import *

def main(infile, outfile):
    data = []
    with open(infile, "r") as f:
        for line in f.readlines():
            data.append([float(p) for p in line.rstrip().split()])
    points = np.array(data).astype(np.float32)
    write_fvecs(outfile, points)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input.txt> <output.fvecs>\n")
        sys.stderr.flush()
        sys.exit(0)
    sys.exit(main(sys.argv[1], sys.argv[2]))
