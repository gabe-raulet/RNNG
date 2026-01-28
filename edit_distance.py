import numpy as np
import sys

def random_sequences(n):
    sizes = np.random.randint(30,40,n);
    for i in range(n):
        yield "".join(["ACGT"[j] for j in np.random.randint(0,4,sizes[i])])

def edit_distance(s, t):
    m, n = len(s), len(t)
    v0 = [i for i in range(n+1)]
    v1 = [0 for i in range(n+1)]
    for i in range(m):
        v1[0] = i+1
        for j in range(n):
            delcost = v0[j+1]+1
            inscost = v1[j+0]+1
            subcost = v0[j] if s[i] == t[j] else v0[j]+1
            v1[j+1] = min(delcost, min(inscost, subcost))
        v0, v1 = v1, v0
    return v0[n]

def main(n, radius, outfile):
    sequences = list(random_sequences(n))
    edges = []
    with open(f"{outfile}.seqs", "w") as f:
        for kmer in sequences:
            f.write(kmer + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write(f"Usage: {sys.argv[0]} <n> <radius> <outfile>\n")
        sys.stderr.flush()
        sys.exit(0)
    sys.exit(main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]))
