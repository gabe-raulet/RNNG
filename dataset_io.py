import sys
import numpy as np
import os

def format_large_number(n, byte=False):
    if byte:
        chunk = 1024
        units = ["", "K", "M", "G", "T", "P"]
    else:
        chunk = 1000
        units = ["", "K", "M", "B", "T", "P"]
    for unit in units:
        if abs(n) < chunk:
            return f"{n:.1f}{unit}"
        n /= chunk
    return f"{n:.2f}E"

def sizeof_fmt(num, suffix="B"):
    """
    Reference: https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
    """
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

def info_u8bin(fname: str) -> tuple[np.int32, np.int32, int, str]:
    with open(fname, "rb") as f:
        filesize = os.path.getsize(fname)
        n, d = np.fromfile(f, count=2, dtype=np.uint32)
    return n, d, filesize, "uint8"

def info_fbin(fname: str) -> tuple[np.int32, np.int32, int, str]:
    with open(fname, "rb") as f:
        filesize = os.path.getsize(fname)
        n, d = np.fromfile(f, count=2, dtype=np.uint32)
    return n, d, filesize, "float32"

def info_fvecs(fname: str) -> tuple[np.int32, np.int32, int, str]:
    with open(fname, "rb") as f:
        filesize = os.path.getsize(fname)
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert filesize % (4*(d+1)) == 0
        n = filesize//(4*(d+1))
    return n, d, filesize, "float32"

def read_u8bin(fname: str, start: int=0, count: int=None) -> np.ndarray:
    with open(fname, "rb") as f:
        n, d = np.fromfile(f, count=2, dtype=np.uint32)
        n = (n - start) if count is None else count
        arr = np.fromfile(f, count=n*d, dtype=np.uint8, offset=start*d)
    return arr.reshape(n,d)

def read_fbin(fname: str, start: int=0, count: int=None) -> np.ndarray:
    with open(fname, "rb") as f:
        n, d = np.fromfile(f, count=2, dtype=np.uint32)
        n = (n - start) if count is None else count
        arr = np.fromfile(f, count=n*d, dtype=np.float32, offset=start*4*d)
    return arr.reshape(n,d)

def read_fvecs(fname: str, start: int=0, count: int=None) -> np.ndarray:
    with open(fname, "rb") as f:
        n = os.path.getsize(fname)
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert n % (4*(d+1)) == 0
        n = (n//(4*(d+1)) - start) if count is None else count
        f.seek(0)
        arr = np.fromfile(f, count=n*(d+1), dtype=np.float32, offset=start*(4*(d+1)))
    return arr.reshape(-1,d+1)[:,1:].copy()

def write_u8bin(fname: str, points: np.ndarray):
    with open(fname, "wb") as f:
        n, d = points.shape
        np.array([n, d], dtype=np.uint32).tofile(f)
        points.tofile(f)

def write_fbin(fname: str, points: np.ndarray):
    with open(fname, "wb") as f:
        n, d = points.shape
        np.array([n, d], dtype=np.uint32).tofile(f)
        points.tofile(f)

def write_fvecs(fname: str, points: np.ndarray):
    with open(fname, "wb") as f:
        n, d = points.shape
        arr = np.insert(points, 0, np.int32(d).view(np.float32), axis=1)
        arr.tofile(f)

"""
Main interface:
"""

def info_file(fname: str) -> tuple[np.int32, np.int32, int, str]:
    """
    Get info on a file of points. Currently only accepts files with
    the extensions *.{u8bin, fbin, fvecs}. Returns a tuple[np.int32, np.int32, int, str]
    where:

        np.int32: number of points
        np.int32: dimension of points
        int:      filesize in bytes
        str:      atom type (e.g. np.float32, np.uint8, etc.)
    """
    if fname.endswith(".u8bin"): return info_u8bin(fname)
    elif fname.endswith(".fbin"): return info_fbin(fname)
    elif fname.endswith(".fvecs"): return info_fvecs(fname)
    else: raise ValueError(f"cannot read file '{fname}': unknown extension")

def read_file(fname: str, start: int=0, count: int=None) -> np.ndarray:
    """
    Read a file of points. Currently only accepts files with the extensions
    *.{u8bin, fbin, fvecs}. Returns a 2-d numpy array whose rows are points.

    `start` provides specifies an offset to read from (in the number of points),
    and defaults to 0, the first point in the file.

    `count` specifies the number of points to read from the file. If it
    is None (the default), then it reads until the end of the file.
    """
    if fname.endswith(".u8bin"): return read_u8bin(fname, start, count)
    elif fname.endswith(".fbin"): return read_fbin(fname, start, count)
    elif fname.endswith(".fvecs"): return read_fvecs(fname, start, count)
    else: raise ValueError(f"cannot read file '{fname}': unknown extension")

def write_file(fname: str, points: np.ndarray):
    """
    Writes a file of points. Currently only writes files with the extensions
    *.{u8bin, fbin, fvecs}.
    """
    if fname.endswith(".u8bin"): write_u8bin(fname, points)
    elif fname.endswith(".fbin"): write_fbin(fname, points)
    elif fname.endswith(".fvecs"): write_fvecs(fname, points)
    else: raise ValueError(f"cannot write file '{fname}': unknown extension")

def main(files):

    sys.stdout.write(f"path\tnum\tdim\ttype\tsize\n")
    for fname in files:
        n, d, filesize, kind = info_file(fname)
        sys.stdout.write(f"{fname}\t{n}\t{d}\t{kind}\t{sizeof_fmt(filesize)}\n")
        sys.stdout.flush()
    return 0

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <files>\n")
        sys.stderr.flush()
        sys.exit(1)
    else:
        sys.exit(main(sys.argv[1:]))
