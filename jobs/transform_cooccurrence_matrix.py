import argparse
import numpy as np
from scipy.sparse import csr_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    print("load co-occurrence matrix")
    with np.load(args.input_path) as d:
        X = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(d["shape"][0], d["shape"][1]))
    print(repr(X))

    print("get global counts")
    d = 1 / X.diagonal() ** 0.5

    print("normalize")
    X = X.multiply(d[None, :]).multiply(d[:, None]).tocsr()

    print("save")
    np.savez(args.output_path, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)
