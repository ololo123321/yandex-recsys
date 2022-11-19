import argparse
import tqdm
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path")
    parser.add_argument("--matrix_path")
    parser.add_argument("--vec_vocab_path")
    parser.add_argument("--topk_tracks", type=int, default=1000)
    args = parser.parse_args()

    print("loading train data")
    train_data = []
    track_counts = defaultdict(int)
    num_likes = 0
    with open(args.train_data_path) as f:
        for line in tqdm.tqdm(f):
            tracks = line.strip().split()
            train_data.append(tracks)
            num_likes += len(tracks)
            for track in tracks:
                track_counts[track] += 1
    print("num training examples:", len(train_data))
    print("num likes:", num_likes)

    top_tracks = [x[0] for x in sorted(track_counts.items(), key=lambda x: -x[1])[:args.topk_tracks]]
    vec = CountVectorizer(analyzer=lambda x: x, vocabulary=top_tracks)

    print("build co-occurrences matrix")
    X = vec.transform(train_data)  # [N, V]
    print(repr(X))
    print("build co-occurrences matrix")
    X = X.T.dot(X)  # [V, V]
    print(repr(X))
    print("save co-occurrences matrix")
    np.savez(args.matrix_path, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)

    print("save vec vocab path")
    with open(args.vec_vocab_path, "w") as f:
        for track in top_tracks:
            f.write(track + "\n")
