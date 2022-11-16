import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path")
    parser.add_argument("--predictions_path")
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    preds = []
    with open(args.predictions_path) as f:
        for line in f:
            preds.append(line.strip().split())

    accuracy = 0.0
    mrr = 0.0
    n = 0
    with open(args.test_data_path) as f:
        for line, tracks in zip(f, preds):
            gold = line.strip().split()[-1]
            for i in range(min(args.k, len(tracks))):
                if tracks[i] == gold:
                    mrr += 1 / (1 + i)
                    if i == 0:
                        accuracy += 1
                    break
            n += 1
    accuracy /= n
    mrr /= n
    print("mrr:", mrr)
    print("accuracy:", accuracy)
    print("evidence:", n)
