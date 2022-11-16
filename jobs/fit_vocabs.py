import os
import argparse
import logging
import json
import tqdm
from collections import defaultdict


logger = logging.getLogger("fit_vocabs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_to_artist_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--num_special_tokens", type=int, default=3)  # pad, bos, unk
    parser.add_argument("--min_frequency_track", type=int, default=0)
    parser.add_argument("--min_frequency_artist", type=int, default=0)
    parser.add_argument("--unk_id", type=int, default=1)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--artists_as_token_types", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    assert args.unk_id < args.num_special_tokens

    track2id = {}
    artist2id = {}

    if args.min_frequency_track > 0 or (args.min_frequency_artist > 0):
        assert args.train_data_path is not None

        logger.info("load track2artist")
        track2artist = {}
        with open(args.track_to_artist_path) as f:
            _ = next(f)
            for line in tqdm.tqdm(f):
                track, artist = line.strip().split(",")
                track2artist[track] = artist

        logger.info("get counts")
        track_counts = defaultdict(int)
        artist_counts = defaultdict(int)
        with open(args.train_data_path) as f:
            for line in tqdm.tqdm(f):
                for track in line.strip().split():
                    track_counts[track] += 1
                    artist_counts[track2artist[track]] += 1

        logger.info("encode tracks and artists")
        for track, artist in track2artist.items():
            # track
            if args.min_frequency_track > 0:
                if track_counts[track] >= args.min_frequency_track:
                    track2id[track] = len(track2id) + args.num_special_tokens
                else:
                    track2id[track] = args.unk_id
            else:
                track2id[track] = len(track2id) + args.num_special_tokens

            # artist
            if artist not in artist2id.keys():
                if args.min_frequency_artist > 0:
                    if artist_counts[artist] >= args.min_frequency_artist:
                        artist2id[artist] = len(artist2id) + args.num_special_tokens
                    else:
                        artist2id[artist] = args.unk_id
                else:
                    artist2id[artist] = len(artist2id) + args.num_special_tokens

        logger.info("saving counts")
        with open(os.path.join(args.output_dir, "track_counts.json"), "w") as f:
            json.dump(track_counts, f)

        with open(os.path.join(args.output_dir, "artist_counts.json"), "w") as f:
            json.dump(artist_counts, f)
    else:
        logger.info("encode tracks and artists")
        print(args.track_to_artist_path)
        with open(args.track_to_artist_path) as f:
            next(f)
            for line in tqdm.tqdm(f):
                track, artist = line.strip().split(",")
                track2id[track] = len(track2id) + args.num_special_tokens
                if artist not in artist2id.keys():
                    artist2id[artist] = len(artist2id) + args.num_special_tokens

    if args.artists_as_token_types == 1:
        # order of packing embeddings in a single matrix:
        # - special tokens
        # - tracks
        # - artists
        num_tracks = len(track2id)
        artist2id = {k: v + num_tracks for k, v in artist2id.items()}

    logger.info("saving encodings")
    with open(os.path.join(args.output_dir, "tracks_vocab.json"), "w") as f:
        json.dump(track2id, f)

    with open(os.path.join(args.output_dir, "artists_vocab.json"), "w") as f:
        json.dump(artist2id, f)
