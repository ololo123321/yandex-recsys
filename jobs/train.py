import os
import sys
import logging
import json
from typing import List
import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import torch.distributed as dist
import transformers
from transformers.integrations import MLflowCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import get_track_id_to_artist_id


def get_logger(training_args):
    logger = logging.getLogger("training")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.local_rank in [-1, 0]:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    args = hydra.utils.instantiate(cfg.training_args)
    logger = get_logger(args)
    logger.info(f"world size: {args.world_size}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"hydra outputs dir: {os.getcwd()}")
    if args.local_rank in [-1, 0]:
        print(OmegaConf.to_yaml(cfg))

    with open(cfg.track_vocab_path) as f:
        track2id = json.load(f)

    with open(cfg.artist_vocab_path) as f:
        artist2id = json.load(f)

    def load_data(path) -> List[List[int]]:
        data = []
        pbar = tqdm.tqdm(position=args.local_rank, leave=True, desc=f"rank {args.local_rank}")
        with open(path) as f:
            for i, line in enumerate(f):
                if i == cfg.limit:
                    break
                if (args.world_size > 1) and (i % args.world_size != args.local_rank):
                    continue
                tracks = line.strip().split()
                data.append(tracks)
                pbar.update(1)
        pbar.close()
        return data

    logger.info("loading train data...")
    train_data = load_data(cfg.train_data_path)

    if args.world_size > 1:
        logger.info("syncing number of training examples...")
        n = torch.tensor(len(train_data)).to(args.device)
        ns = [torch.empty_like(n) for _ in range(args.world_size)]
        dist.all_gather(ns, n)
        logger.info(f"num training examples per process: {[x.item() for x in ns]}")
        dist.all_reduce(n, op=dist.ReduceOp.MIN)
        n = n.item()
        logger.info(f"synced number of training examples: {n}")
        train_data = train_data[:n]

    logger.info("loading test data...")
    valid_data = load_data(cfg.valid_data_path)

    logger.info("loading track->artist map...")
    track2artist = {}
    with open(cfg.track_to_artist_path) as f:
        _ = next(f)
        for line in tqdm.tqdm(f, disable=(args.local_rank > 0)):
            track, artist = line.strip().split(",")
            track2artist[track] = artist

    ds_train = hydra.utils.instantiate(cfg.training_dataset)
    ds_train.data = train_data
    ds_train.track2id = track2id
    ds_train.artist2id = artist2id
    ds_train.track2artist = track2artist

    ds_valid = hydra.utils.instantiate(cfg.valid_dataset)
    ds_valid.data = valid_data
    ds_valid.track2id = track2id
    ds_valid.artist2id = artist2id
    ds_valid.track2artist = track2artist

    collator = hydra.utils.instantiate(cfg.collator)

    cfg.model.num_tracks = len(track2id)
    cfg.model.num_artists = len(artist2id)
    model = hydra.utils.instantiate(cfg.model)

    # save updated config to re-use in inference
    with open(os.path.join(cfg.training_args.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # it's needed only in trainer -> don't need to save
    track_id_to_artist_id = get_track_id_to_artist_id(
        track2id=track2id,
        artist2id=artist2id,
        track2artist=track2artist,
        num_special_tokens=cfg.model.num_special_tokens
    )
    # не очень удобно так прокидывать, потому что под капотом список конвертируется в ListConfig
    # cfg.trainer_params.track_id_to_artist_id = track_id_to_artist_id

    trainer_cls = hydra.utils.instantiate(cfg.trainer_cls)
    trainer = trainer_cls(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        **cfg.trainer_params
    )
    trainer.track_id_to_artist_id = torch.tensor(track_id_to_artist_id, device=args.device, dtype=torch.long)
    trainer.remove_callback(MLflowCallback)

    trainer.train()


if __name__ == "__main__":
    main()
