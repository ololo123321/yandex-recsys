import os
import sys
import logging
import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger("predict")


@hydra.main(config_path="../config", config_name="predict")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info("loading train data...")
    test_data = []
    with open(cfg.input_path) as f:
        for line in tqdm.tqdm(f):
            tracks = line.strip().split()
            test_data.append(tracks)
            if len(test_data) == cfg.limit:
                break

    logger.info(f"num test examples: {len(test_data)}")
    predictor = hydra.utils.instantiate(cfg.predictor)
    preds = predictor(test_data)

    logger.info("saving predictions...")
    with open(cfg.output_path, "w") as f:
        for tracks in preds:
            f.write(" ".join(tracks) + "\n")


if __name__ == "__main__":
    main()
