import logging
import pathlib

import hydra
import numpy as np
import pandas as pd
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from safetensors.torch import load_model
from spaceship.dataset import load_data
from spaceship.model import NeuralNetwork
from torch.utils.data import DataLoader


CURRENT_DIR = pathlib.Path(__file__).parent
CONFIG_PATH = str(CURRENT_DIR / "configs")
CONFIG_NAME = "config"
VERSION_BASE = "1.3"

logger = logging.getLogger(__name__)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=VERSION_BASE)
def main(cfg: DictConfig):
    url = cfg.general.github_url
    fs = DVCFileSystem(url, rev="main")
    fs.get_file(cfg.model.dvc_model_path, CURRENT_DIR / cfg.model.local_model_path)

    # Load and preprocess test data
    dataset_test, indexes = load_data(
        github_url=cfg.general.github_url,
        categorical_columns=list(cfg.data.categorical_columns),
        continious_columns=list(cfg.data.continious_columns),
        bool_columns=list(cfg.data.bool_columns),
        local_preprocessor_path=CURRENT_DIR / cfg.data.local_preprocessor_path,
        dvc_test_csv_path=cfg.data.dvc_test_csv_path,
        local_test_csv_path=CURRENT_DIR / cfg.data.local_test_csv_path,
        dvc_preprocessor_path=cfg.data.dvc_preprocessor_path,
        is_train=False,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=cfg.data.valid_bs,
        shuffle=False,
    )

    # Load model
    params = {
        "p_dropout": cfg.model.p_dropout,
        "lr": cfg.model.lr,
        "hidden_size": cfg.model.hidden_size,
    }
    model = NeuralNetwork(
        input_size=dataset_test.features.shape[1],
        **params,
    )
    load_model(model, CURRENT_DIR / cfg.model.local_model_path)

    logger.info("Start inference")
    # Inference
    preds = []
    for batch_x in loader_test:
        preds.append(model(batch_x).detach().argmax(axis=1).numpy())

    preds = np.hstack(preds)

    # Load predictions
    res_df = pd.DataFrame(
        {"PassengerId": indexes.values, "Transported": preds.astype("bool")}
    )

    submissions_dir = (CURRENT_DIR / cfg.general.result_path).parent
    submissions_dir.mkdir(exist_ok=True)
    res_df.to_csv(CURRENT_DIR / cfg.general.result_path, index=False)


if __name__ == "__main__":
    main()
