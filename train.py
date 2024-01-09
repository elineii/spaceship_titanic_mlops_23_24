import logging
import pathlib

import hydra
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from safetensors.torch import save_model
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
    # Load and preprocess train data
    dataset_train, dataset_val = load_data(
        github_url=cfg.general.github_url,
        categorical_columns=list(cfg.data.categorical_columns),
        continious_columns=list(cfg.data.continious_columns),
        bool_columns=list(cfg.data.bool_columns),
        local_preprocessor_path=CURRENT_DIR / cfg.data.local_preprocessor_path,
        dvc_train_csv_path=cfg.data.dvc_train_csv_path,
        local_train_csv_path=CURRENT_DIR / cfg.data.local_train_csv_path,
        val_size=cfg.data.val_size,
        random_state=cfg.general.random_state,
        continious_inputer_strategy=cfg.data.continious_inputer_strategy,
        is_train=True,
    )
    loader_train, loader_val = (
        DataLoader(dataset_train, batch_size=cfg.data.train_bs),
        DataLoader(dataset_val, batch_size=cfg.data.valid_bs, shuffle=False),
    )

    mlflow_logger = MLFlowLogger(
        cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_uri,
    )

    # Train model
    params = {
        "p_dropout": cfg.model.p_dropout,
        "lr": cfg.model.lr,
        "hidden_size": cfg.model.hidden_size,
    }
    model = NeuralNetwork(
        input_size=dataset_train.features.shape[1],
        **params,
    )
    mlflow_logger.log_hyperparams(params)

    trainer = L.Trainer(max_epochs=cfg.model.max_epochs, logger=mlflow_logger)
    trainer.fit(model, loader_train, loader_val)

    # Save model
    save_model(model, CURRENT_DIR / cfg.model.local_model_path)


if __name__ == "__main__":
    main()
