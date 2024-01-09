import hydra
import lightning as L
from dataset import load_data
from lightning.pytorch.loggers import MLFlowLogger
from model import NeuralNetwork
from omegaconf import DictConfig
from safetensors.torch import save_model
from torch.utils.data import DataLoader


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Load and preprocess train data
    dataset_train, dataset_val = load_data(
        github_url=cfg.general.github_url,
        categorical_columns=cfg.data.categorical_columns,
        continious_columns=cfg.data.continious_columns,
        bool_columns=cfg.data.bool_columns,
        local_preprocessor_path=cfg.data.local_preprocessor_path,
        dvc_train_csv_path=cfg.data.dvc_train_csv_path,
        local_train_csv_path=cfg.data.local_train_csv_path,
        val_size=cfg.data.val_size,
        random_state=cfg.general.random_state,
        continious_inputer_strategy=cfg.data.continious_inputer_strategy,
        is_train=True,
    )
    loader_train, loader_val = (
        DataLoader(dataset_train, batch_size=cfg.data.train_bs),
        DataLoader(dataset_val, batch_size=cfg.data.valid_bs, shuffle=False),
    )

    # Train model
    model = NeuralNetwork(
        input_size=dataset_train.features.shape[1], hidden_size=cfg.model.hidden_size
    )

    mlflow_logger = MLFlowLogger(
        cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_uri,
    )

    trainer = L.Trainer(max_epochs=cfg.model.max_epochs, logger=mlflow_logger)
    trainer.fit(model, loader_train, loader_val)

    # Save model
    save_model(model, cfg.model.local_model_path)


if __name__ == "__main__":
    main()
