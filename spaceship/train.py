import lightning as L
from config import Config
from dataset import load_data
from lightning.pytorch.loggers import MLFlowLogger
from model import NeuralNetwork
from safetensors.torch import save_model
from torch.utils.data import DataLoader


def main():
    # Load and preprocess train data
    dataset_train, dataset_val = load_data(is_train=True)
    loader_train, loader_val = (
        DataLoader(dataset_train, batch_size=Config.train_bs),
        DataLoader(dataset_val, batch_size=Config.valid_bs, shuffle=False),
    )

    # Train model
    model = NeuralNetwork(
        input_size=dataset_train.features.shape[1], hidden_size=Config.hidden_size
    )

    mlflow_logger = MLFlowLogger(
        "spaceship",
        tracking_uri=Config.mlflow_uri,
        log_model=False,
    )

    trainer = L.Trainer(max_epochs=Config.max_epochs, logger=mlflow_logger)
    trainer.fit(model, loader_train, loader_val)

    # Save model
    save_model(model, "../models/model.safetensor")


if __name__ == "__main__":
    main()
