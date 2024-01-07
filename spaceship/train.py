from torch.utils.data import DataLoader
import lightning as L

from model import NeuralNetwork
from dataset import load_data
from safetensors.torch import save_model
from config import Config


def main():
    # Load and preprocess train data
    dataset_train, dataset_val = load_data(is_train=True)
    loader_train, loader_val = (
        DataLoader(dataset_train, batch_size=Config.train_bs),
        DataLoader(dataset_val, batch_size=Config.valid_bs, shuffle=False)
    )

    # Train model
    model = NeuralNetwork(
        input_size=dataset_train.features.shape[1],
        hidden_size=Config.hidden_size
    )
    trainer = L.Trainer(max_epochs=Config.max_epochs)
    trainer.fit(model, loader_train, loader_val)

    # Запись метрик

    # Сохранение на диск
    save_model(model, "../models/model.safetensor")


if __name__ == "__main__":
    main()
