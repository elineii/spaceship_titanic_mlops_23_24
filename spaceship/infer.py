import numpy as np
import pandas as pd
from config import Config
from dataset import load_data
from model import NeuralNetwork
from safetensors.torch import load_model
from torch.utils.data import DataLoader


def main():
    # Load and preprocess train data
    dataset_test, indexes = load_data(is_train=False)
    loader_test = DataLoader(
        dataset_test,
        batch_size=Config.valid_bs,
        shuffle=False,
    )

    # Load model
    model = NeuralNetwork(
        input_size=dataset_test.features.shape[1], hidden_size=Config.hidden_size
    )
    load_model(model, "../models/model.safetensor")

    # Inference
    preds = []
    for batch_x in loader_test:
        preds.append(model(batch_x).detach().argmax(axis=1).numpy())

    preds = np.hstack(preds)

    # Load predictions
    res_df = pd.DataFrame(
        {"PassengerId": indexes.values, "Transported": preds.astype("bool")}
    )
    res_df.to_csv("res.csv", index=False)


if __name__ == "__main__":
    main()
