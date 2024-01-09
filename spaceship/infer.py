import hydra
import numpy as np
import pandas as pd
from dataset import load_data
from dvc.api import DVCFileSystem
from model import NeuralNetwork
from omegaconf import DictConfig
from safetensors.torch import load_model
from torch.utils.data import DataLoader


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    url = cfg.general.github_url
    fs = DVCFileSystem(url, rev="main")
    fs.get_file(cfg.model.dvc_model_path, cfg.model.local_model_path)

    # Load and preprocess test data
    dataset_test, indexes = load_data(
        github_url=cfg.general.github_url,
        categorical_columns=cfg.data.categorical_columns,
        continious_columns=cfg.data.continious_columns,
        bool_columns=cfg.data.bool_columns,
        local_preprocessor_path=cfg.data.local_preprocessor_path,
        dvc_test_csv_path=cfg.data.dvc_test_csv_path,
        local_test_csv_path=cfg.data.local_test_csv_path,
        dvc_preprocessor_path=cfg.data.dvc_preprocessor_path,
        is_train=False,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=cfg.data.valid_bs,
        shuffle=False,
    )

    # Load model
    model = NeuralNetwork(
        input_size=dataset_test.features.shape[1], hidden_size=cfg.model.hidden_size
    )
    load_model(model, cfg.model.local_model_path)

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
