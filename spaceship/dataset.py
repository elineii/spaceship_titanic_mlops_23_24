import logging

import pandas as pd
import torch
from dvc.api import DVCFileSystem
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skops.io import dump, load
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class SpaceshipTitanicDataset(Dataset):
    def __init__(self, features, target=None, is_test=False):
        self.features = features
        self.target = target
        self.is_test = is_test

    def __getitem__(self, idx):
        data = self.features[idx]
        if self.is_test:
            return torch.tensor(data, dtype=torch.float32)
        else:
            target = self.target[idx]
            return torch.tensor(data, dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32
            )

    def __len__(self):
        return len(self.features)


def load_data(
    github_url,
    categorical_columns,
    continious_columns,
    bool_columns,
    local_preprocessor_path,
    dvc_train_csv_path=None,
    local_train_csv_path=None,
    dvc_test_csv_path=None,
    local_test_csv_path=None,
    dvc_preprocessor_path=None,
    val_size=None,
    random_state=None,
    continious_inputer_strategy=None,
    is_train=True,
):
    fs = DVCFileSystem(github_url, rev="main")

    if is_train:
        logger.info("Start loading train data")
        # Download and preprocess train data
        fs.get_file(dvc_train_csv_path, local_train_csv_path)
        df_train = pd.read_csv(local_train_csv_path)
        target_train = df_train["Transported"].map({True: 1, False: 0}).values
        df_train = df_train[categorical_columns + continious_columns + bool_columns]

        # Train_val_split
        X_train, X_val, y_train, y_val = train_test_split(
            df_train,
            target_train,
            test_size=val_size,
            random_state=random_state,
        )

        # Fit and apply Pipeline to preprocess the train data
        # and save it to use later for test data

        continious_pipeline = Pipeline(
            [
                (
                    "fill_na_transformer",
                    SimpleImputer(strategy=continious_inputer_strategy),
                ),
                ("preprocess_transformer", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "fill_na_transformer",
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                ),
                ("preprocess_transformer", OneHotEncoder()),
            ]
        )

        bool_pipeline = Pipeline(
            [
                ("fill_na_transformer", SimpleImputer(strategy="most_frequent")),
                ("preprocess_transformer", OneHotEncoder(drop="if_binary")),
            ]
        )

        column_transformer = ColumnTransformer(
            [
                (
                    "preprocess_categorical",
                    categorical_pipeline,
                    categorical_columns,
                ),
                (
                    "preprocess_continious",
                    continious_pipeline,
                    continious_columns,
                ),
                (
                    "preprocess_bool",
                    bool_pipeline,
                    bool_columns,
                ),
            ],
            remainder="drop",
        )

        X_train_preprocessed = column_transformer.fit_transform(X_train)
        dump(column_transformer, local_preprocessor_path)

        X_val_preprocessed = column_transformer.transform(X_val)

        dataset_train = SpaceshipTitanicDataset(
            X_train_preprocessed, y_train, is_test=False
        )
        dataset_val = SpaceshipTitanicDataset(X_val_preprocessed, y_val, is_test=False)
        logger.info("Train data successfully loaded")
        return dataset_train, dataset_val

    else:
        # Download and preprocess test data
        logger.info("Start loading test data")
        fs.get_file(dvc_test_csv_path, local_test_csv_path)
        df_test = pd.read_csv(local_test_csv_path)
        indexes = df_test["PassengerId"]
        df_test = df_test[categorical_columns + continious_columns + bool_columns]
        fs.get_file(dvc_preprocessor_path, local_preprocessor_path)
        column_transformer = load(local_preprocessor_path, trusted=["numpy.dtype"])

        df_test_preprocessed = column_transformer.transform(df_test)

        dataset_test = SpaceshipTitanicDataset(df_test_preprocessed, is_test=True)
        logger.info("Test data successfully loaded")
        return dataset_test, indexes
