import pandas as pd
import torch
from config import Config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skops.io import dump, load
from torch.utils.data import Dataset


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


def load_data(is_train=True):
    if is_train:
        # Download and preprocess train data
        df_train = pd.read_csv("../data/train.csv")
        target_train = df_train["Transported"].map({True: 1, False: 0}).values
        df_train = df_train[
            Config.categorical_columns + Config.continious_columns + Config.bool_columns
        ]

        # Train_val_split
        X_train, X_val, y_train, y_val = train_test_split(
            df_train,
            target_train,
            test_size=Config.val_size,
            random_state=Config.random_state,
        )

        # Fit and apply Pipeline to preprocess the train data
        # and save it to use later for test data

        continious_pipeline = Pipeline(
            [
                (
                    "fill_na_transformer",
                    SimpleImputer(strategy=Config.continious_inputer_strategy),
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
                    Config.categorical_columns,
                ),
                (
                    "preprocess_continious",
                    continious_pipeline,
                    Config.continious_columns,
                ),
                (
                    "preprocess_bool",
                    bool_pipeline,
                    Config.bool_columns,
                ),
            ],
            remainder="drop",
        )

        X_train_preprocessed = column_transformer.fit_transform(X_train)
        dump(column_transformer, "preprocessing_pipeline.skops")

        X_val_preprocessed = column_transformer.transform(X_val)

        dataset_train = SpaceshipTitanicDataset(
            X_train_preprocessed, y_train, is_test=False
        )
        dataset_val = SpaceshipTitanicDataset(X_val_preprocessed, y_val, is_test=False)
        return dataset_train, dataset_val

    else:
        # Download and preprocess train data
        df_test = pd.read_csv("../data/test.csv")
        indexes = df_test["PassengerId"]
        df_test = df_test[
            Config.categorical_columns + Config.continious_columns + Config.bool_columns
        ]
        column_transformer = load("preprocessing_pipeline.skops", trusted=["numpy.dtype"])

        df_test_preprocessed = column_transformer.transform(df_test)

        dataset_test = SpaceshipTitanicDataset(df_test_preprocessed, is_test=True)
        return dataset_test, indexes
