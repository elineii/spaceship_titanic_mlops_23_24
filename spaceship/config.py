import torch.nn as nn


class Config:
    continious_inputer_strategy = "median"
    random_state = 42
    val_size = 0.1
    lr = 1e-2
    nb_epochs = 5
    train_bs = 32
    valid_bs = 64
    train_split = 0.8
    k_folds = 5
    max_epochs = 10
    hidden_size = 32
    device = "cpu"
    train_loss_fn = nn.BCEWithLogitsLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss()
    target_name = "Transported"
    mlflow_uri = "http://127.0.0.1:5000"
    continious_columns = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]
    categorical_columns = ["HomePlanet", "Destination"]
    bool_columns = ["CryoSleep", "VIP"]
