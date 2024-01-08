import torch
import torch.nn as nn
from config import Config
from lightning import LightningModule


def binary_acc(y_pred, y_test):
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc


class NeuralNetwork(LightningModule):
    def __init__(self, input_size, hidden_size, num_classes=2, p_droput=0.1):
        super(NeuralNetwork, self).__init__()
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Model Architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_droput)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=Config.lr)
        return optimizer

    def step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y.long())
        acc = binary_acc(pred.argmax(axis=1), y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
