import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import train_dataloader, test_dataloader
from models import LstmCell

import tqdm
import matplotlib.pyplot as plt


torch.backends.cudnn.deterministic = True

torch.manual_seed(42)
np.random.seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"


H1_IN = 13
H2_OUT = 32
OUTPUT_DIM = 4

model = LstmCell(H1_IN, H2_OUT, OUTPUT_DIM).to(device)


optim = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


def calculate_accuracy(prediction, y):
    top_pred = prediction.argmax(axis=1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float()/y.shape[0]
    return acc


def train(model, iterator, optim, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for x, y in iterator:
        optim.zero_grad()

        x = x.to(device)
        y = y.to(device)

        prediction = model(x)

        loss = criterion(prediction, y)
        loss.backward()

        acc = calculate_accuracy(prediction, y)

        optim.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)


def evaluate(model, iterator, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    for x, y in iterator:

        x = x.to(device)
        y = y.to(device)

        prediction = model(x)

        loss = criterion(prediction, y)

        acc = calculate_accuracy(prediction, y)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)


def plot_history(train_hist, valid_hist, n_epoch, filename, type_of):
    plt.figure(figsize=(18, 8))

    x_data = list(range(n_epoch))
    plt.plot(x_data, train_hist, color='y', label="train", marker='.')
    plt.plot(x_data, valid_hist, color='g', label="valid")
    plt.title("Train-Valid Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(type_of)
    plt.legend()
    plt.savefig(filename)


if __name__ == "__main__":
    N_EPOCHS = 50
    best_valid_loss = float('inf')

    train_loss_hist, train_acc_hist = [], []
    valid_loss_hist, valid_acc_hist = [], []

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(
            model, train_dataloader, optim, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_dataloader, device)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        valid_loss_hist.append(valid_loss)
        valid_acc_hist.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "music-model.pt")

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}")
            print(
                f"Train loss: {train_loss:.3f} | Train Accuracy: {train_acc:.3f}")
            print(
                f"Valid loss: {valid_loss:.3f} | Valid Accuracy: {valid_acc:.3f}")

    plot_history(train_loss_hist, valid_loss_hist, N_EPOCHS,
                 filename="./assets/lossescomparison.jpg", type_of="Loss")
    plot_history(train_acc_hist, valid_acc_hist, N_EPOCHS,
                 filename="./assets/accuracycomparison.jpg", type_of="Accuracy")
