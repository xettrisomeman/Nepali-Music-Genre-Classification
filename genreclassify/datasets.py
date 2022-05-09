import json
import numpy as np

from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(42)
np.random.seed(42)



def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data

    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y




def prepare_dataset(data_path, test_size=0.25):
    """Creates train, validation and test sets.


    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation


    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)


    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = prepare_dataset(data_path="data.json")




class CustomDataset(Dataset):

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        features = torch.from_numpy(self.features).float()
        labels = torch.from_numpy(self.labels).long()
        return features[index], labels[index]

    def __len__(self):
        return len(self.features)



train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
)

