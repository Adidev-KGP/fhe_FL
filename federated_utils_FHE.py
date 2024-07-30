# from typing import List, Tuple, Union

# import numpy as np
# import openml
# from sklearn.linear_model import LogisticRegression

# XY = Tuple[np.ndarray, np.ndarray]
# Dataset = Tuple[XY, XY]
# LogRegParams = Union[XY, Tuple[np.ndarray]]
# XYList = List[XY]


# def get_model_parameters(model: LogisticRegression) -> LogRegParams:
#     """Returns the paramters of a sklearn LogisticRegression model."""
#     if model.fit_intercept:
#         params = [
#             model.coef_,
#             model.intercept_,
#         ]
#     else:
#         params = [
#             model.coef_,
#         ]
#     return params


# def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
#     """Sets the parameters of a sklearn LogisticRegression model."""
#     model.coef_ = params[0]
#     if model.fit_intercept:
#         model.intercept_ = params[1]
#     return model


# def set_initial_params(model: LogisticRegression):
#     """Sets initial parameters as zeros Required since model params are uninitialized
#     until model.fit is called.

#     But server asks for initial parameters from clients at launch. Refer to
#     sklearn.linear_model.LogisticRegression documentation for more information.
#     """
#     n_classes = 10  # MNIST has 10 classes
#     n_features = 784  # Number of features in dataset
#     model.classes_ = np.array([i for i in range(10)])

#     model.coef_ = np.zeros((n_classes, n_features))
#     if model.fit_intercept:
#         model.intercept_ = np.zeros((n_classes,))


# def load_mnist() -> Dataset:
#     """Loads the MNIST dataset using OpenML.

#     OpenML dataset link: https://www.openml.org/d/554
#     """
#     mnist_openml = openml.datasets.get_dataset(554)
#     Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
#     X = Xy[:, :-1]  # the last column contains labels
#     y = Xy[:, -1]
#     # First 60000 samples consist of the train set
#     x_train, y_train = X[:60000], y[:60000]
#     x_test, y_test = X[60000:], y[60000:]
#     return (x_train, y_train), (x_test, y_test)


# def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
#     """Shuffle X and y."""
#     rng = np.random.default_rng()
#     idx = rng.permutation(len(X))
#     return X[idx], y[idx]


# def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
#     """Split X and y into a number of partitions."""
#     return list(zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions)))


from typing import List, Tuple, Union
import numpy as np
import openml
import torch
import torch.nn as nn

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
NNParams = List[np.ndarray]
XYList = List[XY]

def get_model_parameters(model: nn.Module) -> NNParams:
    """Returns the parameters of a PyTorch model as NumPy arrays."""
    return [param.detach().cpu().numpy() for param in model.parameters()]

def set_model_parameters(model: nn.Module, params: NNParams) -> nn.Module:
    """Sets the parameters of a PyTorch model from NumPy arrays."""
    for param, new_param in zip(model.parameters(), params):
        param.data = torch.tensor(new_param).to(param.device)
    return model

def set_initial_params(model: nn.Module):
    """Sets initial parameters to zeros. Required to initialize model params."""
    for param in model.parameters():
        param.data = torch.zeros_like(param.data)

def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML."""
    try:
        mnist_openml = openml.datasets.get_dataset(
            554, 
            download_data=True, 
            download_qualities=True, 
            download_features_meta_data=True
        )
        Xy, _, _, _ = mnist_openml.get_data(dataset_format="dataframe")
        Xy = Xy.to_numpy()  # Convert dataframe to numpy array
        X = Xy[:, :-1].astype(np.float32)  # Convert features to float32
        y = Xy[:, -1].astype(np.int64)     # Convert labels to int64
    except Exception as e:
        print(f"Failed to load dataset from OpenML: {e}")
        # Use a local copy of the MNIST dataset as a fallback
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"].astype(np.int64)
        X = X.astype(np.float32)

    # First 60000 samples consist of the train set
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)

def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions)))
