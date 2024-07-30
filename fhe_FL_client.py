# import warnings

# import federated_utils_FHE
# import flwr as fl
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import log_loss

# if __name__ == "__main__":
#     # Load MNIST dataset from https://www.openml.org/d/554
#     (X_train, y_train), (X_test, y_test) = federated_utils_FHE.load_mnist()

#     # Split train set into 10 partitions and randomly use one for training.
#     partition_id = np.random.choice(10)
#     (X_train, y_train) = federated_utils_FHE.partition(X_train, y_train, 10)[partition_id]

#     # Create LogisticRegression Model
#     model = LogisticRegression(
#         penalty="l2",
#         max_iter=2,  # local epoch
#         warm_start=True,  # prevent refreshing weights when fitting
#     )

#     # Setting initial parameters, akin to model.compile for Keras models
#     federated_utils_FHE.set_initial_params(model)

#     # Define Flower client
#     class MnistClient(fl.client.NumPyClient):
#         def get_parameters(self, config):  # type: ignore
#             return federated_utils_FHE.get_model_parameters(model)

#         def fit(self, parameters, config):  # type: ignore
#             federated_utils_FHE.set_model_params(model, parameters)
#             # Ignore convergence failure due to low local epochs
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 model.fit(X_train, y_train)
#             print(f"Training finished for round {config['server_round']}")
#             return federated_utils_FHE.get_model_parameters(model), len(X_train), {}

#         def evaluate(self, parameters, config):  # type: ignore
#             federated_utils_FHE.set_model_params(model, parameters)
#             loss = log_loss(y_test, model.predict_proba(X_test))
#             accuracy = model.score(X_test, y_test)
#             return loss, len(X_test), {"accuracy": accuracy}

#     # Start Flower client
#     fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())


import warnings
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss
import federated_utils_FHE

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = federated_utils_FHE.load_mnist()
partition_id = np.random.choice(10)
(X_train, y_train) = federated_utils_FHE.partition(X_train, y_train, 10)[partition_id]

# Define the custom neural network
class CustomNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CustomNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

input_shape = 784  # MNIST data has 28x28 images, flattened to 784
num_classes = 10   # MNIST has 10 classes
model = CustomNN(input_shape, num_classes)

# Initialize model parameters
federated_utils_FHE.set_initial_params(model)

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):  # type: ignore
        print("HUE HUE HUE GET PARAMETERS")
        return federated_utils_FHE.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        federated_utils_FHE.set_model_parameters(model, parameters)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(1):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        print(f"Training finished for round {config['server_round']}")
        return federated_utils_FHE.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        federated_utils_FHE.set_model_parameters(model, parameters)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            loss = nn.CrossEntropyLoss()(outputs, y_test_tensor)
            accuracy = (outputs.argmax(dim=1) == y_test_tensor).float().mean().item()
        return
