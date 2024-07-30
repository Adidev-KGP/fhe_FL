# import pickle
# from typing import Dict

# import federated_utils_FHE
# import flwr as fl
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import log_loss


# def fit_round(server_round: int) -> Dict:
#     """Send round number to client."""
#     return {"server_round": server_round}


# def get_evaluate_fn(model: LogisticRegression):
#     """Return an evaluation function for server-side evaluation."""

#     # Load test data here to avoid the overhead of doing it in `evaluate` itself
#     _, (X_test, y_test) = federated_utils_FHE.load_mnist()

#     # The `evaluate` function will be called after every round
#     def evaluate(server_round, parameters: fl.common.NDArrays, config):
#         # Update model with the latest parameters
#         federated_utils_FHE.set_model_params(model, parameters)
#         loss = log_loss(y_test, model.predict_proba(X_test))
#         accuracy = model.score(X_test, y_test)
#         return loss, {"accuracy": accuracy}

#     return evaluate


# # Start Flower server for five rounds of federated learning
# if __name__ == "__main__":
#     model = LogisticRegression()
#     federated_utils_FHE.set_initial_params(model)
#     strategy = fl.server.strategy.FedAvg(
#         min_available_clients=2,
#         evaluate_fn=get_evaluate_fn(model),
#         on_fit_config_fn=fit_round,
#     )
#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         strategy=strategy,
#         config=fl.server.ServerConfig(num_rounds=5),
#     )
#     with open("model.pkl", "wb") as file:
#         pickle.dump(model, file)


import pickle
from typing import Dict
import flwr as fl
import torch
import torch.nn as nn
import federated_utils_FHE

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

input_shape = 784
num_classes = 10
model = CustomNN(input_shape, num_classes)
federated_utils_FHE.set_initial_params(model)

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: nn.Module):
    """Return an evaluation function for server-side evaluation."""
    _, (X_test, y_test) = federated_utils_FHE.load_mnist()

    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        federated_utils_FHE.set_model_parameters(model, parameters)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            loss = nn.CrossEntropyLoss()(outputs, y_test_tensor)
            accuracy = (outputs.argmax(dim=1) == y_test_tensor).float().mean().item()
        return loss.item(), {"accuracy": accuracy}

    return evaluate

# Start Flower server for federated learning
if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
