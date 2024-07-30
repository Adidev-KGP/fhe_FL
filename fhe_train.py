# import torch
# import torch.nn as nn
# import torch.optim as optim
# import brevitas.nn as qnn
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from concrete.ml.torch.compile import compile_brevitas_qat_model

# # Define the PyTorch model using Brevitas
# class QATSimpleNet(nn.Module):
#     def __init__(self, input_shape, n_bits=3):
#         super(QATSimpleNet, self).__init__()
#         self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
#         self.fc1 = qnn.QuantLinear(input_shape, 64, True, weight_bit_width=n_bits, bias_quant=None)
#         self.relu1 = nn.ReLU()
#         self.quant1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
#         self.fc2 = qnn.QuantLinear(64, 32, True, weight_bit_width=n_bits, bias_quant=None)
#         self.relu2 = nn.ReLU()
#         self.quant2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
#         self.fc3 = qnn.QuantLinear(32, 1, True, weight_bit_width=n_bits, bias_quant=None)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.quant_inp(x)
#         x = self.relu1(self.fc1(x))
#         x = self.quant1(x)
#         x = self.relu2(self.fc2(x))
#         x = self.quant2(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x

# # Input shape based on your dataset
# input_shape = 10  # Example input shape, update according to your dataset
# model = QATSimpleNet(input_shape)

# # Create representative input for compilation
# torch_input = torch.randn(100, input_shape)

# # Compile the model with Concrete ML
# quantized_module = compile_brevitas_qat_model(
#     model,
#     torch_input,
#     rounding_threshold_bits={"n_bits": 6, "method": "approximate"}
# )

# # # Save the quantized module for later use
# # torch.save(quantized_module, 'quantized_model.pt')

# # Example training data, replace with your actual data
# X_train = np.random.rand(1000, input_shape).astype(np.float32)
# y_train = np.random.randint(0, 2, 1000).astype(np.float32)

# # Convert to PyTorch tensors
# X_train_tensor = torch.tensor(X_train)
# y_train_tensor = torch.tensor(y_train).unsqueeze(1)  # Ensure y is of shape (N, 1)

# # Create DataLoader
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Define loss and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# # Save the trained model
# torch.save(model.state_dict(), 'trained_model.pt')

# print("HELLO")
# x_test = np.array([np.random.randn(10)])

# # Perform encrypted inference
# y_pred = quantized_module.forward(x_test, fhe="execute")
# print(f"Encrypted prediction: {y_pred}")


import argparse
import os
import random
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import brevitas.nn as qnn
from torch.utils.data import DataLoader, TensorDataset
from concrete.ml.torch.compile import compile_brevitas_qat_model

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse arguments
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=[0, 1, 2],
    default=0,
    help="Partition of the dataset (0, 1 or 2). "
    "The dataset is divided into 3 partitions created artificially.",
)
args, _ = parser.parse_known_args()

def loadData():
    # Load the dataset
    df = pd.read_csv('credit_risk_dataset.csv')
    return df

def preprocess():
    data = loadData()
    # Drop all non-numerical columns
    numerical_df = data.select_dtypes(include=[np.number])
    # Drop rows with any missing entries
    cleaned_df = numerical_df.dropna()
    return cleaned_df

def prepareTrainData():
    cleanedData = preprocess()
    # Separate features and target
    X = cleanedData.drop('loan_status', axis=1)
    y = cleanedData['loan_status']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)
    # Standardize the numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "columns": X.columns.tolist()  # Save column names
    }
    return data

# Define the PyTorch model using Brevitas
class QATSimpleNet(nn.Module):
    def __init__(self, input_shape, n_bits=3):
        super(QATSimpleNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(input_shape, 64, True, weight_bit_width=n_bits, bias_quant=None)
        self.relu1 = nn.ReLU()
        self.quant1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(64, 32, True, weight_bit_width=n_bits, bias_quant=None)
        self.relu2 = nn.ReLU()
        self.quant2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(32, 1, True, weight_bit_width=n_bits, bias_quant=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.fc1(x))
        x = self.quant1(x)
        x = self.relu2(self.fc2(x))
        x = self.quant2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train_and_compile_model(data):
    input_shape = data["X_train"].shape[1]
    model = QATSimpleNet(input_shape)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_tensor = torch.tensor(data["y_train"].values, dtype=torch.float32).unsqueeze(1)  # Ensure y is of shape (N, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    start_time = time.time()  # Start timing the training
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            # Calculate accuracy
            predicted = outputs.round()  # Convert probabilities to binary predictions
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    training_time = time.time() - start_time  # End timing the training
    print(f"Training time: {training_time:.4f} seconds")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pt')

    # Create representative input for compilation
    torch_input = torch.randn(100, input_shape, dtype=torch.float32)

    # Compile the model with Concrete ML
    quantized_module = compile_brevitas_qat_model(
        model,
        torch_input,
        rounding_threshold_bits={"n_bits": 6, "method": "approximate"}
    )

    return quantized_module

def capture_user_input(columns, scaler):
    user_input = []
    print("Please enter the following details:")
    for col in columns:
        value = float(input(f"{col}: "))
        user_input.append(value)
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)  # Standardize the input
    return user_input

def showPrediction(quantized_module, user_input):
    # Perform encrypted inference
    start_time = time.time()
    y_pred = quantized_module.forward(user_input, fhe="execute")
    inference_time = time.time() - start_time
    print(f"Encrypted prediction: {y_pred}")
    print(f"Inference time: {inference_time:.4f} seconds")

def main():
    data = prepareTrainData()
    quantized_module = train_and_compile_model(data)

    # Make initial predictions based on user input
    user_input = capture_user_input(data["columns"], data["scaler"])

    showPrediction(quantized_module, user_input)

if __name__ == "__main__":
    main()
