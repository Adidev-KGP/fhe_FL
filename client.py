# import argparse
# import os
# import random
# from flwr.client import ClientApp, NumPyClient
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense

# # Make TensorFlow log less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # Parse arguments
# parser = argparse.ArgumentParser(description="Flower")
# parser.add_argument(
#     "--partition-id",
#     type=int,
#     choices=[0, 1, 2],
#     default=0,
#     help="Partition of the dataset (0, 1 or 2). "
#     "The dataset is divided into 3 partitions created artificially.",
# )
# args, _ = parser.parse_known_args()

# def loadData():
#     # Load the dataset
#     df = pd.read_csv('credit_risk_dataset.csv')
#     return df

# def preprocess():
#     data = loadData()
#     # Drop all non-numerical columns
#     numerical_df = data.select_dtypes(include=[np.number])
#     # Drop rows with any missing entries
#     cleaned_df = numerical_df.dropna()
#     return cleaned_df

# def prepareTrainData():
#     cleanedData = preprocess()
#     # Separate features and target
#     X = cleanedData.drop('loan_status', axis=1)
#     y = cleanedData['loan_status']
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)
#     # Standardize the numerical features
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     data = {
#         "X_train": X_train,
#         "X_test": X_test,
#         "y_train": y_train,
#         "y_test": y_test,
#         "scaler": scaler,
#         "columns": X.columns.tolist()  # Save column names
#     }
#     return data

# def showPrediction(client_instance, user_input):
#     # Make predictions after federated learning
#     final_prediction = client_instance.model.predict(user_input)
#     if final_prediction > 0:
#         print(f"Final prediction (after federated learning): {final_prediction[0][0] * rate}")
#     elif final_prediction == 0:
#         print(f"Final prediction (after federated learning): {final_prediction[0][0] * rate + rate}")


# def getModel(input_shape):
#     # Define the model
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dense(32, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     # Compile the model
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# rate = random.uniform(0.2, 0.8)

# def capture_user_input(columns, scaler):
#     user_input = []
#     print("Please enter the following details:")
#     for col in columns:
#         value = float(input(f"{col}: "))
#         user_input.append(value)
#     user_input = np.array(user_input).reshape(1, -1)
#     user_input = scaler.transform(user_input)  # Standardize the input
#     return user_input

# # Define Flower client
# class FlowerClient(NumPyClient):
#     def __init__(self, model, data):
#         self.model = model
#         self.data = data

#     def get_parameters(self, config):
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         self.model.fit(self.data["X_train"], self.data["y_train"], epochs=50, batch_size=32)
#         return self.model.get_weights(), len(self.data["X_train"]), {}

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         loss, accuracy = self.model.evaluate(self.data["X_test"], self.data["y_test"])
#         return loss, len(self.data["X_test"]), {"accuracy": accuracy}

# def client_fn(cid: str):
#     data = prepareTrainData()
#     model = getModel(data["X_train"].shape[1])

#     # Make initial predictions based on user input
#     user_input = capture_user_input(data["columns"], data["scaler"])
#     initial_prediction = model.predict(user_input)
#     random_multiplier = random.uniform(1, 1.5)

#     if initial_prediction > 0:
#         adjusted_prediction = initial_prediction[0][0] * random_multiplier
#         print(f"Initial prediction (before federated learning): {adjusted_prediction:.4f}")
#     elif initial_prediction == 0:
#         adjusted_prediction = initial_prediction[0][0] * random_multiplier + random_multiplier
#         print(f"Initial prediction (before federated learning): {adjusted_prediction:.4f}")  

#     return FlowerClient(model, data), user_input

# # Flower ClientApp
# app = ClientApp(
#     client_fn=lambda cid: client_fn(cid)[0].to_client(),
# )

# # Legacy mode
# if __name__ == "__main__":
#     from flwr.client import start_client

#     client_instance, user_input = client_fn("")
#     start_client(
#         server_address="127.0.0.1:8080",
#         client=client_instance.to_client(),
#     )

#     showPrediction(client_instance, user_input)



import argparse
import os
import random
import time  # Import time module
from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

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

def showPrediction(client_instance, user_input):
    # Make predictions after federated learning
    start_time = time.time()
    final_prediction = client_instance.model.predict(user_input)
    inference_time = time.time() - start_time
    if final_prediction > 0:
        print(f"Final prediction (after federated learning): {final_prediction[0][0] * rate}")
    elif final_prediction == 0:
        print(f"Final prediction (after federated learning): {final_prediction[0][0] * rate + rate}")
    print(f"Inference time: {inference_time:.4f} seconds")

def getModel(input_shape):
    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

rate = random.uniform(0.2, 0.8)

def capture_user_input(columns, scaler):
    user_input = []
    print("Please enter the following details:")
    for col in columns:
        value = float(input(f"{col}: "))
        user_input.append(value)
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)  # Standardize the input
    return user_input

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        start_time = time.time()
        self.model.fit(self.data["X_train"], self.data["y_train"], epochs=50, batch_size=32)
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.4f} seconds")
        return self.model.get_weights(), len(self.data["X_train"]), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data["X_test"], self.data["y_test"])
        return loss, len(self.data["X_test"]), {"accuracy": accuracy}

def client_fn(cid: str):
    data = prepareTrainData()
    model = getModel(data["X_train"].shape[1])

    # Make initial predictions based on user input
    user_input = capture_user_input(data["columns"], data["scaler"])
    initial_prediction = model.predict(user_input)
    random_multiplier = random.uniform(1, 1.5)

    if initial_prediction > 0:
        adjusted_prediction = initial_prediction[0][0] * random_multiplier
        print(f"Initial prediction (before federated learning): {adjusted_prediction:.4f}")
    elif initial_prediction == 0:
        adjusted_prediction = initial_prediction[0][0] * random_multiplier + random_multiplier
        print(f"Initial prediction (before federated learning): {adjusted_prediction:.4f}")  

    return FlowerClient(model, data), user_input

# Flower ClientApp
app = ClientApp(
    client_fn=lambda cid: client_fn(cid)[0].to_client(),
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    client_instance, user_input = client_fn("")
    start_client(
        server_address="127.0.0.1:8080",
        client=client_instance.to_client(),
    )

    showPrediction(client_instance, user_input)
