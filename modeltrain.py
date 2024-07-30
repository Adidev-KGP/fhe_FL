import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

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
        "y_test": y_test
    }

    return data

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

def trainModel():
    data = prepareTrainData()
    input_shape = data["X_train"].shape[1]

    model = getModel(input_shape)

    # Train the model
    history = model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(data["X_test"], data["y_test"])
    print(f'Test Accuracy: {accuracy:.4f}')

    # Plot training & validation accuracy and loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

    return model

finalModel = trainModel()

# Make predictions
data = prepareTrainData()
predictions = finalModel.predict(data["X_test"])

print("the rp", predictions)
