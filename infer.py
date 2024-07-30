from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import tensorflow as tf
import pandas as pd

def pipeline(data):
    # Fill missing values
    imputer = SimpleImputer(strategy='mean')
    data[data.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(data.select_dtypes(include=[np.number]))

    # Convert categorical data to numerical if needed
    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    data['person_home_ownership'] = data['person_home_ownership'].map({'RENT': 1, 'OWN': 2, 'MORTGAGE': 3, 'OTHER': 4})
    data['loan_intent'] = data['loan_intent'].map({'PERSONAL': 1, 'EDUCATION': 2, 'MEDICAL': 3, 'VENTURE': 4, 'HOMEIMPROVEMENT': 5,
                                                  'DEBTCONSOLIDATION': 6})
    data['loan_grade'] = data['loan_grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

    # Normalize numerical features
    num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length']
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    return data

# Convert the processed data to a TensorFlow-friendly format
def make_inference_input_fn(data):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices(dict(data))
        dataset = dataset.batch(1)  # Batch size of 1 for inference
        return dataset
    return input_function


# Sample custom input data
custom_data = {
    'person_age': [25],
    'person_income': [5000000],
    'person_home_ownership': ['RENT'],
    'person_emp_length': [5],
    'loan_intent': ['EDUCATION'],
    'loan_grade': ['B'],
    'loan_amnt': [10000],
    'loan_percent_income': [0.2],
    'cb_person_default_on_file': ['Y'],
    'cb_person_cred_hist_length': [3]
}

model = tf.keras.models.load_model('credit_risk_model.h5')

custom_df = pd.DataFrame(custom_data)
# custom_df
dataNew = pipeline(custom_df)
custom_input_fn = make_inference_input_fn(dataNew)
predictions = model.predict(custom_input_fn())
print("Predictions are", predictions)
