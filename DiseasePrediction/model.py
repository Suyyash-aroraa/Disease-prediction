import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import stats

class DiseasePredictor:
    def __init__(self):
        self.svm_model = SVC(probability=True)  # Enable probability estimates
        self.nb_model = GaussianNB()
        self.rf_model = RandomForestClassifier(random_state=42)
        self.encoder = LabelEncoder()
        self.symptom_index = {}
        self.predictions_classes = []
        self.symptom_columns = []

    def train(self, training_data_path):
        # Read the training data
        data = pd.read_csv(training_data_path).dropna(axis=1)

        # Get symptoms list and create index
        self.symptom_columns = data.columns[:-1].tolist()
        for index, symptom in enumerate(self.symptom_columns):
            # Convert underscores to spaces and capitalize words
            formatted_symptom = " ".join(word.capitalize() for word in symptom.split("_"))
            self.symptom_index[formatted_symptom] = index

        # Encode the target variable
        data["prognosis"] = self.encoder.fit_transform(data["prognosis"])
        self.predictions_classes = self.encoder.classes_

        # Split features and target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Train the models
        self.svm_model.fit(X, y)
        self.nb_model.fit(X, y)
        self.rf_model.fit(X, y)

    def predict(self, symptoms):
        if not symptoms:
            return "No symptoms provided", 0.0

        # Create input vector
        input_vector = np.zeros(len(self.symptom_columns))
        for symptom in symptoms:
            if symptom in self.symptom_index:
                input_vector[self.symptom_index[symptom]] = 1

        # Get probability predictions from all models
        svm_proba = self.svm_model.predict_proba([input_vector])
        nb_proba = self.nb_model.predict_proba([input_vector])
        rf_proba = self.rf_model.predict_proba([input_vector])

        # Average the probabilities
        avg_proba = (svm_proba + nb_proba + rf_proba) / 3
        max_prob_idx = np.argmax(avg_proba)

        # Get the predicted disease and confidence
        disease = self.encoder.inverse_transform([max_prob_idx])[0]
        original_confidence = float(avg_proba[0][max_prob_idx] * 100)

        # Increase confidence by 20%, but cap at 100%
        adjusted_confidence = min(original_confidence * 1.2, 100)

        return disease, adjusted_confidence