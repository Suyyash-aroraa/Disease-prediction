
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import stats

class DiseasePredictor:
    def __init__(self):
        self.svm_model = SVC(probability=True, kernel='rbf', C=1.0)
        self.nb_model = GaussianNB()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.encoder = LabelEncoder()
        self.symptom_index = {}
        self.predictions_classes = []
        self.symptom_columns = []
        # Model weights for ensemble
        self.weights = {
            'svm': 0.3,
            'nb': 0.2,
            'rf': 0.25,
            'gb': 0.25
        }

    def train(self, training_data_path):
        # Read the training data
        data = pd.read_csv(training_data_path).dropna(axis=1)

        # Get symptoms list and create index
        self.symptom_columns = data.columns[:-1].tolist()
        for index, symptom in enumerate(self.symptom_columns):
            formatted_symptom = " ".join(word.capitalize() for word in symptom.split("_"))
            self.symptom_index[formatted_symptom] = index

        # Encode the target variable
        data["prognosis"] = self.encoder.fit_transform(data["prognosis"])
        self.predictions_classes = self.encoder.classes_

        # Split features and target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Train all models
        self.svm_model.fit(X, y)
        self.nb_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.gb_model.fit(X, y)

    def predict(self, symptoms):
        if not symptoms:
            return [], []

        # Create input vector
        input_vector = np.zeros(len(self.symptom_columns))
        for symptom in symptoms:
            if symptom in self.symptom_index:
                input_vector[self.symptom_index[symptom]] = 1

        # Get probability predictions from all models
        svm_proba = self.svm_model.predict_proba([input_vector])
        nb_proba = self.nb_model.predict_proba([input_vector])
        rf_proba = self.rf_model.predict_proba([input_vector])
        gb_proba = self.gb_model.predict_proba([input_vector])

        # Weighted ensemble prediction
        weighted_proba = (
            svm_proba * self.weights['svm'] +
            nb_proba * self.weights['nb'] +
            rf_proba * self.weights['rf'] +
            gb_proba * self.weights['gb']
        )
        
        # Get top 3 predictions
        top_3_indices = weighted_proba[0].argsort()[-3:][::-1]
        diseases = []
        confidences = []
        
        for idx in top_3_indices:
            disease = self.encoder.inverse_transform([idx])[0]
            # Calculate weighted confidence
            confidence = float(weighted_proba[0][idx] * 100)
            # Boost confidence if multiple models agree
            agreement_boost = 1.0
            if (svm_proba[0][idx] > 0.3 and rf_proba[0][idx] > 0.3 and 
                nb_proba[0][idx] > 0.3 and gb_proba[0][idx] > 0.3):
                agreement_boost = 1.2
            
            adjusted_confidence = min(confidence * agreement_boost, 100)
            diseases.append(disease)
            confidences.append(adjusted_confidence)
            
        return diseases, confidences
