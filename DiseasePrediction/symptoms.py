import pandas as pd

def get_all_symptoms():
    data = pd.read_csv("Training.csv")
    symptoms = data.columns[:-1]  # Exclude the last column (prognosis)
    return [" ".join(s.split("_")).title() for s in symptoms]

AVAILABLE_SYMPTOMS = get_all_symptoms()

COMMON_SYMPTOMS = [
    "Fever", "Cough", "Fatigue", "Shortness of Breath", "Headache",
    "Chest Pain", "Body Aches", "Nausea", "Diarrhea", "Loss of Smell",
    "Sore Throat", "Runny Nose", "Muscle Pain", "Loss of Appetite",
    "Chills", "Dizziness", "Vomiting", "Abdominal Pain"
]