import streamlit as st
from model import DiseasePredictor
from nlp_processor import SymptomExtractor
from symptoms import AVAILABLE_SYMPTOMS

# Initialize the disease predictor and symptom extractor
@st.cache_resource
def load_models():
    predictor = DiseasePredictor()
    predictor.train("Training.csv")
    extractor = SymptomExtractor()
    return predictor, extractor

def main():
    st.title("Disease Prediction System")

    # Load models
    predictor, extractor = load_models()

    # Initialize session state for symptoms
    if 'confirmed_symptoms' not in st.session_state:
        st.session_state.confirmed_symptoms = set()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input with symptom suggestions
    prompt = st.chat_input("Describe your symptoms")

    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Extract symptoms from user input
        extracted_symptoms = extractor.extract_symptoms(prompt, predictor.symptom_index)

        # Find suggested symptoms based on input
        words = prompt.lower().split()
        suggested_symptoms = [
            symptom for symptom in predictor.symptom_index.keys()
            if any(word in symptom.lower() for word in words)
            and symptom not in extracted_symptoms
        ]

        # Assistant response
        with st.chat_message("assistant"):
            if not extracted_symptoms and not suggested_symptoms:
                st.write("I couldn't identify any specific symptoms. Please try being more specific.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't identify any specific symptoms. Please try being more specific."
                })
            else:
                # Add extracted symptoms to confirmed symptoms
                st.session_state.confirmed_symptoms.update(extracted_symptoms)

                # Show identified symptoms
                if extracted_symptoms:
                    response = "I identified the following symptoms:\n" + "\n".join([f"- {s}" for s in extracted_symptoms])
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                # Show symptom suggestions if any
                if suggested_symptoms:
                    selected_suggestions = st.multiselect(
                        "Did you also mean any of these symptoms?",
                        suggested_symptoms,
                        key="symptom_suggestions"
                    )
                    if selected_suggestions:
                        st.session_state.confirmed_symptoms.update(selected_suggestions)

                # Make prediction with current symptoms
                if st.session_state.confirmed_symptoms:
                    disease, confidence = predictor.predict(list(st.session_state.confirmed_symptoms))
                    result = f"\nPrediction: {disease} (Confidence: {confidence:.1f}%)"
                    st.write(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})

                    if confidence < 50:
                        st.write("Note: The confidence is low. Please consult a healthcare professional for accurate diagnosis.")

    # Add a clear button
    if st.button("Clear All"):
        st.session_state.confirmed_symptoms.clear()
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()