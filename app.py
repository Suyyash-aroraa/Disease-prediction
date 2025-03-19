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
    # Set page config for favicon
    st.set_page_config(
        page_title="Disease Prediction System",
        page_icon="generated-icon.png"
    )
    
    # Add logo to top-left
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("generated-icon.png", width=80)
    with col2:
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

        # Assistant response
        with st.chat_message("assistant"):
            if not extracted_symptoms:
                st.write("I couldn't identify any specific symptoms. Please try being more specific.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't identify any specific symptoms. Please try being more specific."
                })
            else:
                # Add extracted symptoms to confirmed symptoms
                st.session_state.confirmed_symptoms.update(extracted_symptoms)

                # Show identified symptoms
                response = "I identified the following symptoms:\n" + "\n".join([f"- {s}" for s in extracted_symptoms])
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Make prediction with current symptoms if at least 3 are present
                if len(st.session_state.confirmed_symptoms) >= 3:
                    diseases, confidences = predictor.predict(list(st.session_state.confirmed_symptoms))
                    result = "\nTop 3 Predictions:"
                    from prescriptions import PRESCRIPTIONS
                    
                    # Collect all medicines for predicted diseases
                    all_medicines = {}
                    for disease, confidence in zip(diseases, confidences):
                        if disease in PRESCRIPTIONS:
                            for medicine in PRESCRIPTIONS[disease]:
                                if medicine not in all_medicines:
                                    all_medicines[medicine] = []
                                all_medicines[medicine].append(disease)
                    
                    for i, (disease, confidence) in enumerate(zip(diseases, confidences), 1):
                        result += f"\n{i}. {disease} (Confidence: {confidence:.1f}%)"
                        if disease in PRESCRIPTIONS:
                            result += f"\n   Recommended medications: {', '.join(PRESCRIPTIONS[disease])}"
                    
                    st.write(result)
                    
                    # Display common medicines only
                    st.write("\n🏥 Common Medicine Recommendations:")
                    common_medicines = [medicine for medicine, diseases in all_medicines.items() if len(diseases) > 1]
                    if common_medicines:
                        for medicine in common_medicines:
                            st.write(f"- {medicine}")
                    
                    st.write("\n⚠️ Note: These are general recommendations. Please consult a healthcare professional for proper diagnosis and treatment.")
                    st.session_state.messages.append({"role": "assistant", "content": result})

                    if confidences[0] < 50:
                        st.write("Note: The confidence is low. Please consult a healthcare professional for accurate diagnosis.")
                else:
                    remaining = 3 - len(st.session_state.confirmed_symptoms)
                    st.write(f"Please describe {remaining} more symptom{'s' if remaining > 1 else ''} for a prediction.")

    # Add a clear button
    if st.button("Clear All"):
        st.session_state.confirmed_symptoms.clear()
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()