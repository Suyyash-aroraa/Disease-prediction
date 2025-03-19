import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from difflib import get_close_matches

class SymptomExtractor:
    def __init__(self):
        # Download all required NLTK data with error handling
        required_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' 
                              else f'corpora/{package}' if package in ['stopwords', 'wordnet']
                              else package)
            except LookupError:
                print(f"Downloading {package}...")
                nltk.download(package, quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"Error in tokenization: {e}")
            tokens = text.split()  # Fallback to simple splitting

        # Remove punctuation and stopwords
        tokens = [token for token in tokens 
                 if token not in string.punctuation 
                 and token not in self.stop_words]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def extract_symptoms(self, text, symptom_index):
        tokens = self.preprocess_text(text)

        # Convert symptom index keys to lowercase for matching
        symptom_lookup = {symptom.lower(): symptom for symptom in symptom_index}

        # Extract symptoms using fuzzy matching
        extracted_symptoms = set()

        # First try exact matches
        text_lower = text.lower()
        for symptom_lower, original_symptom in symptom_lookup.items():
            if symptom_lower in text_lower:
                extracted_symptoms.add(original_symptom)

        # Then try token-based matching
        remaining_tokens = " ".join(tokens)
        for symptom_lower, original_symptom in symptom_lookup.items():
            symptom_tokens = self.preprocess_text(symptom_lower)
            if all(token in remaining_tokens for token in symptom_tokens):
                extracted_symptoms.add(original_symptom)

        # Finally try fuzzy matching for unmatched tokens
        if not extracted_symptoms:
            for token in tokens:
                matches = get_close_matches(token, symptom_lookup.keys(), n=1, cutoff=0.8)
                if matches:
                    extracted_symptoms.add(symptom_lookup[matches[0]])

        return list(extracted_symptoms)