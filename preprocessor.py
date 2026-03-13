import spacy
import re
from nltk.corpus import stopwords

# Load spaCy model for lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Remove HTML tags and non-alphabetical characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization and Lemmatization with spaCy
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
    
    return " ".join(tokens)
