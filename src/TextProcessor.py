import pandas as pd
import nltk
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class TextPreprocessor:
    def __init__(self, remove_special_chars=False, remove_stopwords=False):
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(stopwords.words("english")) if remove_stopwords else None

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()

        if self.remove_special_chars:
            text = re.sub(r"[^\w\s]", "", text)

        # remove extra whitespace
        text = " ".join(text.split())

        # remove stopwords if enabled
        if self.remove_stopwords:
            words = word_tokenize(text)
            text = " ".join([word for word in words if word not in self.stopwords])

        return text
