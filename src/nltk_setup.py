# nltk_setup.py
import nltk
import os


def download_nltk_resources():
    """Download all necessary NLTK resources for the project."""
    print("Downloading NLTK resources...")

    # Define the path where NLTK will store its data
    nltk_data_dir = os.path.expanduser("~/nltk_data")

    # Create the directory if it doesn't exist
    os.makedirs(nltk_data_dir, exist_ok=True)

    # List of resources needed for the project
    resources = [
        "punkt",  # For sentence tokenization
        "stopwords",  # For stopword removal
        "wordnet",  # For lemmatization (if used in future)
        "omw-1.4",  # Open Multilingual WordNet (needed for wordnet)
    ]

    # Download each resource
    for resource in resources:
        try:
            nltk.download(resource, quiet=False, raise_on_error=True)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Failed to download {resource}: {e}")

    print("NLTK resource download completed")


if __name__ == "__main__":
    download_nltk_resources()
