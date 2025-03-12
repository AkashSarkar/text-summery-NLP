from data_processor import DataProcessor
import os
from datasets import load_dataset
import pandas as pd

import nltk_setup


def download_bbc_news_dataset():
    """
    Download BBC News dataset from HuggingFace or an alternative source.
    Returns the path to the saved CSV file.
    """
    print("Downloading BBC News Summary dataset...")

    try:
        # Try to load from HuggingFace - this is just an example as BBC News Summary isn't directly available
        # For this example, we'll use a different dataset and adapt it
        cnn_dataset = load_dataset(
            "cnn_dailymail", "3.0.0", split="train[:100]"
        )  # Just load 100 samples for testing

        # Convert to DataFrame
        data = {
            "text": [item["article"] for item in cnn_dataset],
            "summary": [item["highlights"] for item in cnn_dataset],
        }
        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "news-summary.csv"
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

        print(f"Dataset saved to {csv_path}")
        return csv_path

    except Exception as e:
        print(f"Error downloading dataset from HuggingFace: {e}")

        # Alternative: create a sample dataset for testing
        print("Creating a sample dataset instead...")

        # Create a minimal sample dataset
        data = {
            "text": [
                "The BBC is facing a major crisis as the TV license fee comes under scrutiny. The government has frozen the license fee for two years and is considering its future. The BBC says this will impact its programming and services.",
                "Scientists have discovered a new species of deep-sea fish that can survive extreme pressure. The fish was found at a depth of over 8,000 meters in the Mariana Trench. Researchers say this discovery could help in understanding adaptation to extreme environments.",
            ],
            "summary": [
                "BBC facing crisis as TV license fee is frozen for two years with future under consideration.",
                "New deep-sea fish species discovered in Mariana Trench that can survive extreme pressure.",
            ],
        }
        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample-news-summary.csv"
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

        print(f"Sample dataset saved to {csv_path}")
        return csv_path


def main():
    # First, ensure NLTK resources are downloaded
    nltk_setup.download_nltk_resources()

    # Download or create dataset
    data_path = download_bbc_news_dataset()

    # Initialize the data processor
    processor = DataProcessor()

    # Load the BBC News Summary dataset
    processor.load_data(data_path)

    # Preprocess the data
    processor.preprocess_data(remove_special_chars=False, remove_stopwords=False)

    # Analyze the dataset
    processor.analyze_data()

    # Print a sample
    processor.print_sample(index=0)

    # Save the processed data
    processor.save_processed_data("processed_bbc_summary.csv")


if __name__ == "__main__":
    main()
