from data_analyzer import DataAnalyzer
from text_processor import TextPreprocessor
import pandas as pd


class DataProcessor:
    """Main class for dataset processing operations"""

    def __init__(self, file_path=None):
        """
        Initialize the data processor.

        Args:
            file_path (str, optional): Path to the dataset file
        """
        self.file_path = file_path
        self.df = None
        self.processed_df = None
        self.preprocessor = None
        self.analyzer = None

    def load_data(self, file_path=None):
        """
        Load the dataset from a file.

        Args:
            file_path (str, optional): Path to the dataset file.
                                      If not provided, uses the path from initialization.

        Returns:
            pandas.DataFrame: The loaded dataset
        """
        if file_path:
            self.file_path = file_path

        if not self.file_path:
            raise ValueError("File path not provided")

        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully with shape: {self.df.shape}")
            self.analyzer = DataAnalyzer(self.df)
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def preprocess_data(self, remove_special_chars=False, remove_stopwords=False):
        """
        Preprocess the dataset.

        Args:
            remove_special_chars (bool): Whether to remove special characters
            remove_stopwords (bool): Whether to remove stopwords

        Returns:
            pandas.DataFrame: The preprocessed dataset
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        # Initialize the preprocessor
        self.preprocessor = TextPreprocessor(
            remove_special_chars=remove_special_chars, remove_stopwords=remove_stopwords
        )

        # Create a copy to avoid modifying the original
        self.processed_df = self.df.copy()

        # Preprocess text and summary if available
        if "text" in self.processed_df.columns:
            self.processed_df["clean_text"] = self.processed_df["text"].apply(
                self.preprocessor.preprocess
            )

        if "summary" in self.processed_df.columns:
            self.processed_df["clean_summary"] = self.processed_df["summary"].apply(
                self.preprocessor.preprocess
            )

        # Update the analyzer with the processed data
        self.analyzer = DataAnalyzer(self.processed_df)

        return self.processed_df

    def analyze_data(self):
        """
        Analyze the dataset and print statistics.

        Returns:
            dict: Statistics about the dataset
        """
        if self.analyzer is None:
            if self.processed_df is not None:
                self.analyzer = DataAnalyzer(self.processed_df)
            elif self.df is not None:
                self.analyzer = DataAnalyzer(self.df)
            else:
                raise ValueError("Dataset not loaded. Call load_data() first.")

        self.analyzer.print_statistics()
        return self.analyzer.get_statistics()

    def save_processed_data(self, output_path):
        """
        Save the processed dataset to a file.

        Args:
            output_path (str): Path where to save the processed dataset
        """
        if self.processed_df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        self.processed_df.to_csv(output_path, index=False)
        print(f"Processed data saved to '{output_path}'")

    def get_sample(self, index=0):
        """
        Get a sample from the dataset for inspection.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            pandas.Series: The sample at the specified index
        """
        df = self.processed_df if self.processed_df is not None else self.df

        if df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        if index >= len(df):
            raise IndexError(
                f"Index {index} out of bounds for dataset with {len(df)} samples"
            )

        return df.iloc[index]

    def print_sample(self, index=0, max_chars=200):
        """
        Print a sample from the dataset in a readable format.

        Args:
            index (int): Index of the sample to print
            max_chars (int): Maximum number of characters to print for text fields
        """
        sample = self.get_sample(index)

        print(f"\nSample #{index}:")

        if "text" in sample:
            print("\nOriginal text (first", max_chars, "chars):")
            print(
                sample["text"][:max_chars] + "..."
                if len(sample["text"]) > max_chars
                else sample["text"]
            )

        if "clean_text" in sample:
            print("\nCleaned text (first", max_chars, "chars):")
            print(
                sample["clean_text"][:max_chars] + "..."
                if len(sample["clean_text"]) > max_chars
                else sample["clean_text"]
            )

        if "summary" in sample:
            print("\nOriginal summary:")
            print(sample["summary"])

        if "clean_summary" in sample:
            print("\nCleaned summary:")
            print(sample["clean_summary"])
