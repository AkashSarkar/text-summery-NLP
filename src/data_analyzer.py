import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK resources are downloaded
nltk.download("punkt", quiet=True)


class DataAnalyzer:
    """Class for analyzing dataset characteristics"""

    def __init__(self, df):
        """
        Initialize the data analyzer.

        Args:
            df (pandas.DataFrame): The dataset to analyze
        """
        self.df = df
        self.stats = {}

    def compute_statistics(self):
        """Compute various statistics about the dataset"""
        # Check for missing values
        self.stats["missing_values"] = self.df.isnull().sum()
        self.stats["total_samples"] = len(self.df)

        # Calculate text statistics if 'text' column exists
        if "text" in self.df.columns:
            self.df["text_length"] = self.df["text"].apply(
                lambda x: len(x) if isinstance(x, str) else 0
            )
            self.df["sentence_count"] = self.df["text"].apply(
                lambda x: len(sent_tokenize(x)) if isinstance(x, str) else 0
            )
            self.df["word_count"] = self.df["text"].apply(
                lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0
            )

            self.stats["avg_text_length"] = self.df["text_length"].mean()
            self.stats["avg_sentence_count"] = self.df["sentence_count"].mean()
            self.stats["avg_word_count"] = self.df["word_count"].mean()

        # Calculate summary statistics if 'summary' column exists
        if "summary" in self.df.columns:
            self.df["summary_length"] = self.df["summary"].apply(
                lambda x: len(x) if isinstance(x, str) else 0
            )
            self.df["summary_sentence_count"] = self.df["summary"].apply(
                lambda x: len(sent_tokenize(x)) if isinstance(x, str) else 0
            )
            self.df["summary_word_count"] = self.df["summary"].apply(
                lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0
            )

            self.stats["avg_summary_length"] = self.df["summary_length"].mean()
            self.stats["avg_summary_sentence_count"] = self.df[
                "summary_sentence_count"
            ].mean()
            self.stats["avg_summary_word_count"] = self.df["summary_word_count"].mean()

            # Calculate compression ratio
            if "text_length" in self.df.columns:
                self.df["compression_ratio"] = (
                    self.df["summary_length"] / self.df["text_length"]
                )
                self.stats["avg_compression_ratio"] = self.df[
                    "compression_ratio"
                ].mean()

        # Category distribution if available
        if "category" in self.df.columns:
            self.stats["category_distribution"] = self.df["category"].value_counts()

        return self.stats

    def print_statistics(self):
        """Print the computed statistics in a readable format"""
        if not self.stats:
            self.compute_statistics()

        print("\nDataset Analysis:")
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Missing values:\n{self.stats['missing_values']}")

        if "avg_text_length" in self.stats:
            print(f"\nText statistics:")
            print(
                f"Average text length (characters): {self.stats['avg_text_length']:.2f}"
            )
            print(f"Average sentence count: {self.stats['avg_sentence_count']:.2f}")
            print(f"Average word count: {self.stats['avg_word_count']:.2f}")

        if "avg_summary_length" in self.stats:
            print(f"\nSummary statistics:")
            print(
                f"Average summary length (characters): {self.stats['avg_summary_length']:.2f}"
            )
            print(
                f"Average summary sentence count: {self.stats['avg_summary_sentence_count']:.2f}"
            )
            print(
                f"Average summary word count: {self.stats['avg_summary_word_count']:.2f}"
            )

            if "avg_compression_ratio" in self.stats:
                print(
                    f"Average compression ratio: {self.stats['avg_compression_ratio']:.2f}"
                )

        if "category_distribution" in self.stats:
            print("\nCategory distribution:")
            print(self.stats["category_distribution"])

    def get_statistics(self):
        """Return the computed statistics dictionary"""
        if not self.stats:
            self.compute_statistics()
        return self.stats
