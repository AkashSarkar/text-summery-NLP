import DataProcessor


def main():
    # Initialize the data processor
    processor = DataProcessor()

    # Load the BBC News Summary dataset
    processor.load_data("bbc-summary.csv")

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
